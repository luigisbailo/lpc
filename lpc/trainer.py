import importlib
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Subset
from scipy.special import softmax
import random

from lpc.utils import (
    load_checkpoint,
    save_checkpoint,
    gather_dict_outputs_ddp,
)
from lpc.losses import (
    arcface_loss,
    cosface_loss,
    supervised_contrastive_loss,
)
from lpc.metrics import (
    get_entropy,
    get_coeff_var,
    get_distance_margins,
    get_binarity_metrics,
    get_collapse_metrics,
    estimate_lipschitz_gradient_norm_stream, 
)
from lpc.deepfool import deepfool


class Trainer:
    """Trainer for a neural‑network classifier."""

    def __init__(
        self,
        network,
        architecture,
        trainset,
        testset,
        training_hypers,
        architecture_type,
        l2_loss=False,
        cosface_loss=False,
        arcface_loss=False,
        scl=False,
        store_penultimate=False,
        verbose=True,
    ):
        self.network = network
        self.architecture = architecture
        self.trainset = trainset
        self.testset = testset
        self.training_hypers = training_hypers
        self.architecture_type = architecture_type
        self.l2_loss = l2_loss
        self.cosface = cosface_loss
        self.arcface = arcface_loss
        self.scl = scl
        self.store_penultimate = store_penultimate
        self.verbose = verbose
        self.rank = None
        self.world_size = None
        self.distributed = False

    # ---------------------------------------------------------------------
    #  Public interface
    # ---------------------------------------------------------------------

    def fit(self, checkpoint_file, rank, world_size, distributed):
        """Run training and return aggregated results (rank‑0 only)."""
        self.rank = rank
        self.world_size = world_size
        self.distributed = distributed

        torch.cuda.set_device(rank)
        self._prepare_model()
        self._setup_optimizer()

        start_epoch, gamma, converged = self._load_checkpoint(checkpoint_file)
        if start_epoch >= self.training_hypers["total_epochs"]:
            print("Last epoch already reached. Exiting")
            return None

        last_train_epoch = start_epoch + self.training_hypers["train_epochs"]

        # data loader
        num_workers = self.world_size * 4
        self.train_sampler = DistributedSampler(
            self.trainset, num_replicas=world_size, rank=rank
        )
        trainloader = DataLoader(
            self.trainset,
            batch_size=self.training_hypers["batch_size"],
            sampler=self.train_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )

        # bookkeeping
        res_list = []
        epoch_training_time_list = []
        start_training_time = time.time()

        # convenient aliases for γ scheduling
        sched_type = self.training_hypers.get("gamma_scheduler_type", "exponential")
        gamma_min = self.training_hypers["gamma"]
        gamma_max = 10 ** self.training_hypers["gamma_max_exp"]
        T = self.training_hypers.get("gamma_scheduler_T", 1)
        t0 = self.training_hypers.get("gamma_scheduler_init", 0)

        # -----------------------------------------------------------------
        #  Main epoch loop
        # -----------------------------------------------------------------
        for epoch in range(start_epoch + 1, last_train_epoch + 1):
            # update γ **before** the forward pass
            if sched_type == "cosine":
                if epoch >= t0:
                    t = min(epoch - t0, T)
                    alpha = 0.5 * (1 - math.cos(math.pi * t / T))
                    gamma = gamma_min + (gamma_max - gamma_min) * alpha
                else:
                    gamma = gamma_min
            elif sched_type == "linear":             
                if epoch >= t0:
                    t = min(epoch - t0, T)
                    alpha = t / T
                    gamma = gamma_min + (gamma_max - gamma_min) * alpha
                else:
                    gamma = gamma_min
            else:  
                if (
                    epoch % self.training_hypers["gamma_scheduler_step"] == 0
                    and gamma < gamma_max
                    and epoch >= self.training_hypers["gamma_scheduler_init"]
                ):
                    gamma *= self.training_hypers["gamma_scheduler_factor"]

            # training phase ------------------------------------------------
            self.train_sampler.set_epoch(epoch)
            epoch_time = self._train_epoch(epoch, trainloader, gamma)
            epoch_training_time_list.append(epoch_time)

            # LR scheduler (unchanged) -------------------------------------
            if epoch >= self.training_hypers["lr_scheduler_start"]:
                self.scheduler.step()
                self.opt.param_groups[1]["lr"] = self.training_hypers["lr"]

            # evaluation & logging every n epochs or at the very end --------
            is_last_epoch = epoch == last_train_epoch
            if is_last_epoch or epoch % self.training_hypers["logging"] == 0:
                res_epoch = self._evaluate_and_log(epoch, gamma, is_last_epoch)
                if res_epoch is not None:
                    res_list.append(res_epoch)

        # -----------------------------------------------------------------
        #  Wrap‑up
        # -----------------------------------------------------------------
        elapsed_training_time = time.time() - start_training_time
        if self.rank == 0:
            res_dict_stack = self._aggregate_results(
                res_list, elapsed_training_time, epoch_training_time_list, converged
            )
            if checkpoint_file:
                self._save_checkpoint(checkpoint_file, epoch, gamma, converged)
            return res_dict_stack
        return None

    # ---------------------------------------------------------------------
    #  Initialisation helpers
    # ---------------------------------------------------------------------

    def _prepare_model(self):
        """Put model on the current GPU and wrap in DDP if necessary."""
        self.model = self.network.to(self.rank)
        self.model_ddp = (
            DDP(self.model, device_ids=[self.rank]) if self.world_size > 1 else self.model
        )

    def _setup_optimizer(self):
        """Create optimiser and its LR scheduler (cosine for learning‑rate)."""
        if self.world_size > 1:
            excluded_params = set(self.model_ddp.module.output_layer.parameters())
        else:
            excluded_params = set(self.model_ddp.output_layer.parameters())

        other_params = [p for p in self.model_ddp.parameters() if p not in excluded_params]
        params_to_update = [
            {"params": other_params, "lr": self.training_hypers["lr"]},
            {"params": list(excluded_params), "lr": self.training_hypers["lr"]},
        ]
        torch_optim_module = importlib.import_module("torch.optim")
        self.opt = getattr(torch_optim_module, self.training_hypers["optimizer"])(
            params_to_update, weight_decay=self.training_hypers["weight_decay"]
        )

        start_schedule_epoch = self.training_hypers.get("lr_scheduler_start", 0)
        total_epochs = self.training_hypers["total_epochs"]
        T_max_epochs = max(1, total_epochs - start_schedule_epoch)
        self.scheduler = CosineAnnealingLR(
            self.opt,
            T_max=T_max_epochs,
            eta_min=self.training_hypers.get("lr_min", 0),
        )

    # ---------------------------------------------------------------------
    #  Check‑pointing
    # ---------------------------------------------------------------------

    def _load_checkpoint(self, checkpoint_file):
        start_epoch = 0
        gamma = self.training_hypers["gamma"]
        converged = False
        checkpoint_loaded = False

        if self.rank == 0:
            try:
                checkpoint = load_checkpoint(checkpoint_file)
                self.model_ddp.load_state_dict(checkpoint["state_dict"])
                self.opt.load_state_dict(checkpoint["optimizer"])
                self.scheduler.load_state_dict(checkpoint["scheduler"])
                start_epoch = checkpoint["epoch"]
                gamma = checkpoint["gamma"]
                converged = checkpoint.get("converged", False)
                checkpoint_loaded = True
                print(f"Resuming from checkpoint at epoch {start_epoch}")
            except FileNotFoundError:
                print("No checkpoint found – starting from scratch")

        checkpoint_loaded_tensor = torch.tensor(checkpoint_loaded).to(self.rank)
        if self.distributed:
            dist.broadcast(checkpoint_loaded_tensor, src=0)
        if checkpoint_loaded_tensor.item():
            objs = [
                self.model_ddp.state_dict(),
                self.opt.state_dict(),
                self.scheduler.state_dict(),
                start_epoch,
                gamma,
                converged,
            ]
            if self.distributed:
                dist.broadcast_object_list(objs, src=0)
            self.model_ddp.load_state_dict(objs[0])
            self.opt.load_state_dict(objs[1])
            self.scheduler.load_state_dict(objs[2])
            start_epoch, gamma, converged = objs[3:6]
        return start_epoch, gamma, converged

    def _save_checkpoint(self, filename, epoch, gamma, converged):
        checkpoint = {
            "epoch": epoch,
            "gamma": gamma,
            "state_dict": self.model_ddp.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "converged": converged,
        }
        print("Checkpoint file:", filename)
        save_checkpoint(checkpoint, filename)

    def _train_epoch(self, epoch, trainloader, gamma):
        epoch_start = time.time()
        self.model_ddp.train()

        for x_input_batch, y_batch in trainloader:
            x_input_batch = x_input_batch.to(self.rank)
            y_batch = y_batch.to(self.rank)
            self.opt.zero_grad(set_to_none=True)
            output_dict = self.model_ddp(x_input_batch)
            loss = self._compute_loss(output_dict, y_batch, epoch, gamma)
            loss.backward()
            self.opt.step()

        return (time.time() - epoch_start) / 60  # minutes

    def _compute_loss(self, output_dict, y_batch, epoch, gamma):
        x_output = output_dict["x_output"]
        x_penultimate = output_dict["x_penultimate"]
        x_backbone = output_dict["x_backbone"]

        loss = nn.CrossEntropyLoss()(x_output, y_batch).to(self.rank)

        if self.l2_loss:
            loss += F.mse_loss(
                x_penultimate, torch.zeros_like(x_penultimate), reduction="mean"
            ) * gamma

        if self.scl:
            x_for_scl = x_backbone if self.architecture_type in {"lin_pen", "nonlin_pen"} else x_penultimate
            loss += supervised_contrastive_loss(x_for_scl, y_batch, device=self.rank)

        if self.arcface:
            scale = min(16.0 + (epoch - 1) * 1.0, 64.0)
            margin = min(0.1 + (epoch - 1) * 0.01, 0.5)
            loss = arcface_loss(
                features=x_penultimate,
                labels=y_batch,
                weight=self.model_ddp.output_layer.weight,
                device=self.rank,
                margin=margin,
                scale=scale,
            )

        if self.cosface:
            loss = cosface_loss(
                features=x_penultimate,
                labels=y_batch,
                weight=self.model_ddp.output_layer.weight,
                device=self.rank,
            )

        return loss

    def _evaluate_and_log(self, epoch, gamma, is_last_epoch):
        if self.distributed:
            dist.barrier()
        num_workers = self.world_size * 4

        res_epoch = {}
        eval_trainloader = DataLoader(
            self.trainset,
            batch_size=self.training_hypers['batch_size'],
            sampler=self.train_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        eval_train = self.eval(eval_trainloader, self.rank)

        # Add debug info
        if self.rank == 0:
            print(f"Rank {self.rank}: eval_train keys: {list(eval_train.keys())}")
            for key, val in eval_train.items():
                if isinstance(val, np.ndarray):
                    print(f"Rank {self.rank}: {key} shape: {val.shape}, contains_nan: {np.isnan(val).any()}")

        if self.distributed:
            # Synchronize before gathering data
            dist.barrier()
            print(f"Rank {self.rank}: Before gathering")
            try:
                gathered_eval_train = gather_dict_outputs_ddp(eval_train, self.rank, self.world_size)
                print(f"Rank {self.rank}: After gathering")
            except Exception as e:
                print(f"Rank {self.rank}: Exception during gathering: {e}")
                gathered_eval_train = eval_train if self.rank == 0 else None
            # Ensure all processes wait after gathering
            dist.barrier()
        else:
            gathered_eval_train = eval_train
        if self.rank == 0:
            eval_testloader = DataLoader(
                self.testset,
                batch_size= self.training_hypers['batch_size'],
                num_workers=num_workers,
                pin_memory=True,
            )
            eval_test = self.eval(eval_testloader, self.rank)

            res_epoch['epochs'] = epoch
            res_epoch['accuracy_train'] = (gathered_eval_train['y_predicted'] == gathered_eval_train['y_label']).mean()
            res_epoch['accuracy_test'] = (eval_test['y_predicted'] == eval_test['y_label']).mean()
            res_epoch['top1_accuracy_train'] = gathered_eval_train['top1_accuracy']
            res_epoch['top1_accuracy_test'] = eval_test['top1_accuracy']
            res_epoch['top5_accuracy_train'] = gathered_eval_train['top5_accuracy']
            res_epoch['top5_accuracy_test'] = eval_test['top5_accuracy']

            if self.verbose:
                print(f"Epoch {epoch}")
                print(f"gamma: {gamma}")
                print(f"Accuracy train: {np.around(res_epoch['accuracy_train'], 4)}\tAccuracy test: {np.around(res_epoch['accuracy_test'], 4)}")

            start_time = time.time()
            res_epoch['collapse_train'] = get_collapse_metrics(gathered_eval_train)
            res_epoch['collapse_test'] = get_collapse_metrics(eval_test)
            print(f"Elapsed time to compute collapse metrics: {(time.time()-start_time)/60:.2f} minutes")

            start_time = time.time()
            res_epoch['distance_margins_train'] = get_distance_margins(gathered_eval_train)
            res_epoch['distance_margins_test'] = get_distance_margins(eval_test)
            print(f"Elapsed time to compute distance margins: {(time.time()-start_time)/60:.2f} minutes")

            start_time = time.time()
            coeff_var_norm_train, coeff_var_abs_train, norm_mean_train = get_coeff_var(gathered_eval_train)
            coeff_var_norm_test, coeff_var_abs_test, norm_mean_test = get_coeff_var(eval_test)
            res_epoch['coeff_var_abs_train'] = coeff_var_abs_train
            res_epoch['coeff_var_abs_test'] = coeff_var_abs_test
            res_epoch['coeff_var_norm_train'] = coeff_var_norm_train
            res_epoch['coeff_var_norm_test'] = coeff_var_norm_test
            res_epoch['norm_mean_train'] = norm_mean_train
            res_epoch['norm_mean_test'] = norm_mean_test
            print(f"Elapsed time to compute coefficient of variation: {(time.time()-start_time)/60:.2f} minutes")

            # Compute DeepFool score and entropy only on the last epoch.
            if is_last_epoch:
               
                start_time = time.time()
                subsample_size = 1000
                test_indices = random.sample(range(len(self.testset)), min(subsample_size, len(self.testset)))
                subsampled_testset = Subset(self.testset, test_indices)
                train_indices = random.sample(range(len(self.trainset)), min(subsample_size, len(self.trainset)))
                subsampled_trainset = Subset(self.trainset, train_indices)

                loader_test = DataLoader(subsampled_testset, batch_size=self.training_hypers['batch_size'], shuffle=True)
                loader_train = DataLoader(subsampled_trainset, batch_size=self.training_hypers['batch_size'], shuffle=True)

                model_to_eval = self.model # Or self.model_ddp.module if still using DDP context
                res_epoch['lipschitz_estimation_train'] = estimate_lipschitz_gradient_norm_stream(
                    model_to_eval, loader_train, device=self.rank
                )
                res_epoch['lipschitz_estimation_test'] = estimate_lipschitz_gradient_norm_stream(
                    model_to_eval, loader_test, device=self.rank
                )
                print(f"Elapsed time to compute Lipschitz estimation: {(time.time() - start_time)/60:.2f} minutes")

                
                start_time = time.time()
                batch_size_deepfool = 1000
                loader = DataLoader(self.testset, batch_size=batch_size_deepfool, shuffle=True)
                images, labels = next(iter(loader))
                images = images.to(self.rank)
                labels = labels.to(self.rank)
                perturbation_list = []
                for image in images:
                    r_tot, loop_i, label, k_i, pert_image = deepfool(image, self.network)
                    perturbation_list.append(np.linalg.norm(r_tot) / np.linalg.norm(image.cpu().numpy()))
                res_epoch['deepfool_score'] = np.mean(perturbation_list)
                print(f"Elapsed time to compute DeepFool score: {(time.time()-start_time)/60:.2f} minutes")

                start_time = time.time()
                res_epoch['entropy_train'] = get_entropy(gathered_eval_train, k=20, normalize=True)
                res_epoch['entropy_test'] = get_entropy(eval_test, k=20, normalize=True)
                print(f"Elapsed time to compute entropy: {(time.time()-start_time)/60:.2f} minutes")
                  
            else:
                res_epoch['deepfool_score'] = None
                res_epoch['entropy_train'] = None
                res_epoch['entropy_test'] = None
                res_epoch['lipschitz_estimation_train'] = None
                res_epoch['lipschitz_estimation_test']  = None

                
            if res_epoch['accuracy_train'] > self.training_hypers['convergence_thres'] and not getattr(self, 'converged', False):
                self.converged = True
                self.convergence_epoch = epoch
                self.accuracy_test_converged = res_epoch['accuracy_test']
                print('converged!', epoch)

            if self.architecture_type == 'lin_pen':
                start_time = time.time()
                res_epoch['binarity_train'] = get_binarity_metrics(gathered_eval_train)
                res_epoch['binarity_test'] = get_binarity_metrics(eval_test)
                print(f"Elapsed time to compute binarity metrics: {(time.time()-start_time)/60:.2f} minutes")

            if self.store_penultimate and epoch == self.training_hypers['total_epochs']:
                self.penultimate_train = gathered_eval_train['x_penultimate']
                self.penultimate_test = eval_test['x_penultimate']

        if self.distributed:
            dist.barrier()

        return res_epoch if self.rank == 0 else None

    def _aggregate_results(self, res_list, total_training_time, epoch_times, converged):
        res_dict_stack = {}
        for key in res_list[0].keys():
            if isinstance(res_list[0][key], dict):
                res_dict_stack[key] = {}
                for key2 in res_list[0][key].keys():
                    if isinstance(res_list[0][key][key2], dict):
                        res_dict_stack[key][key2] = {}
                        for key3 in res_list[0][key][key2].keys():
                            res_dict_stack[key][key2][key3] = np.vstack([res_epoch[key][key2][key3] for res_epoch in res_list])
                    else:
                        res_dict_stack[key][key2] = np.vstack([res_epoch[key][key2] for res_epoch in res_list])
            else:
                res_dict_stack[key] = np.vstack([res_epoch[key] for res_epoch in res_list])

        res_dict_stack['training_time'] = np.array(total_training_time / 60)
        res_dict_stack['training_time_epochs'] = np.array(epoch_times)

        if getattr(self, 'converged', False):
            res_dict_stack['convergence_epoch'] = self.convergence_epoch
            res_dict_stack['accuracy_test_converged'] = self.accuracy_test_converged
        else:
            res_dict_stack['accuracy_test_converged'] = False
            res_dict_stack['convergence_epoch'] = False

        res_dict_stack['deepfool_score'] = res_list[-1].get('deepfool_score', 0)
        res_dict_stack['entropy_train'] = res_list[-1].get('entropy_train', 0)
        res_dict_stack['entropy_test'] = res_list[-1].get('entropy_test', 0)
        res_dict_stack['lipschitz_estimation_train'] = res_list[-1].get('lipschitz_estimation_train', 0)
        res_dict_stack['lipschitz_estimation_test'] = res_list[-1].get('lipschitz_estimation_test', 0)

        if self.store_penultimate and hasattr(self, 'penultimate_train'):
            res_dict_stack['penultimate_train'] = self.penultimate_train.reshape(1, self.penultimate_train.shape[0], -1)
            res_dict_stack['penultimate_test'] = self.penultimate_test.reshape(1, self.penultimate_test.shape[0], -1)

        return res_dict_stack

    def eval(self, loader, device):
        evaluations = {}
        self.model_ddp.eval()
        x_output_list = []
        x_penultimate_list = []
        y_label_list = []

        with torch.no_grad():
            for x_input_batch, y_label_batch in loader:
                x_input_batch = x_input_batch.to(device)
                y_label_batch = y_label_batch.to(device)

                output_dict_batch = self.network(x_input_batch)
                x_output_batch = output_dict_batch['x_output']
                x_penultimate_batch = output_dict_batch['x_penultimate']

                x_output_list.append(x_output_batch.cpu().numpy())
                x_penultimate_list.append(x_penultimate_batch.cpu().numpy())
                y_label_list.append(y_label_batch.cpu().numpy())

                torch.cuda.empty_cache()

            x_output = np.concatenate(x_output_list, axis=0)
            x_penultimate = np.concatenate(x_penultimate_list, axis=0)
            y_label = np.concatenate(y_label_list, axis=0)

            probs = softmax(x_output, axis=1)  
            y_predicted = np.argmax(probs, axis=1)

            top1_correct = np.sum(y_predicted == y_label)
            top1_accuracy = top1_correct / y_label.shape[0]

            top5_indices = np.argsort(probs, axis=1)[:, -5:]  
            top5_correct = 0
            for i, label in enumerate(y_label):
                if label in top5_indices[i]:
                    top5_correct += 1
            top5_accuracy = top5_correct / y_label.shape[0]

        evaluations.update({
            'x_output': x_output,
            'x_penultimate': x_penultimate,
            'y_predicted': y_predicted,
            'y_label': y_label,
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy,
        })

        return evaluations
