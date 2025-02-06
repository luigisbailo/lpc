import importlib
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

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
)

from lpc.deepfool import deepfool


class Trainer:
    """
    Trainer for a neural network classifier.

    Parameters:
        network (torch.nn.Module): The network model.
        architecture: Architecture details.
        trainset (Dataset): Training dataset.
        testset (Dataset): Testing dataset.
        training_hypers (dict): Training hyperparameters.
        architecture_type (str): Name/type of the architecture.
        l2_loss (bool): Use L2 loss on penultimate activations.
        cosface_loss (bool): Use CosFace loss.
        arcface_loss (bool): Use ArcFace loss.
        scl (bool): Use supervised contrastive loss.
        store_penultimate (bool): Store penultimate activations.
        verbose (bool): Print training progress.
    """
    def __init__(self, network, architecture, trainset, testset,
                 training_hypers, architecture_type,
                 l2_loss=False, cosface_loss=False, arcface_loss=False,
                 scl=False, store_penultimate=False, verbose=True):
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

    def fit(self, checkpoint_file, rank, world_size, distributed):
        """
        Train the model.

        Args:
            checkpoint_file (str): Path to a checkpoint file (if any).
            rank (int): Process rank (for GPU and DDP).
            world_size (int): Total number of processes.
            distributed (bool): Whether using distributed training.
        Returns:
            dict: Aggregated training results (only on rank 0).
        """
        self.rank = rank
        self.world_size = world_size
        self.distributed = distributed

        torch.cuda.set_device(rank)
        self._prepare_model()
        self._setup_optimizer()

        start_epoch, gamma, converged = self._load_checkpoint(checkpoint_file)
        if start_epoch >= self.training_hypers['total_epochs']:
            print('Last epoch already reached. Exiting')
            exit(0)

        last_train_epoch = start_epoch + self.training_hypers['train_epochs']

        self.train_sampler = DistributedSampler(self.trainset, num_replicas=world_size, rank=rank)
        trainloader = DataLoader(
            self.trainset,
            batch_size=self.training_hypers['batch_size'],
            sampler=self.train_sampler,
            num_workers=4,
            pin_memory=True,
        )

        res_list = []
        epoch_training_time_list = []
        start_training_time = time.time()  # Start overall training timer

        for epoch in range(start_epoch + 1, last_train_epoch + 1):
            self.train_sampler.set_epoch(epoch)
            epoch_time = self._train_epoch(epoch, trainloader, gamma)
            epoch_training_time_list.append(epoch_time)

            if (epoch % self.training_hypers['gamma_scheduler_step'] == 0 and 
                gamma < 10 ** self.training_hypers['gamma_max_exp'] and
                epoch > self.training_hypers['gamma_scheduler_init']):
                gamma *= self.training_hypers['gamma_scheduler_factor']

            if epoch > self.training_hypers['lr_scheduler_start']:
                self.scheduler.step()
                self.opt.param_groups[1]['lr'] = self.training_hypers['lr']

            # Determine if this is the last epoch.
            is_last_epoch = (epoch == last_train_epoch)
            if epoch % self.training_hypers['logging'] == 0 or is_last_epoch:
                res_epoch = self._evaluate_and_log(epoch, gamma, is_last_epoch)
                if res_epoch is not None:
                    res_list.append(res_epoch)

        elapsed_training_time = time.time() - start_training_time

        if self.rank == 0:
            res_dict_stack = self._aggregate_results(res_list, elapsed_training_time, epoch_training_time_list, converged)
            if checkpoint_file:
                self._save_checkpoint(checkpoint_file, epoch, gamma, converged)
            return res_dict_stack
        else:
            return None

    def _prepare_model(self):
        """Move model to GPU and wrap with DDP if needed."""
        self.model = self.network.to(self.rank)
        if self.world_size > 1:
            self.model_ddp = DDP(self.model, device_ids=[self.rank])
        else:
            self.model_ddp = self.model

    def _setup_optimizer(self):
        """Set up the optimizer and scheduler."""
        if self.world_size > 1:
            excluded_params = set(self.model_ddp.module.output_layer.parameters())
        else:
            excluded_params = set(self.model_ddp.output_layer.parameters())

        other_params = [param for name, param in self.model_ddp.named_parameters() if param not in excluded_params]
        excluded_params = list(excluded_params)
        params_to_update = [
            {'params': other_params, 'lr': self.training_hypers['lr']},
            {'params': excluded_params, 'lr': self.training_hypers['lr']}
        ]
        torch_optim_module = importlib.import_module("torch.optim")
        self.opt = getattr(torch_optim_module, self.training_hypers['optimizer'])(
            params_to_update,
            weight_decay=self.training_hypers['weight_decay']
        )
        self.scheduler = StepLR(
            self.opt,
            step_size=self.training_hypers['lr_scheduler_step_size'],
            gamma=self.training_hypers['lr_scheduler_gamma']
        )

    def _load_checkpoint(self, checkpoint_file):
        start_epoch = 0
        gamma = self.training_hypers['gamma']
        converged = False
        checkpoint_loaded = False

        if self.rank == 0:
            print(f"architecture name: {self.architecture_type}, l2_loss: {self.l2_loss}, "
                  f"scl: {self.scl}, arcface: {self.arcface}, "
                  f"cosface: {self.cosface}")
            try:
                checkpoint = load_checkpoint(checkpoint_file)
                self.model_ddp.load_state_dict(checkpoint['state_dict'])
                self.opt.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch']
                gamma = checkpoint['gamma']
                converged = checkpoint.get('converged', False)
                checkpoint_loaded = True
                print(f"Resuming from checkpoint at epoch {start_epoch}")
            except FileNotFoundError:
                print("No checkpoint found, starting training from scratch")

        checkpoint_loaded_tensor = torch.tensor(checkpoint_loaded).to(self.rank)
        if self.distributed:
            dist.broadcast(checkpoint_loaded_tensor, src=0)
        if checkpoint_loaded_tensor.item():
            state_dict_list = [
                self.model_ddp.state_dict(),
                self.opt.state_dict(),
                self.scheduler.state_dict(),
                start_epoch,
                gamma,
                converged
            ]
            if self.distributed:
                dist.broadcast_object_list(state_dict_list, src=0)
            self.model_ddp.load_state_dict(state_dict_list[0])
            self.opt.load_state_dict(state_dict_list[1])
            self.scheduler.load_state_dict(state_dict_list[2])
            start_epoch = state_dict_list[3]
            gamma = state_dict_list[4]
            converged = state_dict_list[5]
        return start_epoch, gamma, converged

    def _save_checkpoint(self, filename, epoch, gamma, converged):
        checkpoint = {
            'epoch': epoch,
            'gamma': gamma,
            'state_dict': self.model_ddp.state_dict(),
            'optimizer': self.opt.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'converged': converged,
        }
        print('Checkpoint file:', filename)
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

        epoch_time = (time.time() - epoch_start) / 60  # in minutes
        return epoch_time

    def _compute_loss(self, output_dict, y_batch, epoch, gamma):
        x_output = output_dict['x_output']
        x_penultimate = output_dict['x_penultimate']
        x_backbone = output_dict['x_backbone']

        loss_classification = nn.CrossEntropyLoss()(x_output, y_batch).to(self.rank)
        loss = loss_classification

        if self.l2_loss:
            loss_compression = nn.functional.mse_loss(
                x_penultimate,
                torch.zeros_like(x_penultimate),
                reduction='mean'
            )
            loss = loss + loss_compression * gamma

        if self.scl:
            if self.architecture_type == 'lin_pen' or self.architecture_type == 'nonlin_pen':
                x = x_backbone
            elif self.architecture_type == 'no_pen':
                x = x_penultimate
            loss += supervised_contrastive_loss(x, y_batch, device=self.rank)

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
        res_epoch = {}
        eval_trainloader = DataLoader(
            self.trainset,
            batch_size=4 * self.training_hypers['batch_size'],
            sampler=self.train_sampler,
            num_workers=4,
            pin_memory=True,
        )
        eval_train = self.eval(eval_trainloader, self.rank)
        if self.distributed:
            gathered_eval_train = gather_dict_outputs_ddp(eval_train, self.rank, self.world_size)
        else:
            gathered_eval_train = eval_train

        if self.rank == 0:
            eval_testloader = DataLoader(
                self.testset,
                batch_size=4 * self.training_hypers['batch_size'],
                num_workers=4,
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

            return res_epoch
        else:
            return None

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

        res_dict_stack['entropy_train'] = res_list[-1].get('entropy_train', 0)
        res_dict_stack['entropy_test'] = res_list[-1].get('entropy_test', 0)

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
        x_input_list = []

        with torch.no_grad():
            for x_input_batch, y_label_batch in loader:
                x_input_batch = x_input_batch.to(device)
                y_label_batch = y_label_batch.to(device)

                output_dict_batch = self.network(x_input_batch)
                x_output_batch = output_dict_batch['x_output']
                x_penultimate_batch = output_dict_batch['x_penultimate']

                x_input_list.append(x_input_batch)
                x_output_list.append(x_output_batch)
                x_penultimate_list.append(x_penultimate_batch)
                y_label_list.append(y_label_batch)

            x_input = torch.cat(x_input_list, dim=0)
            x_output = torch.cat(x_output_list, dim=0)
            x_penultimate = torch.cat(x_penultimate_list, dim=0)
            y_label = torch.cat(y_label_list, dim=0)

            probs = torch.softmax(x_output, dim=-1)
            y_predicted = torch.argmax(probs, dim=1)
            top1_correct = (y_predicted == y_label).sum().item()
            top1_accuracy = top1_correct / y_label.size(0)
            top5_values, top5_indices = torch.topk(probs, k=5, dim=1)
            top5_correct = (top5_indices == y_label.unsqueeze(1)).any(dim=1).sum().item()
            top5_accuracy = top5_correct / y_label.size(0)

        evaluations.update({
            'x_output': x_output.cpu().numpy(),
            'x_penultimate': x_penultimate.cpu().numpy(),
            'y_predicted': y_predicted.cpu().numpy(),
            'y_label': y_label.cpu().numpy(),
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy,
        })

        return evaluations