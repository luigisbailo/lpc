import os
import importlib
import time
import math
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Subset

from lpc.utils import (
    load_checkpoint,
    save_checkpoint,
)
from lpc.losses import (
    arcface_loss,
    cosface_loss,
    supervised_contrastive_loss,
)
from lpc.metrics import (
    get_collapse_metrics,
    get_distance_margins,
    get_coeff_var,
    get_entropy,
    get_binarity_metrics,
    estimate_lipschitz_gradient_norm_stream,
)

from lpc.deepfool import deepfool


class Trainer:
    """Runs distributed training, evaluation, and metric collection for classifiers."""
    
    # =========================================================================
    # Configuration Constants
    # =========================================================================
    
    # Maximum samples for various computations (to avoid OOM)
    MAX_SAMPLES_ENTROPY = 10000
    MAX_SAMPLES_COEFF_VAR = 10000000
    MAX_SAMPLES_DISTANCE_MARGINS = 100000
    MAX_SAMPLES_BINARITY = 100000
    MAX_SAMPLES_ACCURACY_CHECK = 10000
    MAX_SAMPLES_COLLAPSE_TEST = 5000
    MAX_CLASSES_FOR_COLLAPSE = 100

    # Batch processing parameters
    COLLAPSE_METRICS_BATCH_SIZE = 100
    DISTANCE_MARGINS_BATCH_SIZE = 100
    CENTROIDS_CHUNK_SIZE = 50
    
    # Memory management
    HIGH_DIM_THRESHOLD = 1024
    LARGE_FEATURE_DIM_THRESHOLD = 100
    
    # =========================================================================
    # Initialization
    # =========================================================================
    
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
        """
        Initialize the Trainer.
        
        Args:
            network: The neural network model to train
            architecture: Architecture configuration dict
            trainset: Training dataset
            testset: Test dataset
            training_hypers: Training hyperparameters dict
            architecture_type: Type of architecture (e.g., 'lin_pen', 'nonlin_pen')
            l2_loss: Whether to use L2 loss on penultimate layer
            cosface_loss: Whether to use CosFace loss
            arcface_loss: Whether to use ArcFace loss
            scl: Whether to use supervised contrastive loss
            store_penultimate: Whether to store penultimate layer outputs
            verbose: Whether to print training progress
        """
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
        
        # Distributed training attributes (set during fit)
        self.rank = None
        self.world_size = None
        self.distributed = False
        
        # Training configuration
        
        # Convergence tracking
        self.converged = False
        self.convergence_epoch = None
        self.accuracy_test_converged = None
        self._gamma_second_regime_start_epoch = None

    # =========================================================================
    # Public API
    # =========================================================================
    
    def fit(self, checkpoint_file, rank, world_size, distributed):
        """Run the training loop and return aggregated metrics on rank 0.

        Args:
            checkpoint_file: Path to checkpoint file for saving/loading.
            rank: Process rank when using distributed training.
            world_size: Number of distributed processes.
            distributed: Whether training runs with distributed data parallelism.

        Returns:
            dict | None: Aggregated metrics on rank 0; None for non-zero ranks.
        """
        print("Weight decay: ", self.training_hypers['weight_decay'])
        # Initialize distributed training context
        self._initialize_distributed_context(rank, world_size, distributed)

        # Setup model and optimizer
        self._prepare_model()
        self._setup_optimizer()

        # Load checkpoint if exists
        start_epoch, gamma, converged = self._load_checkpoint(checkpoint_file)
        if start_epoch >= self.training_hypers["total_epochs"]:
            print("Last epoch already reached. Exiting")
            return None

        # Calculate training epochs
        last_train_epoch = start_epoch + self.training_hypers["train_epochs"]
        # Ensure we don't exceed total_epochs
        last_train_epoch = min(last_train_epoch, self.training_hypers["total_epochs"])

        # Print training configuration
        if self.rank == 0:
            self._print_training_configuration(start_epoch, last_train_epoch)

        # Create data loader
        trainloader = self._create_train_dataloader()

        # Initialize tracking
        res_list = []
        epoch_training_time_list = []
        start_training_time = time.time()

        # Main training loop
        for epoch in range(start_epoch + 1, last_train_epoch + 1):

            # Set epoch for data sampler (ensures proper data shuffling)
            self.train_sampler.set_epoch(epoch)

            # Update learning rate with warmup
            self._apply_learning_rate_schedule(epoch)

            # Update gamma
            gamma = self._update_gamma(epoch, gamma)

            # Train one epoch
            epoch_time = self._train_epoch(epoch, trainloader, gamma)
            epoch_training_time_list.append(epoch_time)

            # Update learning rate scheduler
            self._update_lr_scheduler(epoch)

            # Periodic accuracy check
            self._check_periodic_accuracy(trainloader, epoch, gamma)

            # Full evaluation at logging intervals or at the very last epoch of total training
            is_last_epoch = epoch == last_train_epoch
            should_log = epoch % self.training_hypers["logging"] == 0 

            if should_log:
                res_epoch = self._evaluate_and_log(epoch, gamma, is_last_epoch)
                if res_epoch is not None:
                    res_list.append(res_epoch)

        # Finalize training
        elapsed_training_time = time.time() - start_training_time
        if self.rank == 0:
            res_dict_stack = self._aggregate_results(
                res_list, elapsed_training_time, epoch_training_time_list, converged
            )
            if checkpoint_file:
                self._save_checkpoint(checkpoint_file, last_train_epoch, gamma, converged)

            # Create completion flag if we've reached total_epochs
            if last_train_epoch >= self.training_hypers["total_epochs"]:
                self._create_completion_flag(checkpoint_file)

            return res_dict_stack
        return None
    
    # =========================================================================
    # Setup and Initialization Methods
    # =========================================================================
    
    def _initialize_distributed_context(self, rank, world_size, distributed):
        """Initialize distributed training context."""
        self.rank = rank
        self.world_size = world_size
        self.distributed = distributed
        torch.cuda.set_device(rank)
    
    def _prepare_model(self):
        """Prepare model for training (move to GPU and wrap with DDP if needed)."""
        self.model = self.network.to(self.rank)
        
        if self.world_size > 1:
            self.model_ddp = DDP(
                self.model,
                device_ids=[self.rank],
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True
            )
        else:
            self.model_ddp = self.model
    
    def _setup_optimizer(self):
        """Create optimizer and learning rate scheduler."""
        # Separate output layer parameters from others
        if self.world_size > 1:
            excluded_params = set(self.model_ddp.module.output_layer.parameters())
        else:
            excluded_params = set(self.model_ddp.output_layer.parameters())
        
        other_params = [p for p in self.model_ddp.parameters() if p not in excluded_params]
        
        # Apply initial warmup factor if enabled
        initial_lr = self._get_initial_learning_rate()
        
        # Create parameter groups
        params_to_update = [
            {"params": other_params, "lr": initial_lr},
            {"params": list(excluded_params), "lr": initial_lr},
        ]
        
        # Create optimizer
        torch_optim_module = importlib.import_module("torch.optim")
        self.opt = getattr(torch_optim_module, self.training_hypers["optimizer"])(
            params_to_update, weight_decay=self.training_hypers["weight_decay"]
        )
        
        # Create learning rate scheduler
        self._create_lr_scheduler()
    
    def _get_initial_learning_rate(self):
        """Calculate initial learning rate considering warmup."""
        lr = self.training_hypers["lr"]
        if "warmup_factor" in self.training_hypers and "warmup_end" in self.training_hypers:
            lr = self.training_hypers["lr"] * self.training_hypers["warmup_factor"]
        return lr
    
    def _create_lr_scheduler(self):
        """Create cosine annealing learning rate scheduler."""
        start_schedule_epoch = self.training_hypers.get("lr_scheduler_start", 0)
        total_epochs = self.training_hypers["total_epochs"]
        T_max_epochs = max(1, total_epochs - start_schedule_epoch)
        
        self.scheduler = CosineAnnealingLR(
            self.opt,
            T_max=T_max_epochs,
            eta_min=self.training_hypers.get("lr_min", 0),
        )
    
    def _create_train_dataloader(self):
        """Create distributed data loader for training."""
        num_workers = self.world_size * 4
        self.train_sampler = DistributedSampler(
            self.trainset, num_replicas=self.world_size, rank=self.rank
        )

        return DataLoader(
            self.trainset,
            batch_size=self.training_hypers["batch_size"],
            sampler=self.train_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    def _create_completion_flag(self, checkpoint_file):
        """Write a completion flag file when training reaches total_epochs.

        Args:
            checkpoint_file: Path to the checkpoint whose name is used for the flag.
        """
        checkpoint_dir = os.path.dirname(checkpoint_file)
        checkpoint_basename = os.path.basename(checkpoint_file)

        if checkpoint_basename.startswith('checkpoint_'):
            base_name = checkpoint_basename[11:]
        else:
            base_name = checkpoint_basename

        if base_name.endswith('.pth.tar'):
            base_name = base_name[:-8]

        flag_filename = f"training_completed_{base_name}.flag"
        flag_file_path = os.path.join(checkpoint_dir, flag_filename)

        with open(flag_file_path, 'w') as f:
            f.write(f"Training completed at epoch {self.training_hypers['total_epochs']}\n")
            f.write(f"Completion time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"Training completed! Created flag file: {flag_file_path}")
    
    # =========================================================================
    # Training Configuration Display
    # =========================================================================
    
    def _print_training_configuration(self, start_epoch, last_train_epoch):
        """Print training configuration and expected learning rate schedule."""
        
        if "warmup_factor" in self.training_hypers and "warmup_end" in self.training_hypers:
            self._print_expected_lr_schedule(start_epoch, last_train_epoch)
        self._print_optimization_settings()
        
        # Print gamma scheduler information
        dual_regime = ('gamma_scheduler_factor_1' in self.training_hypers and 
                       'gamma_scheduler_factor_2' in self.training_hypers)

        if dual_regime:
            print("\nGamma Scheduler (Dual Regime):")
            print("------------------------------")
            print(f"Initial gamma: {self.training_hypers['gamma']}")
            
            # Calculate gamma_max values with coefficient support
            gamma_max_coeff_1 = self.training_hypers.get('gamma_max_coeff_1', 1.0)
            gamma_max_coeff_2 = self.training_hypers.get('gamma_max_coeff_2', 1.0)
            gamma_max_1 = gamma_max_coeff_1 * (10 ** self.training_hypers['gamma_max_exp_1'])
            gamma_max_2 = gamma_max_coeff_2 * (10 ** self.training_hypers['gamma_max_exp_2'])
            
            print(f"First regime: factor={self.training_hypers['gamma_scheduler_factor_1']}, "
                  f"step={self.training_hypers['gamma_scheduler_step_1']}, "
                  f"max={gamma_max_1:.2e} (coeff={gamma_max_coeff_1}, exp={self.training_hypers['gamma_max_exp_1']})")
            print(f"Second regime: factor={self.training_hypers['gamma_scheduler_factor_2']}, "
                  f"step={self.training_hypers['gamma_scheduler_step_2']}, "
                  f"max={gamma_max_2:.2e} (coeff={gamma_max_coeff_2}, exp={self.training_hypers['gamma_max_exp_2']})")
            print(f"Start epoch: {self.training_hypers['gamma_scheduler_init']}")
            print("------------------------------\n")
        else:
            # Single regime
            gamma_max_coeff = self.training_hypers.get('gamma_max_coeff', 1.0)
            gamma_max = gamma_max_coeff * (10 ** self.training_hypers['gamma_max_exp'])
            
            print(f"\nGamma Scheduler (Single Regime): {self.training_hypers.get('gamma_scheduler_type', 'exponential')}")
            print(f"Gamma max: {gamma_max:.2e} (coeff={gamma_max_coeff}, exp={self.training_hypers['gamma_max_exp']})")

    
    def _print_expected_lr_schedule(self, start_epoch, last_train_epoch):
        """Print expected learning rates at each logging epoch."""
        lr = self.training_hypers["lr"]
        warmup_begin = self.training_hypers.get("warmup_begin", 0)
        warmup_factor = self.training_hypers["warmup_factor"]
        warmup_end = self.training_hypers["warmup_end"]
        logging_interval = self.training_hypers["logging"]
        lr_scheduler_start = self.training_hypers["lr_scheduler_start"]
        
        print(f"\nExpected learning rate schedule:")
        print("--------------------------------")
        print("Epoch | Backbone LR | Linear Layer LR")
        print("--------------------------------")
        
        for epoch in range(start_epoch + 1, last_train_epoch + 1, logging_interval):
            backbone_lr = lr
            linear_lr = lr

            if epoch <= warmup_begin:
                warmup_lr = lr * warmup_factor
                backbone_lr = warmup_lr
                linear_lr = warmup_lr
            elif epoch <= warmup_end:
                alpha = (epoch - warmup_begin) / (warmup_end - warmup_begin)
                warmup_lr = lr * (warmup_factor + (1 - warmup_factor) * alpha)
                backbone_lr = warmup_lr
                linear_lr = warmup_lr
            
            if epoch >= lr_scheduler_start:
                t = epoch - lr_scheduler_start
                T_max = max(1, self.training_hypers["total_epochs"] - lr_scheduler_start)
                eta_min = self.training_hypers.get("lr_min", 0)
                backbone_lr = eta_min + 0.5 * (lr - eta_min) * (1 + math.cos(math.pi * t / T_max))
            
            print(f"{epoch:5d} | {backbone_lr:.6f} | {linear_lr:.6f}")
        
        print("--------------------------------\n")
    
    def _print_optimization_settings(self):
        """Print optimization settings."""
        print("\nOptimization Settings:")
        print("----------------------")
        print(f"Dataset: {len(self.trainset)} training samples, {len(self.testset)} test samples")
        n_classes = len(self.trainset.classes) if hasattr(self.trainset, 'classes') else 'unknown'
        print(f"Number of classes: {n_classes}")
        print(f"Batch size: {self.training_hypers['batch_size']}")
        print(f"Number of GPUs: {self.world_size}")
        
        epochs_accuracy = self.training_hypers.get("epochs_accuracy", None)
        if epochs_accuracy:
            print(f"Periodic accuracy checking: every {epochs_accuracy} epochs")
        print("----------------------\n")
    
    # =========================================================================
    # Checkpointing
    # =========================================================================
    
    def _load_checkpoint(self, checkpoint_file):
        """
        Load checkpoint if exists.
        
        Returns:
            tuple: (start_epoch, gamma, converged)
        """
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
        
        # Broadcast checkpoint data to all ranks
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
        """Save training checkpoint."""
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
    
    # =========================================================================
    # Training Loop Methods
    # =========================================================================
    
    def _apply_learning_rate_schedule(self, epoch):
        """Apply warmup learning rate schedule if enabled."""
        if "warmup_factor" not in self.training_hypers or "warmup_end" not in self.training_hypers:
            return

        warmup_factor = self.training_hypers["warmup_factor"]
        warmup_end = self.training_hypers["warmup_end"]
        warmup_begin = self.training_hypers.get("warmup_begin", 0)  

        if epoch <= warmup_begin:
            # Keep learning rate constant at warmup_factor * lr
            current_lr = self.training_hypers["lr"] * warmup_factor
        elif epoch <= warmup_end:
            # Linear interpolation from warmup_factor*lr to lr
            alpha = (epoch - warmup_begin) / (warmup_end - warmup_begin)
            current_lr = self.training_hypers["lr"] * (warmup_factor + (1 - warmup_factor) * alpha)
        else:
            return  # No warmup adjustment needed

        # Update learning rate for both parameter groups
        for param_group in self.opt.param_groups:
            param_group['lr'] = current_lr
    
    def _update_gamma(self, epoch, current_gamma):
        """Update gamma according to the configured single or dual scheduler."""
        # Check if we're using dual regime or single regime
        dual_regime = ('gamma_scheduler_factor_1' in self.training_hypers and 
                       'gamma_scheduler_factor_2' in self.training_hypers)

        if dual_regime:
            # Validate dual regime parameters
            required_params = [
                'gamma_scheduler_factor_1', 'gamma_scheduler_step_1', 'gamma_max_exp_1',
                'gamma_scheduler_factor_2', 'gamma_scheduler_step_2', 'gamma_max_exp_2',
                'gamma_scheduler_init', 'gamma'
            ]
            missing_params = [p for p in required_params if p not in self.training_hypers]
            if missing_params:
                raise ValueError(f"Missing required parameters for dual gamma regime: {missing_params}")

            # Dual regime is always exponential
            gamma_min = self.training_hypers["gamma"]
            
            # Get coefficients (default to 1.0 for backward compatibility)
            gamma_max_coeff_1 = self.training_hypers.get("gamma_max_coeff_1", 1.0)
            gamma_max_coeff_2 = self.training_hypers.get("gamma_max_coeff_2", 1.0)
            
            # Calculate gamma_max values with coefficient
            gamma_max_1 = gamma_max_coeff_1 * (10 ** self.training_hypers["gamma_max_exp_1"])
            gamma_max_2 = gamma_max_coeff_2 * (10 ** self.training_hypers["gamma_max_exp_2"])

            if gamma_max_1 >= gamma_max_2:
                raise ValueError(f"gamma_max_1 ({gamma_max_1:.2e}) must be less than "
                               f"gamma_max_2 ({gamma_max_2:.2e})")

            init_epoch = self.training_hypers["gamma_scheduler_init"]

            if epoch < init_epoch:
                gamma = gamma_min
            elif current_gamma < gamma_max_1:
                # First regime
                epochs_since_init = epoch - init_epoch
                if epochs_since_init % self.training_hypers["gamma_scheduler_step_1"] == 0:
                    gamma = min(current_gamma * self.training_hypers["gamma_scheduler_factor_1"], gamma_max_1)
                else:
                    gamma = current_gamma
            elif current_gamma < gamma_max_2:
                # Second regime
                # Store the epoch when we transitioned to second regime
                if self._gamma_second_regime_start_epoch is None:
                    self._gamma_second_regime_start_epoch = epoch

                epochs_in_second_regime = epoch - self._gamma_second_regime_start_epoch
                if epochs_in_second_regime % self.training_hypers["gamma_scheduler_step_2"] == 0:
                    gamma = min(current_gamma * self.training_hypers["gamma_scheduler_factor_2"], gamma_max_2)
                else:
                    gamma = current_gamma
            else:
                gamma = current_gamma

        else:
            # Single regime (original behavior)
            sched_type = self.training_hypers.get("gamma_scheduler_type", "exponential")
            gamma_min = self.training_hypers["gamma"]
            
            # Get coefficient (default to 1.0 for backward compatibility)
            gamma_max_coeff = self.training_hypers.get("gamma_max_coeff", 1.0)
            
            # Calculate gamma_max with coefficient
            gamma_max = gamma_max_coeff * (10 ** self.training_hypers["gamma_max_exp"])
            
            T = self.training_hypers.get("gamma_scheduler_T", 1)
            t0 = self.training_hypers.get("gamma_scheduler_init", 0)

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
            else:  # exponential
                if (
                    epoch % self.training_hypers["gamma_scheduler_step"] == 0
                    and current_gamma < gamma_max
                    and epoch >= self.training_hypers["gamma_scheduler_init"]
                ):
                    gamma = current_gamma * self.training_hypers["gamma_scheduler_factor"]
                else:
                    gamma = current_gamma

        return gamma
    
    def _train_epoch(self, epoch, trainloader, gamma):
        """Run one training epoch and return its duration in minutes."""
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
        """Compute training loss based on configured loss functions."""
        x_output = output_dict["x_output"]
        x_penultimate = output_dict["x_penultimate"]
        x_backbone = output_dict["x_backbone"]
        
        # Base cross-entropy loss
        loss = nn.CrossEntropyLoss()(x_output, y_batch).to(self.rank)
        
        # Add L2 loss on penultimate layer if enabled
        if self.l2_loss:
            loss += F.mse_loss(
                x_penultimate, torch.zeros_like(x_penultimate), reduction="mean"
            ) * gamma
        
        # Add supervised contrastive loss if enabled
        if self.scl:
            x_for_scl = x_backbone if self.architecture_type in {"lin_pen", "nonlin_pen"} else x_penultimate
            loss += supervised_contrastive_loss(x_for_scl, y_batch, device=self.rank)
        
        # Use ArcFace loss if enabled 
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
        
        # Use CosFace loss if enabled 
        if self.cosface:
            loss = cosface_loss(
                features=x_penultimate,
                labels=y_batch,
                weight=self.model_ddp.output_layer.weight,
                device=self.rank,
            )
        
        return loss
    
    def _update_lr_scheduler(self, epoch):
        """Update learning rate scheduler."""
        if epoch >= self.training_hypers["lr_scheduler_start"]:
            self.scheduler.step()
            # Keep linear layer LR constant
            self.opt.param_groups[1]["lr"] = self.training_hypers["lr"]
    
    def _check_periodic_accuracy(self, trainloader, epoch, gamma):
        """Check accuracy periodically without full metrics computation."""
        epochs_accuracy = self.training_hypers.get("epochs_accuracy", None)
        if not epochs_accuracy:
            return
        
        # Only check if not already doing full evaluation
        if epoch % epochs_accuracy == 0 and epoch % self.training_hypers["logging"] != 0:
            self._evaluate_accuracy_only(trainloader, epoch, gamma)
            
    # =========================================================================
    # Evaluation Methods
    # =========================================================================
    
    def eval(self, loader, device):
        """Evaluate the model on loader and return predictions and features."""
        evaluations = {}
        self.model_ddp.eval()
        
        x_output_list = []
        x_penultimate_list = []
        y_label_list = []
        
        with torch.no_grad():
            for x_input_batch, y_label_batch in loader:
                x_input_batch = x_input_batch.to(device)
                y_label_batch = y_label_batch.to(device)
                
                output_dict_batch = self.model_ddp(x_input_batch)
                x_output_batch = output_dict_batch['x_output']
                x_penultimate_batch = output_dict_batch['x_penultimate']
                
                # Keep on GPU to avoid repeated transfers
                x_output_list.append(x_output_batch)
                x_penultimate_list.append(x_penultimate_batch)
                y_label_list.append(y_label_batch)
            
            # Concatenate on GPU first, then transfer once
            x_output_gpu = torch.cat(x_output_list, dim=0)
            x_penultimate_gpu = torch.cat(x_penultimate_list, dim=0)
            y_label_gpu = torch.cat(y_label_list, dim=0)
            
            # Compute predictions and accuracies on GPU
            y_predicted_gpu = torch.argmax(x_output_gpu, dim=1)
            
            # Top-1 accuracy
            top1_correct = (y_predicted_gpu == y_label_gpu).sum().item()
            top1_accuracy = top1_correct / y_label_gpu.shape[0]
            
            # Top-5 accuracy
            _, top5_indices = torch.topk(x_output_gpu, k=5, dim=1)
            y_label_expanded = y_label_gpu.view(-1, 1)
            top5_correct = (top5_indices == y_label_expanded).sum().item()
            top5_accuracy = top5_correct / y_label_gpu.shape[0]
            
            # Transfer to CPU only once
            x_output = x_output_gpu.cpu().numpy()
            x_penultimate = x_penultimate_gpu.cpu().numpy()
            y_predicted = y_predicted_gpu.cpu().numpy()
            y_label = y_label_gpu.cpu().numpy()
            
            # Clear GPU memory
            del x_output_gpu, x_penultimate_gpu, y_label_gpu, y_predicted_gpu
            torch.cuda.empty_cache()
        
        evaluations.update({
            'x_output': x_output,
            'x_penultimate': x_penultimate,
            'y_predicted': y_predicted,
            'y_label': y_label,
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy,
        })
        
        return evaluations
    
    def _evaluate_accuracy_only(self, trainloader, epoch, gamma):
        """
        Quick accuracy evaluation for periodic checks.
        
        Evaluates accuracy on a subsample of training and test sets without
        computing expensive metrics.
        """
        if self.distributed:
            dist.barrier()
        
        self.model_ddp.eval()
        
        # Evaluate training accuracy on subsample
        train_correct = 0
        train_total = 0
        
        with torch.no_grad():
            for batch_idx, (x_input_batch, y_batch) in enumerate(trainloader):
                if train_total >= self.MAX_SAMPLES_ACCURACY_CHECK:
                    break
                
                x_input_batch = x_input_batch.to(self.rank)
                y_batch = y_batch.to(self.rank)
                
                output_dict = self.model_ddp(x_input_batch)
                x_output = output_dict['x_output']
                
                predictions = torch.argmax(x_output, dim=1)
                train_correct += (predictions == y_batch).sum().item()
                train_total += y_batch.size(0)
        
        # Aggregate across GPUs
        if self.distributed:
            stats = torch.tensor([train_correct, train_total], dtype=torch.float32).cuda(self.rank)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            train_correct = int(stats[0].item())
            train_total = int(stats[1].item())
        
        train_accuracy = train_correct / train_total if train_total > 0 else 0.0
        
        # Evaluate test accuracy on rank 0
        if self.rank == 0:
            test_correct = 0
            test_total = 0
            
            # Create a simple test loader
            test_loader = DataLoader(
                self.testset,
                batch_size=self.training_hypers['batch_size'],
                shuffle=True,
                num_workers=4
            )
            
            with torch.no_grad():
                for batch_idx, (x_input_batch, y_batch) in enumerate(test_loader):
                    if test_total >= self.MAX_SAMPLES_ACCURACY_CHECK:
                        break
                    
                    x_input_batch = x_input_batch.to(self.rank)
                    y_batch = y_batch.to(self.rank)
                    
                    output_dict = self.model_ddp(x_input_batch)
                    x_output = output_dict['x_output']
                    
                    predictions = torch.argmax(x_output, dim=1)
                    test_correct += (predictions == y_batch).sum().item()
                    test_total += y_batch.size(0)
            
            test_accuracy = test_correct / test_total if test_total > 0 else 0.0
            
            # Print results
            print(f"\nEpoch {epoch} (periodic accuracy check):")
            print(f"  Train accuracy (on {train_total} samples): {train_accuracy:.4f}")
            print(f"  Test accuracy (on {test_total} samples): {test_accuracy:.4f}")
            print(f"  Current gamma: {gamma}")
            print(f"  Current learning rates - Backbone: {self.opt.param_groups[0]['lr']:.6f}, Linear: {self.opt.param_groups[1]['lr']:.6f}")
            print()
        
        # Set model back to train mode
        self.model_ddp.train()
        
        if self.distributed:
            dist.barrier()
    
    def _evaluate_and_log(self, epoch, gamma, is_last_epoch):
        """Run full evaluation and metric logging, returning metrics on rank 0."""
        if self.distributed:
            dist.barrier()
        
        # Create evaluation data loader
        num_workers = self.world_size * 4
        eval_trainloader = DataLoader(
            self.trainset,
            batch_size=self.training_hypers['batch_size'],
            sampler=self.train_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        # Evaluate locally on each GPU
        local_eval_train = self.eval(eval_trainloader, self.rank)
        
        # Clear GPU cache
        if self.distributed:
            torch.cuda.empty_cache()
        
        # Only rank 0 evaluates test set
        local_eval_test = None
        if self.rank == 0:
            # Print current learning rates
            backbone_lr = self.opt.param_groups[0]['lr']
            linear_lr = self.opt.param_groups[1]['lr']
            print(f"Current learning rates - Backbone: {backbone_lr:.6f}, Linear Layer: {linear_lr:.6f}")
            
            eval_testloader = DataLoader(
                self.testset,
                batch_size=self.training_hypers['batch_size'],
                num_workers=num_workers,
                pin_memory=True,
            )
            local_eval_test = self.eval(eval_testloader, self.rank)
            
            # Clear cache after test evaluation
            torch.cuda.empty_cache()
        
        # Synchronize before metrics computation
        if self.distributed:
            dist.barrier()
        
        # Compute metrics in distributed manner
        metrics = self._compute_distributed_metrics(local_eval_train, local_eval_test, is_last_epoch)
        
        if self.rank == 0:
            res_epoch = {'epochs': epoch}
            res_epoch.update(metrics)
            
            if self.verbose:
                print(f"Epoch {epoch}")
                print(f"gamma: {gamma}")
                print(f"Accuracy train: {np.around(res_epoch['accuracy_train'], 4)}\tAccuracy test: {np.around(res_epoch['accuracy_test'], 4)}")
            
            # Check convergence
            if res_epoch['accuracy_train'] > self.training_hypers['convergence_thres'] and not self.converged:
                self.converged = True
                self.convergence_epoch = epoch
                self.accuracy_test_converged = res_epoch['accuracy_test']
                print('converged!', epoch)
            
            # Store penultimate if needed (only for final epoch)
            if self.store_penultimate and epoch == self.training_hypers['total_epochs']:
                self.penultimate_train = local_eval_train['x_penultimate']
                self.penultimate_test = local_eval_test['x_penultimate'] if local_eval_test else None
        
        # Skip barrier on last epoch to avoid NCCL timeout during expensive rank-0-only computations
        if self.distributed and not is_last_epoch:
            dist.barrier()

        return res_epoch if self.rank == 0 else None
    
    # =========================================================================
    # Distributed Metrics Computation
    # =========================================================================
    
    def _compute_distributed_metrics(self, local_eval_train, local_eval_test, is_last_epoch):
        """
        Compute all metrics in a distributed manner.
        
        This method orchestrates the computation of various metrics across
        distributed GPUs, aggregating results efficiently.
        """
        metrics = {}
        
        # 1. Accuracies
        self._add_accuracy_metrics(metrics, local_eval_train, local_eval_test)
        
        # 2. Collapse metrics
        self._add_collapse_metrics(metrics, local_eval_train, local_eval_test)
        
        # 3. Coefficient of variation
        self._add_coeff_var_metrics(metrics, local_eval_train, local_eval_test)
        
        # Clear memory before expensive computations
        if self.distributed:
            torch.cuda.empty_cache()
        
        # 4. Distance margins
        self._add_distance_margin_metrics(metrics, local_eval_train, local_eval_test)
        
        # 5. Last epoch metrics
        if is_last_epoch and self.rank == 0:
            self._add_last_epoch_metrics(metrics, local_eval_test)
        else:
            self._add_empty_last_epoch_metrics(metrics)

        # Synchronize after expensive rank-0-only computations on last epoch
        if self.distributed and is_last_epoch:
            dist.barrier()

        # 6. Architecture-specific metrics
        if self.architecture_type == 'lin_pen' and self.rank == 0:
            self._add_binarity_metrics(metrics, local_eval_train, local_eval_test)

        # Final synchronization after all rank-0-only computations
        if self.distributed and is_last_epoch:
            dist.barrier()

        if self.rank == 0:
            print("=== Metrics computation complete ===\n")
        
        return metrics
    
    def _add_accuracy_metrics(self, metrics, local_eval_train, local_eval_test):
        """Add accuracy metrics to results."""
        if self.rank == 0:
            print("Computing distributed accuracies...")

        # Compute local statistics
        local_correct_train = (local_eval_train['y_predicted'] == local_eval_train['y_label']).sum()
        local_total_train = len(local_eval_train['y_label'])

        _, top5_indices = torch.topk(torch.from_numpy(local_eval_train['x_output']), k=5, dim=1)
        y_label_expanded = torch.from_numpy(local_eval_train['y_label']).view(-1, 1)
        local_top5_correct_train = (top5_indices == y_label_expanded).sum().item()

        if self.distributed:
            # Aggregate counts across all GPUs
            counts_train = torch.tensor([local_correct_train, local_total_train, 
                                        local_correct_train,
                                        local_top5_correct_train], 
                                       dtype=torch.float32).cuda(self.rank)
            dist.all_reduce(counts_train, op=dist.ReduceOp.SUM)

            if self.rank == 0:
                total_correct_train = int(counts_train[0].item())
                total_samples_train = int(counts_train[1].item())
                total_top5_correct_train = int(counts_train[3].item()) # Get the aggregated top-5 count

                metrics['accuracy_train'] = total_correct_train / total_samples_train
                metrics['top1_accuracy_train'] = total_correct_train / total_samples_train # Top-1 is same as accuracy
                metrics['top5_accuracy_train'] = total_top5_correct_train / total_samples_train
        else:
            metrics['accuracy_train'] = (local_eval_train['y_predicted'] == local_eval_train['y_label']).mean()
            metrics['top1_accuracy_train'] = local_eval_train['top1_accuracy']
            metrics['top5_accuracy_train'] = local_eval_train['top5_accuracy']

        # Test metrics only on rank 0
        if self.rank == 0 and local_eval_test is not None:
            metrics['accuracy_test'] = (local_eval_test['y_predicted'] == local_eval_test['y_label']).mean()
            metrics['top1_accuracy_test'] = local_eval_test['top1_accuracy']
            metrics['top5_accuracy_test'] = local_eval_test['top5_accuracy']
    
    def _add_collapse_metrics(self, metrics, local_eval_train, local_eval_test):
        """Add collapse metrics to results."""
        if self.rank == 0:
            print("Computing distributed collapse metrics...")
            sys.stdout.flush()

        start_time = time.time()

        # Compute collapse metrics for training set (distributed)
        if self.distributed:
            collapse_train = self._compute_collapse_metrics_distributed(local_eval_train)
        else:
            collapse_train = get_collapse_metrics(local_eval_train)

        if self.rank == 0:
            metrics['collapse_train'] = collapse_train
            elapsed = (time.time()-start_time)/60
            print(f"Elapsed time to compute collapse metrics (train): {elapsed:.2f} minutes")
            sys.stdout.flush()

            if local_eval_test is not None:
                print("Computing collapse metrics for test set...")
                sys.stdout.flush()
                start_time = time.time()

                # Check number of unique classes in test set
                unique_test_classes = np.unique(local_eval_test['y_label'])
                n_test_classes = len(unique_test_classes)

                # Subsample classes if more than 100
                if n_test_classes > self.MAX_CLASSES_FOR_COLLAPSE:
                    print(f"Test set has {n_test_classes} classes. Subsampling to {self.MAX_CLASSES_FOR_COLLAPSE} classes for collapse metrics...")

                    # Randomly select 100 classes
                    np.random.seed(42)  # For reproducibility
                    selected_classes = np.random.choice(unique_test_classes, self.MAX_CLASSES_FOR_COLLAPSE, replace=False)

                    # Filter the test data to only include samples from selected classes
                    class_mask = np.isin(local_eval_test['y_label'], selected_classes)
                    subsampled_eval_test = {
                        'x_penultimate': local_eval_test['x_penultimate'][class_mask],
                        'y_predicted': local_eval_test['y_predicted'][class_mask],
                        'y_label': local_eval_test['y_label'][class_mask]
                    }

                    # Filter out samples where predictions are outside selected classes
                    pred_class_mask = np.isin(subsampled_eval_test['y_predicted'], selected_classes)
                    if not np.all(pred_class_mask):
                        print(f"Filtering out {np.sum(~pred_class_mask)} samples with predictions outside selected classes")
                        subsampled_eval_test = {
                            'x_penultimate': subsampled_eval_test['x_penultimate'][pred_class_mask],
                            'y_predicted': subsampled_eval_test['y_predicted'][pred_class_mask],
                            'y_label': subsampled_eval_test['y_label'][pred_class_mask]
                        }

                    print(f"Subsampled to {len(subsampled_eval_test['y_label'])} test samples from {self.MAX_CLASSES_FOR_COLLAPSE} classes")

                    # Add sample size limit as before
                    test_samples = len(subsampled_eval_test['y_label'])
                    if test_samples > self.MAX_SAMPLES_COLLAPSE_TEST:
                        indices = np.random.choice(test_samples, self.MAX_SAMPLES_COLLAPSE_TEST, replace=False)
                        final_eval_test = {
                            'x_penultimate': subsampled_eval_test['x_penultimate'][indices],
                            'y_predicted': subsampled_eval_test['y_predicted'][indices],
                            'y_label': subsampled_eval_test['y_label'][indices]
                        }
                        print(f"Further subsampled to {self.MAX_SAMPLES_COLLAPSE_TEST} samples")
                    else:
                        final_eval_test = subsampled_eval_test

                    collapse_test = get_collapse_metrics(final_eval_test)

                else:
                    test_samples = len(local_eval_test['y_label'])
                    if test_samples > self.MAX_SAMPLES_COLLAPSE_TEST:
                        indices = np.random.choice(test_samples, self.MAX_SAMPLES_COLLAPSE_TEST, replace=False)
                        subsampled_eval_test = {
                            'x_penultimate': local_eval_test['x_penultimate'][indices],
                            'y_predicted': local_eval_test['y_predicted'][indices],
                            'y_label': local_eval_test['y_label'][indices]
                        }
                        collapse_test = get_collapse_metrics(subsampled_eval_test)
                    else:
                        collapse_test = get_collapse_metrics(local_eval_test)

                metrics['collapse_test'] = collapse_test

                elapsed = (time.time()-start_time)/60
                print(f"Elapsed time to compute collapse metrics (test): {elapsed:.2f} minutes")
                sys.stdout.flush()
    
    def _add_coeff_var_metrics(self, metrics, local_eval_train, local_eval_test):
        """Add coefficient of variation metrics to results."""
        if self.rank == 0:
            print("Computing distributed coefficient of variation...")
            sys.stdout.flush()
        
        start_time = time.time()
        coeff_var_train = self._compute_distributed_coeff_var(local_eval_train)
        
        if self.rank == 0:
            metrics['coeff_var_abs_train'] = coeff_var_train[0]
            metrics['coeff_var_norm_train'] = coeff_var_train[1]
            metrics['norm_mean_train'] = coeff_var_train[2]
            
            if local_eval_test is not None:
                coeff_var_norm_test, coeff_var_abs_test, norm_mean_test = get_coeff_var(local_eval_test)
                metrics['coeff_var_abs_test'] = coeff_var_abs_test
                metrics['coeff_var_norm_test'] = coeff_var_norm_test
                metrics['norm_mean_test'] = norm_mean_test
            
            elapsed = (time.time()-start_time)/60
            print(f"Elapsed time to compute coefficient of variation: {elapsed:.2f} minutes")
            sys.stdout.flush()
    
    def _add_distance_margin_metrics(self, metrics, local_eval_train, local_eval_test):
        """Add distance margin metrics to results."""
        if self.rank == 0:
            print("Computing distributed distance margins...")
            sys.stdout.flush()
        
        start_time = time.time()
        distance_margins_train = self._compute_distributed_distance_margins(local_eval_train)
        
        if self.rank == 0:
            metrics['distance_margins_train'] = distance_margins_train
            if local_eval_test is not None:
                metrics['distance_margins_test'] = get_distance_margins(local_eval_test)
            elapsed = (time.time()-start_time)/60
            print(f"Elapsed time to compute distance margins: {elapsed:.2f} minutes")
            sys.stdout.flush()
    
    def _add_last_epoch_metrics(self, metrics, local_eval_test):
        """Add expensive metrics computed only on last epoch."""
        print("5. Computing last epoch metrics...")
        
        # DeepFool
        self._add_deepfool_metric(metrics)
        
        # Entropy
        self._add_entropy_metrics(metrics, local_eval_test)
        
        # Lipschitz estimation
        self._add_lipschitz_metrics(metrics)
    
    def _add_empty_last_epoch_metrics(self, metrics):
            """Add empty placeholders for last epoch metrics."""
            if self.rank == 0:
                # Original metrics
                metrics['deepfool_score'] = None
                metrics['entropy_train'] = None
                metrics['entropy_test'] = None
                metrics['lipschitz_estimation_train'] = None
                metrics['lipschitz_estimation_test'] = None

                # New DeepFool metrics
                metrics['deepfool_score_all'] = None
                metrics['deepfool_score_correct'] = None
                metrics['deepfool_n_samples'] = None
                metrics['deepfool_n_correct'] = None

                # New Lipschitz metrics
                metrics['lipschitz_estimation_train_correct'] = None
                metrics['lipschitz_estimation_test_correct'] = None
    
    def _add_deepfool_metric(self, metrics):
        """Compute and add DeepFool metric for both all samples and correctly classified only."""
        if self.rank != 0:
            return

        print("   Computing DeepFool score...")
        start_time = time.time()

        batch_size_deepfool = 1000
        loader = DataLoader(self.testset, batch_size=batch_size_deepfool, shuffle=True)
        images, labels = next(iter(loader))
        images = images.to(self.rank)
        labels = labels.to(self.rank)

        # Use the underlying module to avoid DDP gradient synchronization
        model_for_deepfool = self.model_ddp.module if hasattr(self.model_ddp, 'module') else self.model_ddp

        # Temporarily disable parameter gradients so DDP does not trigger all-reduce hooks
        original_training_state = model_for_deepfool.training
        original_requires_grad = []

        try:
            model_for_deepfool.eval()
            for param in model_for_deepfool.parameters():
                original_requires_grad.append(param.requires_grad)
                param.requires_grad_(False)

            # Get predictions to identify correctly classified samples
            with torch.no_grad():
                outputs = model_for_deepfool(images)
                predictions = torch.argmax(outputs['x_output'], dim=1)

            # Identify correctly classified samples
            correct_mask = (predictions == labels).cpu().numpy()
            n_correct = np.sum(correct_mask)

            print(f"   Found {n_correct}/{len(images)} correctly classified samples")

            # Compute DeepFool for ALL samples
            perturbation_list_all = []
            perturbation_list_correct = []

            for image_idx in range(images.size(0)):
                img_single = images[image_idx]
                r_tot, loop_i, label, k_i, pert_image = deepfool(img_single, model_for_deepfool)
                perturbation_norm = np.linalg.norm(r_tot) / np.linalg.norm(img_single.cpu().numpy())

                # Add to all samples list
                perturbation_list_all.append(perturbation_norm)

                # Add to correct samples list if applicable
                if correct_mask[image_idx]:
                    perturbation_list_correct.append(perturbation_norm)

            # Compute metrics
            metrics['deepfool_score'] = np.mean(perturbation_list_all) if perturbation_list_all else 0.0
            metrics['deepfool_score_all'] = metrics['deepfool_score']  # Keep backward compatibility
            metrics['deepfool_score_correct'] = np.mean(perturbation_list_correct) if perturbation_list_correct else None
            metrics['deepfool_n_samples'] = len(perturbation_list_all)
            metrics['deepfool_n_correct'] = len(perturbation_list_correct)

            print(f"   DeepFool (all samples): {metrics['deepfool_score']:.4f}")
            if metrics['deepfool_score_correct'] is not None:
                print(f"   DeepFool (correct only): {metrics['deepfool_score_correct']:.4f}")
            print(f"   DeepFool time: {(time.time()-start_time):.2f} seconds")
        finally:
            # Restore original requires_grad flags and training state
            for param, requires_grad in zip(model_for_deepfool.parameters(), original_requires_grad):
                param.requires_grad_(requires_grad)
            if original_training_state:
                model_for_deepfool.train()

    
    def _add_entropy_metrics(self, metrics, local_eval_test):
        """Compute and add entropy metrics."""
        print("   Computing entropy on sampled data...")
        start_time = time.time()
        
        # Sample subset of test data for entropy computation
        sample_size = min(self.MAX_SAMPLES_ENTROPY, len(local_eval_test['y_label']))
        sample_indices = np.random.choice(len(local_eval_test['y_label']), sample_size, replace=False)
        sampled_eval_test = {
            'x_penultimate': local_eval_test['x_penultimate'][sample_indices],
            'y_label': local_eval_test['y_label'][sample_indices]
        }
        
        metrics['entropy_train'] = None  # Skip train entropy due to distributed nature
        metrics['entropy_test'] = get_entropy(sampled_eval_test, k=20, normalize=True)
        print(f"   Entropy time: {(time.time()-start_time):.2f} seconds")
    
    def _add_lipschitz_metrics(self, metrics):
        """Compute and add Lipschitz estimation metrics for both all and correct samples."""
        print("   Computing Lipschitz estimation...")
        start_time = time.time()

        subsample_size = 100000

        # Test set - both all and correct
        test_indices = random.sample(range(len(self.testset)), min(subsample_size, len(self.testset)))
        subsampled_testset = Subset(self.testset, test_indices)

        # Create a filtered dataset with only correctly classified samples
        correct_test_indices = self._get_correctly_classified_indices(
            subsampled_testset, 
            batch_size=self.training_hypers['batch_size']
        )

        if len(correct_test_indices) > 0:
            correct_testset = Subset(subsampled_testset, correct_test_indices)
            loader_test_correct = DataLoader(
                correct_testset, 
                batch_size=self.training_hypers['batch_size'], 
                shuffle=True
            )
        else:
            loader_test_correct = None

        # Loaders for all samples
        loader_test_all = DataLoader(
            subsampled_testset, 
            batch_size=self.training_hypers['batch_size'], 
            shuffle=True
        )

        # Train set - similar approach
        train_indices = random.sample(range(len(self.trainset)), min(subsample_size, len(self.trainset)))
        subsampled_trainset = Subset(self.trainset, train_indices)

        correct_train_indices = self._get_correctly_classified_indices(
            subsampled_trainset,
            batch_size=self.training_hypers['batch_size']
        )

        if len(correct_train_indices) > 0:
            correct_trainset = Subset(subsampled_trainset, correct_train_indices)
            loader_train_correct = DataLoader(
                correct_trainset,
                batch_size=self.training_hypers['batch_size'],
                shuffle=True
            )
        else:
            loader_train_correct = None

        loader_train_all = DataLoader(
            subsampled_trainset,
            batch_size=self.training_hypers['batch_size'],
            shuffle=True
        )

        # Compute Lipschitz estimates
        model_to_eval = self.model_ddp

        # All samples
        metrics['lipschitz_estimation_train'] = estimate_lipschitz_gradient_norm_stream(
            model_to_eval, loader_train_all, device=self.rank
        )
        metrics['lipschitz_estimation_test'] = estimate_lipschitz_gradient_norm_stream(
            model_to_eval, loader_test_all, device=self.rank
        )

        # Correctly classified only
        if loader_train_correct is not None:
            metrics['lipschitz_estimation_train_correct'] = estimate_lipschitz_gradient_norm_stream(
                model_to_eval, loader_train_correct, device=self.rank
            )
        else:
            metrics['lipschitz_estimation_train_correct'] = None

        if loader_test_correct is not None:
            metrics['lipschitz_estimation_test_correct'] = estimate_lipschitz_gradient_norm_stream(
                model_to_eval, loader_test_correct, device=self.rank
            )
        else:
            metrics['lipschitz_estimation_test_correct'] = None

        # Print summary
        print(f"   Lipschitz (test, all): max={metrics['lipschitz_estimation_test']['max_gradient_norm']:.4f}, "
              f"avg={metrics['lipschitz_estimation_test']['average_gradient_norm']:.4f}")
        if metrics['lipschitz_estimation_test_correct'] is not None:
            print(f"   Lipschitz (test, correct): max={metrics['lipschitz_estimation_test_correct']['max_gradient_norm']:.4f}, "
                  f"avg={metrics['lipschitz_estimation_test_correct']['average_gradient_norm']:.4f}")

        print(f"   Lipschitz time: {(time.time() - start_time):.2f} seconds")
    
    def _add_binarity_metrics(self, metrics, local_eval_train, local_eval_test):
        """Add binarity metrics for lin_pen architecture."""
        print("6. Computing binarity metrics...")
        start_time = time.time()
        
        # Sample data for binarity computation
        sample_size = min(self.MAX_SAMPLES_BINARITY, len(local_eval_train['y_label']))
        sample_indices = np.random.choice(len(local_eval_train['y_label']), sample_size, replace=False)
        sampled_eval_train = {
            'x_penultimate': local_eval_train['x_penultimate'][sample_indices]
        }
        metrics['binarity_train'] = get_binarity_metrics(sampled_eval_train)
        
        if local_eval_test is not None:
            sample_size_test = min(self.MAX_SAMPLES_BINARITY, len(local_eval_test['y_label']))
            sample_indices_test = np.random.choice(len(local_eval_test['y_label']), sample_size_test, replace=False)
            sampled_eval_test = {
                'x_penultimate': local_eval_test['x_penultimate'][sample_indices_test]
            }
            metrics['binarity_test'] = get_binarity_metrics(sampled_eval_test)
        
        print(f"   Time: {(time.time()-start_time):.2f} seconds")
    
    def _get_correctly_classified_indices(self, dataset, batch_size=256):
        """
        Get indices of correctly classified samples from a dataset.

        Args:
            dataset: PyTorch dataset or Subset
            batch_size: Batch size for evaluation

        Returns:
            list: Indices of correctly classified samples within the dataset
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model_ddp.eval()

        correct_indices = []
        current_idx = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.rank)
                labels = labels.to(self.rank)

                outputs = self.model_ddp(images)
                if isinstance(outputs, dict):
                    logits = outputs['x_output']
                else:
                    logits = outputs

                predictions = torch.argmax(logits, dim=1)
                correct_mask = (predictions == labels).cpu().numpy()

                # Get indices of correct predictions in this batch
                batch_correct_indices = np.where(correct_mask)[0]
                # Convert to dataset indices
                dataset_indices = batch_correct_indices + current_idx
                correct_indices.extend(dataset_indices.tolist())

                current_idx += len(labels)

        return correct_indices
    # =========================================================================
    # Distributed Computation Helpers
    # =========================================================================
    
    def _compute_collapse_metrics_distributed(self, local_eval):
        """
        Compute collapse metrics in a distributed manner with simplified logic.
        """
        x_penultimate = local_eval["x_penultimate"]
        y_predicted = local_eval["y_predicted"]
        feature_dim = x_penultimate.shape[1]

        # Get local unique classes
        local_classes = np.unique(y_predicted)

        # Initialize ranks_with_data for all code paths
        ranks_with_data = 1  # Default for non-distributed case
        local_has_data = len(y_predicted) > 0  

        if self.distributed:
            # Check how many ranks have data
            has_data_tensor = torch.tensor([1 if local_has_data else 0], dtype=torch.int64).cuda(self.rank)
            dist.all_reduce(has_data_tensor, op=dist.ReduceOp.SUM)
            ranks_with_data = has_data_tensor.item()

            # Gather all classes only if multiple ranks have data
            if ranks_with_data > 1:
                all_local_classes = [None] * self.world_size
                dist.all_gather_object(all_local_classes, local_classes.tolist())

                # Correctly handle cases where some GPUs might not have any samples
                all_classes_list = [classes for classes in all_local_classes if classes]
                if not all_classes_list:
                    return None # No classes found across all GPUs

                all_classes = np.unique(np.concatenate(all_classes_list))
                num_classes = len(all_classes)

                local_size = len(y_predicted)
                size_tensor = torch.tensor([local_size], dtype=torch.int64).cuda(self.rank)
                dist.all_reduce(size_tensor, op=dist.ReduceOp.SUM)
                total_samples = size_tensor.item()
            else:
                # Only one rank has data (e.g., test set)
                if not local_has_data:
                    return None  # This rank has no data
                all_classes = local_classes
                num_classes = len(all_classes)
                total_samples = len(y_predicted)
                all_local_classes = None  # Not needed for single rank
        else:
            all_classes = local_classes
            num_classes = len(all_classes)
            total_samples = len(y_predicted)

        if num_classes == 0:
            return None

        if self.rank == 0 and len(y_predicted) > 0:
            print(f"   Total classes: {num_classes}")
            if self.distributed and ranks_with_data > 1 and 'all_local_classes' in locals() and all_local_classes is not None:
                for i, classes in enumerate(all_local_classes):
                    print(f"   GPU {i}: {len(classes)} unique classes")
            sys.stdout.flush()

        subsample_fraction = 0.2  
        subsample_threshold = 100000 

        if total_samples > subsample_threshold:
            if self.rank == 0 and len(y_predicted) > 0:
                print(f"   Large dataset ({total_samples:,} samples). Subsampling to {subsample_fraction*100:.0f}% for collapse metrics...")
                sys.stdout.flush()

            # Set random seed for reproducibility
            np.random.seed(42 + self.rank)

            # Perform stratified subsampling
            sampled_indices = []
            for class_label in local_classes:
                class_mask = (y_predicted == class_label)
                class_indices = np.where(class_mask)[0]
                n_class_samples = len(class_indices)
                n_to_sample = max(1, int(n_class_samples * subsample_fraction))

                if n_class_samples > 0:
                    sampled_class_indices = np.random.choice(class_indices, n_to_sample, replace=False)
                    sampled_indices.extend(sampled_class_indices)

            sampled_indices = np.array(sampled_indices)

            # Apply subsampling
            x_penultimate = x_penultimate[sampled_indices]
            y_predicted = y_predicted[sampled_indices]

            # Update size after subsampling
            local_size_after = len(sampled_indices)
            if self.distributed and ranks_with_data > 1:
                size_after_tensor = torch.tensor([local_size_after], dtype=torch.int64).cuda(self.rank)
                dist.all_reduce(size_after_tensor, op=dist.ReduceOp.SUM)
                total_after = size_after_tensor.item()
            else:
                total_after = local_size_after

            # Update local classes after subsampling
            local_classes = np.unique(y_predicted)

            if self.rank == 0 and len(y_predicted) > 0:
                print(f"   Subsampled to {total_after:,} samples")
                sys.stdout.flush()

        # Step 1: Compute class means globally
        local_class_sums = {}
        local_class_counts = {}

        for label in all_classes:
            mask = (y_predicted == label)
            if np.sum(mask) > 0:
                local_class_sums[int(label)] = np.sum(x_penultimate[mask], axis=0).astype(np.float64)
                local_class_counts[int(label)] = np.sum(mask)

        # Gather and compute global class means
        if self.distributed and ranks_with_data > 1:
            all_class_stats = [None] * self.world_size
            dist.all_gather_object(all_class_stats, (local_class_sums, local_class_counts))

            global_class_means = {}
            global_class_counts = {}
            for local_sums, local_counts in all_class_stats:
                for label in local_sums:
                    if label not in global_class_means:
                        global_class_means[label] = np.zeros(feature_dim, dtype=np.float64)
                        global_class_counts[label] = 0
                    global_class_means[label] += local_sums[label]
                    global_class_counts[label] += local_counts[label]

            # Finalize global means
            for label in global_class_means:
                global_class_means[label] = global_class_means[label] / global_class_counts[label]
        else:
            # Single rank or non-distributed case
            global_class_means = {}
            global_class_counts = {}
            for label, sum_val in local_class_sums.items():
                if label in local_class_counts and local_class_counts[label] > 0:
                    global_class_means[label] = sum_val / local_class_counts[label]
                    global_class_counts[label] = local_class_counts[label]

        # Compute global mean
        global_sum = np.sum([count * mean for mean, count in 
                           zip(global_class_means.values(), global_class_counts.values())], axis=0)
        global_count = np.sum(list(global_class_counts.values()))
        global_mean = global_sum / global_count

        # Step 2: Compute Sigma_B (between-class covariance)
        sigma_b = np.zeros((feature_dim, feature_dim), dtype=np.float64)
        class_mean_centered = []

        for label, class_mean in global_class_means.items():
            centered_mean = class_mean - global_mean
            class_mean_centered.append(centered_mean)
            sigma_b += np.outer(centered_mean, centered_mean) / num_classes

        # Step 3: Compute Sigma_W by processing each class
        if self.rank == 0:
            print(f"   Computing within-class covariances for {len(local_classes)} local classes...")
            sys.stdout.flush()

        # Each GPU computes sigma_w contribution for its local classes
        # Instead of storing a list, accumulate the sum directly
        local_sigma_w_sum = np.zeros((feature_dim, feature_dim), dtype=np.float64)
        local_class_count = 0

        # Process each class that exists on this GPU
        for idx, label in enumerate(local_classes):
            if self.rank == 0 and idx > 0 and idx % 50 == 0:
                print(f"   Processed {idx}/{len(local_classes)} classes")
                sys.stdout.flush()

            mask = (y_predicted == label)
            if np.sum(mask) > 0 and int(label) in global_class_means:
                class_data = x_penultimate[mask]
                class_mean = global_class_means[int(label)]

                # Compute class covariance using chunked approach
                chunk_size = min(50, max(10, 50000 // feature_dim))
                class_sigma_w = self._compute_class_covariance_chunked(
                    class_data, class_mean, chunk_size=chunk_size
                )

                # Add to running sum instead of list
                local_sigma_w_sum += class_sigma_w
                local_class_count += 1

                if feature_dim > 2000 and idx % 5 == 0:
                    torch.cuda.empty_cache()
                elif idx % 10 == 0:
                    torch.cuda.empty_cache()

        # No need to compute local mean here, just keep the sum
        if local_class_count == 0:
            local_sigma_w_sum = np.zeros((feature_dim, feature_dim), dtype=np.float64)

        # Step 4: Aggregate sigma_w across all GPUs
        if self.distributed and ranks_with_data > 1:
            sigma_w_sum_tensor = torch.from_numpy(local_sigma_w_sum).cuda(self.rank)
            count_tensor = torch.tensor([local_class_count], dtype=torch.float64).cuda(self.rank)

            # Sum contributions from all GPUs
            dist.all_reduce(sigma_w_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

            # Compute final sigma_w as average
            total_class_count = count_tensor.item()
            if total_class_count > 0:
                sigma_w = sigma_w_sum_tensor.cpu().numpy() / total_class_count
            else:
                sigma_w = np.eye(feature_dim, dtype=np.float64)

            # Clean up
            del sigma_w_sum_tensor, count_tensor
            torch.cuda.empty_cache()
        else:
            # Single rank case
            if local_class_count > 0:
                sigma_w = local_sigma_w_sum / local_class_count
            else:
                sigma_w = np.eye(feature_dim, dtype=np.float64)
        
        sigma_w_mean = np.mean(sigma_w)

        # Step 5: Final computations 
        if self.rank == 0:
            # Compute within-class variation
            try:
                sigma_b_pinv = np.linalg.pinv(sigma_b + np.eye(feature_dim) * 1e-8)
                within_class_variation = np.trace(sigma_w @ sigma_b_pinv) / num_classes
            except:
                within_class_variation = 0.0

            within_class_variation_weighted = within_class_variation / feature_dim

            # Compute angle-based metrics
            n_classes = len(class_mean_centered)

            # Compute cosine similarities between class means
            cosines = []
            cosines_max = []

            # Sample pairs if too many classes
            max_pairs = 50000
            total_pairs = n_classes * (n_classes - 1) // 2

            if total_pairs > max_pairs and n_classes > 1:
                np.random.seed(42)  # For reproducibility
                pairs = []
                for _ in range(max_pairs):
                    i, j = np.random.choice(n_classes, 2, replace=False)
                    if i > j:
                        i, j = j, i
                    pairs.append((i, j))
            else:
                pairs = [(i, j) for i in range(n_classes) for j in range(i + 1, n_classes)]

            class_mean_centered_array = np.stack(class_mean_centered, axis=0) if class_mean_centered else np.array([])

            for i, j in pairs:
                vec_i = class_mean_centered_array[i]
                vec_j = class_mean_centered_array[j]
                norm_product = np.linalg.norm(vec_i) * np.linalg.norm(vec_j)
                cosine_sim = np.dot(vec_i, vec_j) / norm_product if norm_product != 0 else 0.0
                cosines.append(cosine_sim)
                cosines_max.append(np.abs(cosine_sim + 1.0 / (n_classes - 1)))

            equiangular = np.std(cosines) if cosines else 0.0
            maxangle = np.mean(cosines_max) if cosines_max else 0.0

            # Compute equinorm
            class_norms = np.linalg.norm(class_mean_centered_array, axis=1) if n_classes > 0 else np.array([])
            equinorm = np.std(class_norms) / np.mean(class_norms) if len(class_norms) > 0 and np.mean(class_norms) != 0 else 0.0

            return {
                "within_class_variation": within_class_variation,
                "within_class_variation_weighted": within_class_variation_weighted,
                "equiangular": equiangular,
                "maxangle": maxangle,
                "equinorm": equinorm,
                "sigma_w": sigma_w_mean,
            }

        return None
    
    
    def _compute_class_covariance_chunked(self, class_data, class_mean, chunk_size=50):
        """
        Compute class covariance matrix in chunks to avoid memory overflow.
        """
        feature_dim = class_data.shape[1]
        n_samples = len(class_data)

        # Accumulator for sum of outer products (unnormalized)
        sum_outer_products = np.zeros((feature_dim, feature_dim), dtype=np.float64)

        # Process samples in chunks
        for chunk_start in range(0, n_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_samples)
            chunk_data = class_data[chunk_start:chunk_end]

            # Compute differences for the chunk
            diff_chunk = chunk_data - class_mean[np.newaxis, :]

            # Accumulate outer products (unnormalized)
            chunk_contribution = diff_chunk.T @ diff_chunk
            sum_outer_products += chunk_contribution

            # Clear memory periodically
            if chunk_start % (chunk_size * 10) == 0:
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

        # Compute actual covariance (divide by n-1 for unbiased estimator)
        if n_samples > 1:
            covariance = sum_outer_products / (n_samples - 1)
        else:
            # Single sample - return zero matrix
            covariance = np.zeros((feature_dim, feature_dim), dtype=np.float64)

        return covariance

    def _compute_distributed_coeff_var(self, local_eval):
        """
        Compute coefficient of variation in a distributed manner.
        
        Uses float64 for numerical stability.
        """
        x_penultimate = local_eval["x_penultimate"]
        
        # Subsample if needed
        if x_penultimate.shape[0] > self.MAX_SAMPLES_COEFF_VAR:
            if self.rank == 0:
                print(f"   Subsampling from {x_penultimate.shape[0]:,} to {self.MAX_SAMPLES_COEFF_VAR:,} for coefficient of variation")
                sys.stdout.flush()
            indices = np.random.choice(x_penultimate.shape[0], self.MAX_SAMPLES_COEFF_VAR, replace=False)
            x_penultimate = x_penultimate[indices]
        
        norms = np.linalg.norm(x_penultimate, axis=1)
        abs_values = np.abs(x_penultimate)
        
        # Compute local statistics using float64 for stability
        local_norm_sum = np.sum(norms).astype(np.float64)
        local_norm_sq_sum = np.sum(norms**2).astype(np.float64)
        local_abs_sum = np.sum(abs_values).astype(np.float64)
        local_abs_sq_sum = np.sum(abs_values**2).astype(np.float64)
        local_count = float(len(norms))
        
        if self.distributed:
            # Aggregate statistics
            stats = torch.tensor([local_norm_sum, local_norm_sq_sum, 
                                local_abs_sum, local_abs_sq_sum, local_count], 
                               dtype=torch.float64).cuda(self.rank)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            
            if self.rank == 0:
                total_count = stats[4].item()
                norm_mean = stats[0].item() / total_count
                norm_var = (stats[1].item() / total_count) - norm_mean**2
                norm_std = np.sqrt(max(0, norm_var))
                
                abs_mean = stats[2].item() / (total_count * x_penultimate.shape[1])
                abs_var = (stats[3].item() / (total_count * x_penultimate.shape[1])) - abs_mean**2
                abs_std = np.sqrt(max(0, abs_var))
                
                coeff_var_norm = norm_std / norm_mean if norm_mean != 0 else 0.0
                coeff_var_abs = abs_std / abs_mean if abs_mean != 0 else 0.0
                
                return coeff_var_abs, coeff_var_norm, norm_mean
        else:
            return get_coeff_var(local_eval)
        
        return None, None, None
    
    def _compute_distance_ratios_memory_efficient(self, x_penultimate, y_label, 
                                                 centroids, all_classes):
        """
        Compute distance ratios in a memory-efficient manner.
        Simplified version without redundant loops.
        """
        distance_ratios = []

        # Convert centroids to array for efficient computation
        centroid_labels = sorted(centroids.keys())
        centroids_array = np.array([centroids[l] for l in centroid_labels])

        # Process each sample directly
        for i, (sample, label) in enumerate(zip(x_penultimate, y_label)):
            if int(label) not in centroids:
                continue

            # Find own centroid index
            own_idx = centroid_labels.index(int(label))

            # Distance to own centroid
            distance_to_own = np.linalg.norm(sample - centroids[int(label)])

            if distance_to_own > 0 and len(centroid_labels) > 1:
                # Compute distances to all centroids
                distances = np.linalg.norm(sample - centroids_array, axis=1)

                # Mask out own centroid
                distances[own_idx] = float('inf')

                # Find minimum distance to other centroids
                min_distance_to_other = np.min(distances)

                # Compute ratio
                ratio = min_distance_to_other / distance_to_own
                distance_ratios.append(ratio)

            # Periodic memory cleanup for large datasets
            if i > 0 and i % 10000 == 0:
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

        return np.array(distance_ratios)


    def _compute_distance_ratios_standard(self, x_penultimate, y_label, 
                                         centroids, all_classes):
        """
        Standard distance ratios computation for smaller datasets.
        Simplified version using vectorized operations.
        """
        # Convert centroids to array
        centroid_labels = sorted(centroids.keys())
        centroids_array = np.array([centroids[l] for l in centroid_labels])
        class_to_idx = {c: i for i, c in enumerate(centroid_labels)}

        # Filter samples that have corresponding centroids
        valid_mask = np.array([int(label) in class_to_idx for label in y_label])
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return np.array([])

        # Compute all distances at once (more memory but faster)
        # Shape: (n_valid_samples, n_centroids)
        valid_samples = x_penultimate[valid_indices]
        all_distances = np.linalg.norm(
            valid_samples[:, np.newaxis, :] - centroids_array[np.newaxis, :, :], 
            axis=2
        )

        # Compute ratios for each valid sample
        distance_ratios = np.zeros(len(valid_indices))

        for i, (sample_idx, distances) in enumerate(zip(valid_indices, all_distances)):
            label = int(y_label[sample_idx])
            own_idx = class_to_idx[label]

            # Distance to own centroid
            distance_to_own = distances[own_idx]

            if distance_to_own > 0 and len(centroids_array) > 1:
                # Mask out own centroid
                other_distances = distances.copy()
                other_distances[own_idx] = float('inf')

                # Find minimum distance to other centroids
                min_distance_to_other = np.min(other_distances)

                # Compute ratio
                distance_ratios[i] = min_distance_to_other / distance_to_own
            else:
                distance_ratios[i] = 0.0

        return distance_ratios


    def _compute_distributed_distance_margins(self, local_eval):
        """
        Compute distance margins in a distributed manner.
        Simplified to automatically choose the appropriate method.
        """
        x_penultimate = local_eval["x_penultimate"]
        y_label = local_eval["y_label"]
        y_predicted = local_eval["y_predicted"]

        # Subsample if dataset is too large
        if len(y_label) > self.MAX_SAMPLES_DISTANCE_MARGINS:
            indices = np.random.choice(len(y_label), self.MAX_SAMPLES_DISTANCE_MARGINS, replace=False)
            x_penultimate = x_penultimate[indices]
            y_label = y_label[indices]
            y_predicted = y_predicted[indices]
            if self.rank == 0:
                print(f"   Subsampling distance margins to {self.MAX_SAMPLES_DISTANCE_MARGINS} samples per GPU")

        # Get unique classes across all GPUs
        unique_classes = np.unique(y_predicted)
        if self.distributed:
            all_classes_list = [None] * self.world_size
            dist.all_gather_object(all_classes_list, unique_classes.tolist())
            all_classes = np.unique(np.concatenate(all_classes_list))
        else:
            all_classes = unique_classes

        # Compute centroids
        centroids = self._compute_global_centroids(x_penultimate, y_predicted, all_classes)

        if centroids is None:
            return None

        # Choose method based on data size and memory constraints
        n_samples = x_penultimate.shape[0]
        n_centroids = len(centroids)
        feature_dim = x_penultimate.shape[1]

        # Estimate memory usage for vectorized approach (in GB)
        memory_estimate_gb = (n_samples * n_centroids * 8) / 1e9  # float64

        # Use memory-efficient method if estimated memory > 1GB or feature dim is large
        if memory_estimate_gb > 0.1 or feature_dim > self.LARGE_FEATURE_DIM_THRESHOLD:
            if self.rank == 0:
                print(f"   Using memory-efficient method (estimated {memory_estimate_gb:.1f} GB)")
            distance_ratios = self._compute_distance_ratios_memory_efficient(
                x_penultimate, y_predicted, centroids, all_classes
            )
        else:
            if self.rank == 0:
                print(f"   Using standard vectorized method")
            distance_ratios = self._compute_distance_ratios_standard(
                x_penultimate, y_predicted, centroids, all_classes
            )

        # Aggregate results
        if self.distributed:
            local_ratios_sum = np.sum(distance_ratios) if len(distance_ratios) > 0 else 0.0
            local_ratios_count = len(distance_ratios)

            ratios_stats = torch.tensor([local_ratios_sum, local_ratios_count], 
                                       dtype=torch.float64).cuda(self.rank)
            dist.all_reduce(ratios_stats, op=dist.ReduceOp.SUM)

            if self.rank == 0:
                total_sum = ratios_stats[0].item()
                total_count = int(ratios_stats[1].item())
                average_distance_ratio = total_sum / total_count if total_count > 0 else 0.0

                return {
                    "distance_ratios": None,  # Don't store full array in distributed mode
                    "average_distance_ratio": average_distance_ratio,
                }
        else:
            average_ratio = np.mean(distance_ratios) if len(distance_ratios) > 0 else 0.0
            return {
                "distance_ratios": distance_ratios,
                "average_distance_ratio": average_ratio,
            }

        return None
    
    def _compute_global_centroids(self, x_penultimate, y_label, all_classes):
        """Compute global centroids across all GPUs."""
        # Compute local class statistics
        local_class_sums = {}
        local_class_counts = {}

        for label in all_classes:
            mask = (y_label == label)
            if np.sum(mask) > 0:
                local_class_sums[int(label)] = np.sum(x_penultimate[mask], axis=0).astype(np.float64)
                local_class_counts[int(label)] = np.sum(mask)

        if self.distributed:
            # Gather and aggregate
            all_class_stats = [None] * self.world_size
            dist.all_gather_object(all_class_stats, (local_class_sums, local_class_counts))

            # Compute global centroids
            global_centroids = {}
            for local_sums, local_counts in all_class_stats:
                for label in local_sums:
                    if label not in global_centroids:
                        global_centroids[label] = {
                            'sum': np.zeros_like(local_sums[label], dtype=np.float64),
                            'count': 0
                        }
                    global_centroids[label]['sum'] += local_sums[label]
                    global_centroids[label]['count'] += local_counts[label]

            # Finalize centroids
            centroids = {}
            for label, stats in global_centroids.items():
                if stats['count'] > 0:
                    centroids[label] = (stats['sum'] / stats['count']).astype(np.float32)

            return centroids
        else:
            # Non-distributed case
            centroids = {}
            for label in all_classes:
                mask = (y_label == label)
                if np.sum(mask) > 0:
                    centroids[int(label)] = np.mean(x_penultimate[mask], axis=0)
            return centroids    
    
    
    # =========================================================================
    # Results Aggregation
    # =========================================================================
    
    def _aggregate_results(self, res_list, total_training_time, epoch_times, converged):
        """
        Aggregate results from all epochs into final results dictionary.
        
        Returns:
            dict: Aggregated results with all metrics stacked appropriately
        """
        res_dict_stack = {}
        if not res_list:
            return res_dict_stack
        
        # Stack results for each key
        for key in res_list[0].keys():
            if isinstance(res_list[0][key], dict):
                res_dict_stack[key] = self._aggregate_nested_dict(res_list, key)
            else:
                res_dict_stack[key] = self._aggregate_values(res_list, key)
        
        # Add training time metrics
        res_dict_stack['training_time'] = np.array(total_training_time / 60)
        res_dict_stack['training_time_epochs'] = np.array(epoch_times)
        
        # Add convergence information
        if self.converged:
            res_dict_stack['convergence_epoch'] = self.convergence_epoch
            res_dict_stack['accuracy_test_converged'] = self.accuracy_test_converged
        else:
            res_dict_stack['accuracy_test_converged'] = False
            res_dict_stack['convergence_epoch'] = False
        
        # Add last epoch metrics
        self._add_last_epoch_values(res_dict_stack, res_list)
        
        # Add penultimate layers if stored
        if self.store_penultimate and hasattr(self, 'penultimate_train') and self.penultimate_train is not None:
            res_dict_stack['penultimate_train'] = self.penultimate_train.reshape(1, self.penultimate_train.shape[0], -1)
        if self.store_penultimate and hasattr(self, 'penultimate_test') and self.penultimate_test is not None:
            res_dict_stack['penultimate_test'] = self.penultimate_test.reshape(1, self.penultimate_test.shape[0], -1)
        
        return res_dict_stack
    
    def _aggregate_nested_dict(self, res_list, key):
        """Aggregate nested dictionary values."""
        aggregated = {}
        for key2 in res_list[0][key].keys():
            valid_items = [res_epoch[key][key2] for res_epoch in res_list 
                          if res_epoch.get(key) is not None and res_epoch[key].get(key2) is not None]
            if not valid_items:
                aggregated[key2] = None
                continue
            
            if isinstance(valid_items[0], dict):
                # Handle triple-nested dictionaries
                aggregated[key2] = {}
                for key3 in valid_items[0].keys():
                    valid_items_key3 = [item[key3] for item in valid_items 
                                       if item is not None and item.get(key3) is not None]
                    if not valid_items_key3:
                        aggregated[key2][key3] = None
                        continue
                    aggregated[key2][key3] = self._stack_values(valid_items_key3, f"{key}->{key2}->{key3}")
            else:
                aggregated[key2] = self._stack_values(valid_items, f"{key}->{key2}")
        
        return aggregated
    
    def _aggregate_values(self, res_list, key):
        """Aggregate non-dictionary values."""
        valid_items = [res_epoch[key] for res_epoch in res_list if res_epoch.get(key) is not None]
        if not valid_items:
            return None
        return self._stack_values(valid_items, key)
    
    def _stack_values(self, values, name):
        """Stack values with error handling."""
        try:
            return np.vstack(values)
        except ValueError as e:
            print(f"ValueError during stacking for {name}: {e}")
            print(f"Shapes: {[item.shape if hasattr(item, 'shape') else type(item) for item in values]}")
            return values
    
    def _add_last_epoch_values(self, res_dict_stack, res_list):
        """Add metrics that are only computed in the last epoch."""
        last_res = res_list[-1] if res_list else {}

        # Original metrics (for backward compatibility)
        res_dict_stack['deepfool_score'] = last_res.get('deepfool_score', 0)
        res_dict_stack['entropy_train'] = last_res.get('entropy_train', 0)
        res_dict_stack['entropy_test'] = last_res.get('entropy_test', 0)
        res_dict_stack['lipschitz_estimation_train'] = last_res.get('lipschitz_estimation_train', 0)
        res_dict_stack['lipschitz_estimation_test'] = last_res.get('lipschitz_estimation_test', 0)

        # New metrics - separate all vs correct
        res_dict_stack['deepfool_score_all'] = last_res.get('deepfool_score_all', 0)
        res_dict_stack['deepfool_score_correct'] = last_res.get('deepfool_score_correct', None)
        res_dict_stack['deepfool_n_samples'] = last_res.get('deepfool_n_samples', 0)
        res_dict_stack['deepfool_n_correct'] = last_res.get('deepfool_n_correct', 0)

        res_dict_stack['lipschitz_estimation_train_correct'] = last_res.get('lipschitz_estimation_train_correct', None)
        res_dict_stack['lipschitz_estimation_test_correct'] = last_res.get('lipschitz_estimation_test_correct', None)
