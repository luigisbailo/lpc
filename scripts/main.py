#!/usr/bin/env python

import os
# Set this before importing any other modules
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import datetime
import importlib
import os
import pickle
import sys
from collections import defaultdict
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
import yaml

import lpc.networks as networks
from lpc.trainer import Trainer


# =============================================================================
# Utility Functions
# =============================================================================

def parse_bool_arg(arg_value: Optional[str], arg_name: str) -> bool:
    """
    Convert a command-line string argument to a boolean.
    If the argument is not provided or is invalid, defaults to False.
    """
    if arg_value is None:
        return False
    arg_str = arg_value.lower()
    if arg_str not in ['true', 'false']:
        print(f"Invalid value for {arg_name}. Defaulting to False.")
        return False
    return arg_str == 'true'


def convert_bool(dictionary: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Convert any string values 'true'/'false' in the dictionary to booleans.
    """
    for key, value in dictionary.items():
        if isinstance(value, str):
            if value.lower() == 'true':
                dictionary[key] = True
            elif value.lower() == 'false':
                dictionary[key] = False
    return dictionary


def parse_config(config_file: str) -> Any:
    """
    Parse the given YAML config file.
    """
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"File not found: {config_file}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)


def update_training_hypers(training_hypers: Dict[str, Any], args: argparse.Namespace) -> None:
    """
    Update the training hyperparameters based on command-line arguments.
    """
    if args.lr is not None:
        # If the lr starts with a decimal point, prepend a 0.
        lr_str = args.lr if not args.lr.startswith('.') else '0' + args.lr
        training_hypers['lr'] = float(lr_str)
    
    if args.weight_decay is not None:
        # If the weight_decay starts with a decimal point, prepend a 0.
        weight_decay_str = args.weight_decay if not args.weight_decay.startswith('.') else '0' + args.weight_decay
        training_hypers['weight_decay'] = float(weight_decay_str)

def process_penultimate_nodes(penultimate_arg: Optional[str], architecture: Dict[str, Any]) -> Tuple[Optional[Any], str]:
    """
    Determine which penultimate configuration to use based on the command-line
    argument. Returns a tuple (penultimate_nodes, prefix) for later use in constructing the filename.
    """
    hypers = architecture.get('hypers', {})
    if penultimate_arg == 'wide':
        return hypers.get('penultimate_nodes_wide'), 'wide_'
    elif penultimate_arg == 'narrow':
        return hypers.get('penultimate_nodes_narrow'), 'narrow_'
    else:
        return hypers.get('penultimate_nodes'), ''

def get_transforms(dataset_name: str, base_sample: Any, mean: Any, std: Any) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Return training and test transforms based on the dataset.
    For ImageNet, we use a standard augmentation pipeline, and for other datasets,
    we use the base_sample to determine crop size and add random crop with padding.
    """

    if dataset_name.lower() == 'imagenet':
        # For ImageNet, use a standard augmentation pipeline.
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        # Extract crop size from a sample for non-ImageNet datasets.
        try:
            # Assume the image tensor shape is (C, H, W); use H as crop size.
            crop_size = base_sample[0][0].shape[1]
        except Exception as e:
            raise ValueError(f"Error determining crop size from base_sample: {e}")

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size, padding=4),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return train_transform, test_transform


def compute_mean_std(dataset: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the per-channel mean and standard deviation of the dataset.
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data = next(iter(loader))[0].numpy()
    mean = np.mean(data, axis=(0, 2, 3))
    std = np.std(data, axis=(0, 2, 3))
    return mean, std


def merge_results(old_results: Any, new_results: Any) -> Any:
    """
    Recursively merge new results into the old results.
    Properly handles nested dictionaries and arrays at any depth.
    """
    # Case 1: Both are numpy arrays - concatenate them
    if isinstance(old_results, np.ndarray) and isinstance(new_results, np.ndarray):
        # Make sure shapes are compatible
        if old_results.ndim == new_results.ndim:
            try:
                return np.concatenate((old_results, new_results))
            except ValueError as e:
                print(f"Error concatenating arrays: {e}")
                print(f"old_shape: {old_results.shape}, new_shape: {new_results.shape}")
                # If concatenation fails, prefer new results
                return new_results
        else:
            return new_results
    
    # Case 2: Both are dictionaries - recursively merge
    elif isinstance(old_results, dict) and isinstance(new_results, dict):
        merged_dict = old_results.copy()  # Start with a copy of old_results
        
        # Loop through all keys in new_results
        for key, new_value in new_results.items():
            if key in old_results:
                # Special handling for training_hypers and architecture
                if key in ('training_hypers', 'architecture'):
                    if not isinstance(old_results[key], list):
                        merged_dict[key] = [old_results[key]]
                    if not isinstance(new_value, list):
                        new_value = [new_value]
                    merged_dict[key].extend(new_value)
                # Special handling for penultimate data
                elif key in ('penultimate_train', 'penultimate_test'):
                    merged_dict[key] = new_value
                # Recursively merge other dictionary values
                else:
                    merged_dict[key] = merge_results(old_results[key], new_value)
            else:
                # Key only in new_results
                merged_dict[key] = new_value
        
        return merged_dict
    
    # Case 3: Incompatible types - prefer the new value
    else:
        return new_results


def merge_with_existing_results(new_results: Dict[str, Any],
                               results_file: str,
                               training_hypers: Dict[str, Any],
                               architecture: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge new results with existing results, with proper handling for nested structures.
    Ensures epochs and other metrics are correctly concatenated.
    """
    final_results = new_results.copy()
    final_results['training_hypers'] = training_hypers
    final_results['architecture'] = architecture

    if not os.path.exists(results_file):
        return final_results

    try:
        # Load old results
        with open(results_file, 'rb') as f:
            old_results = pickle.load(f)
        
        # Check if we have epochs in both results
        if 'epochs' not in old_results or 'epochs' not in new_results:
            return {**old_results, **new_results, 'training_hypers': training_hypers, 'architecture': architecture}
        
        # Get epochs and determine which are new
        old_epochs = np.array(old_results['epochs']).flatten()
        new_epochs = np.array(new_results['epochs']).flatten()
        max_old_epoch = max(old_epochs) if len(old_epochs) > 0 else -1
        new_epoch_indices = [i for i, epoch in enumerate(new_epochs) if epoch > max_old_epoch]
        
        # If no new epochs, just return old results with updated metadata
        if not new_epoch_indices:
            old_results['training_hypers'] = training_hypers
            old_results['architecture'] = architecture
            return old_results
        
        # Function to filter arrays by epoch indices if they match epoch length
        def filter_by_epochs(value):
            """Filter array-like objects by epoch indices if applicable."""
            if not isinstance(value, np.ndarray):
                return value
            
            if value.ndim == 0:  # Scalar array
                return value
                
            try:
                if len(value) == len(new_epochs):
                    return value[new_epoch_indices]
            except (TypeError, AttributeError):
                pass
                
            return value
        
        # Function to recursively filter nested dictionaries
        def filter_nested_dict(d):
            """Recursively filter a nested dictionary."""
            if not isinstance(d, dict):
                return filter_by_epochs(d)
            
            filtered = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    filtered[k] = filter_nested_dict(v)
                else:
                    filtered[k] = filter_by_epochs(v)
            return filtered
        
        # Function to safely concatenate arrays or handle other types
        def safe_concat(old_val, new_val):
            """Safely concatenate arrays, handling various edge cases."""
            if not isinstance(old_val, np.ndarray) or not isinstance(new_val, np.ndarray):
                return new_val
                
            if old_val.ndim == 0 or new_val.ndim == 0:  # Scalar arrays
                return new_val
                
            try:
                return np.concatenate((old_val, new_val))
            except (ValueError, TypeError) as e:
                print(f"Concatenation error: {e} - Using new value")
                return new_val
        
        # Function to recursively merge dictionaries
        def recursive_merge(dict1, dict2):
            """Recursively merge two dictionaries with proper array concatenation."""
            result = dict1.copy()
            
            for k, v in dict2.items():
                if k in dict1:
                    if isinstance(v, dict) and isinstance(dict1[k], dict):
                        # Recursively merge nested dictionaries
                        result[k] = recursive_merge(dict1[k], v)
                    else:
                        # Concatenate arrays or prefer new value
                        result[k] = safe_concat(dict1[k], v)
                else:
                    # Key only in dict2
                    result[k] = v
            
            return result
        
        # Filter new results to only include new epochs
        filtered_new_results = {}
        for key, value in new_results.items():
            if key == 'epochs':
                filtered_new_results[key] = new_epochs[new_epoch_indices]
            elif isinstance(value, dict):
                filtered_new_results[key] = filter_nested_dict(value)
            else:
                filtered_new_results[key] = filter_by_epochs(value)
        
        # Start with a fresh result dictionary
        merged_results = {}
        
        # Explicitly handle epochs first - ensure they're properly concatenated
        merged_results['epochs'] = np.concatenate((old_epochs, filtered_new_results['epochs']))
        print(f"Old epochs: {old_epochs}")
        print(f"New filtered epochs: {filtered_new_results['epochs']}")
        print(f"Merged epochs: {merged_results['epochs']}")
        
        # Merge the rest of the metrics
        for key in set(list(old_results.keys()) + list(filtered_new_results.keys())):
            if key == 'epochs':
                continue  # Already handled
            
            if key in ('training_hypers', 'architecture'):
                # These will be set at the end
                continue
                
            if key in old_results and key in filtered_new_results:
                if isinstance(old_results[key], dict) and isinstance(filtered_new_results[key], dict):
                    # Recursively merge nested dictionaries
                    merged_results[key] = recursive_merge(old_results[key], filtered_new_results[key])
                else:
                    # Concatenate arrays or handle other types
                    merged_results[key] = safe_concat(old_results[key], filtered_new_results[key])
            elif key in filtered_new_results:
                merged_results[key] = filtered_new_results[key]
            else:
                merged_results[key] = old_results[key]
        
        # Set metadata
        merged_results['training_hypers'] = training_hypers
        merged_results['architecture'] = architecture
        
        return merged_results
        
    except Exception as e:
        print(f"Error during merge: {e}")
        import traceback
        traceback.print_exc()
        return final_results

def build_filename(base_name: str, flags: Dict[str, bool], penultimate_prefix: str, sample: Optional[str]) -> str:
    """
    Build a unique filename string based on the base model name and various boolean flags.
    The `flags` parameter is expected to be a dict mapping flag names to booleans.
    """
    name_parts = []
    if flags.get('scl'):
        name_parts.append("scl")
    if flags.get('arcface'):
        name_parts.append("arcface")
    if flags.get('l2_loss'):
        name_parts.append("lpc")

    # Concatenate the parts with underscores.
    filename = penultimate_prefix + ("_".join(name_parts + [base_name]) if name_parts else base_name)
    if sample:
        filename = f"{filename}_{sample}"
    return filename


def check_flag_file(flag_file: str, world_size: int) -> None:
    """
    Check for the existence of a flag file to avoid launching multiple training jobs.
    If the file exists, send a dummy tensor to all processes and exit.
    Otherwise, create the flag file.
    """
    if os.path.exists(flag_file):
        print(f"Flag file {flag_file} exists. Another training job might be running. Exiting.")
        for dst in range(world_size):
            dist.send(torch.tensor([1]), dst=dst)
        sys.exit(1)
    else:
        with open(flag_file, 'w') as f:
            f.write('')


def print_training_info(filename: str, lr: float, world_size: int) -> None:
    """
    Print basic training information.
    """
    print(f"Training {filename}")
    print("Learning rate:", lr)
    print("World size:", world_size)


def setup_distributed_training() -> None:
    """
    Initialize the torch.distributed process group using the NCCL backend.
    """
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=7200))


# =============================================================================
# Main Worker Function
# =============================================================================

def main_worker(args: argparse.Namespace) -> None:
    # Parse configuration.
    config = parse_config(args.config)
    architecture = config['architecture']
    training_hypers = config['training']['hypers']
    dataset_name = config['dataset']['name']

    # Update training hyperparameters based on command-line options.
    update_training_hypers(training_hypers, args)

    # Convert string flags into booleans.
    flags = {
        'l2_loss': parse_bool_arg(args.l2_loss, "l2 loss"),
        'scl': parse_bool_arg(args.scl, "scl"),
        'arcface': parse_bool_arg(args.arcface, "arcface"),
    }
    store_penultimate = parse_bool_arg(args.store_penultimate, "store penultimate")

    # Determine penultimate configuration.
    penultimate_nodes, penultimate_prefix = process_penultimate_nodes(args.penultimate_nodes, architecture)

    # Create a basic transform for computing the dataset mean and std.
    base_transform = transforms.ToTensor()

    torchvision_module = importlib.import_module("torchvision.datasets")
    torch_dataset = getattr(torchvision_module, dataset_name)

    # For ImageNet, use split and hardcoded normalization values.
    if dataset_name.lower() == 'imagenet':
        # Use split keyword for ImageNet and omit download (you need to prepare it manually)
        trainset = torch_dataset(str(args.dataset_dir), split='train', transform=transforms.ToTensor())
    else:
        trainset = torch_dataset(str(args.dataset_dir), train=True, download=True, transform=transforms.ToTensor())

    # For ImageNet, bypass compute_mean_std() by using the hardcoded mean and std.
    if dataset_name.lower() == 'imagenet':
        trainset_mean = [0.485, 0.456, 0.406]
        trainset_std  = [0.229, 0.224, 0.225]
    else:
        trainset_mean, trainset_std = compute_mean_std(trainset)

    transform_train, transform_test = get_transforms(dataset_name, trainset, trainset_mean, trainset_std)

    # Reload datasets with proper transforms
    if dataset_name.lower() == 'imagenet':
        trainset = torch_dataset(str(args.dataset_dir), split='train', transform=transform_train)
        testset  = torch_dataset(str(args.dataset_dir), split='val', transform=transform_test)
    else:
        trainset = torch_dataset(str(args.dataset_dir), train=True, download=True, transform=transform_train)
        testset = torch_dataset(str(args.dataset_dir), train=False, download=True, transform=transform_test)

    # Determine input dimensions and number of classes.
    input_dims = trainset[0][0].shape[1]
    num_classes = len(trainset.classes)

    print("len(trainset.classes):", len(trainset.classes))
    print("len(testset.classes):", len(testset.classes))


    # Convert any boolean strings in the config to actual booleans.
    training_hypers = convert_bool(training_hypers)
    architecture['hypers'] = convert_bool(architecture.get('hypers', {}))

    # Set up distributed training.
    setup_distributed_training()

    world_size = dist.get_world_size() if dist.is_initialized() else torch.cuda.device_count()
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Build a unique filename based on the architecture and flag settings.
    filename = build_filename(args.architecture_type, flags, penultimate_prefix, args.sample)

    flag_file = os.path.join(args.results_dir, f"training_{filename}.flag")
    checkpoint_file = os.path.join(args.results_dir, f"checkpoint_{filename}.pth.tar")
    results_file = os.path.join(args.results_dir, f"{filename}.pkl")

    bias_output = not flags.get('arcface')

    print('bias output: ', bias_output)
    
    if rank == 0:
        # check_flag_file(flag_file, world_size)
        print_training_info(filename, training_hypers['lr'], world_size)

    try:
        # Create the network and trainer objects.
        classifier = getattr(networks, architecture['backbone'])(
            architecture_type=args.architecture_type,
            architecture=architecture,
            num_classes=num_classes,
            penultimate_nodes=penultimate_nodes,
            input_dims=input_dims,
            bias_output=bias_output
        )

        trainer = Trainer(
            network=classifier,
            architecture=architecture,
            trainset=trainset,
            testset=testset,
            training_hypers=training_hypers,
            architecture_type=args.architecture_type,
            l2_loss=flags.get('l2_loss'),
            arcface_loss=flags.get('arcface'),
            scl=flags.get('scl'),
            store_penultimate=store_penultimate,
            verbose=True
        )

        results = trainer.fit(checkpoint_file, rank, world_size, distributed=(world_size > 1))

        if rank == 0:
            final_results = merge_with_existing_results(results, results_file, training_hypers, architecture)
            with open(results_file, 'wb') as f:
                pickle.dump(final_results, f)
    finally:
        if rank == 0 and os.path.exists(flag_file):
            os.remove(flag_file)

    dist.destroy_process_group()


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'NOT SET')}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--architecture-type', required=True)
    parser.add_argument('--sample', required=False)
    parser.add_argument('--lr', required=False)
    parser.add_argument('--weight-decay', required=False)
    parser.add_argument('--l2-loss', required=False)
    parser.add_argument('--arcface', required=False)
    parser.add_argument('--scl', required=False)
    parser.add_argument('--store-penultimate', required=False)
    parser.add_argument('--penultimate-nodes', required=False)
    args = parser.parse_args()
    main_worker(args)


if __name__ == '__main__':
    main()
