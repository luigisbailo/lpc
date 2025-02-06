#!/usr/bin/env python

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
    """
    try:
        # Assume the image tensor shape is (C, H, W)
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
    If both values are arrays, they are concatenated.
    If the key corresponds to 'training_hypers' or 'architecture', the values are
    accumulated in a list.
    """
    if isinstance(old_results, np.ndarray) and isinstance(new_results, np.ndarray):
        return np.concatenate((old_results, new_results))
    elif isinstance(old_results, dict) and isinstance(new_results, dict):
        for key, new_value in new_results.items():
            if key in old_results:
                if key in ('training_hypers', 'architecture'):
                    if not isinstance(old_results[key], list):
                        old_results[key] = [old_results[key]]
                    if not isinstance(new_value, list):
                        new_value = [new_value]
                    old_results[key] += new_value
                else:
                    old_results[key] = merge_results(old_results[key], new_value)
            elif key in ('penultimate_train', 'penultimate_test'):
                old_results[key] = new_value
        return old_results
    else:
        return new_results


def merge_with_existing_results(new_results: Dict[str, Any],
                                results_file: str,
                                training_hypers: Dict[str, Any],
                                architecture: Dict[str, Any]) -> Dict[str, Any]:
    """
    If a results file already exists, merge the new results with the previous ones.
    """
    final_results = new_results.copy()
    final_results['training_hypers'] = training_hypers
    final_results['architecture'] = architecture

    if os.path.exists(results_file):
        try:
            with open(results_file, 'rb') as f:
                old_results = pickle.load(f)
            final_results = merge_results(old_results, new_results)
        except Exception as e:
            print(f"Error loading results file: {e}")
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
    if flags.get('cosface'):
        name_parts.append("cosface")
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
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=3600))


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
        'cosface': parse_bool_arg(args.cosface, "cosface"),
        'arcface': parse_bool_arg(args.arcface, "arcface"),
    }
    store_penultimate = parse_bool_arg(args.store_penultimate, "store penultimate")

    # Determine penultimate configuration.
    penultimate_nodes, penultimate_prefix = process_penultimate_nodes(args.penultimate_nodes, architecture)

    # Create a basic transform for computing the dataset mean and std.
    base_transform = transforms.ToTensor()
    torchvision_module = importlib.import_module("torchvision.datasets")
    torch_dataset = getattr(torchvision_module, dataset_name)

    # Load the training set for computing statistics.
    trainset = torch_dataset(str(args.dataset_dir), train=True, download=True, transform=base_transform)
    trainset_mean, trainset_std = compute_mean_std(trainset)

    transform_train, transform_test = get_transforms(dataset_name, trainset, trainset_mean, trainset_std)

    # Reload the datasets with proper transforms.
    trainset = torch_dataset(str(args.dataset_dir), train=True, download=True, transform=transform_train)
    testset = torch_dataset(str(args.dataset_dir), train=False, download=True, transform=transform_test)

    # Determine input dimensions and number of classes.
    input_dims = trainset[0][0].shape[1]
    num_classes = len(set(trainset.classes))

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

    if rank == 0:
        check_flag_file(flag_file, world_size)
        print_training_info(filename, training_hypers['lr'], world_size)

    try:
        # Create the network and trainer objects.
        classifier = getattr(networks, architecture['backbone'])(
            architecture_type=args.architecture_type,
            architecture=architecture,
            num_classes=num_classes,
            penultimate_nodes=penultimate_nodes,
            input_dims=input_dims,
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
            cosface_loss=flags.get('cosface'),
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--architecture-type', required=True)
    parser.add_argument('--sample', required=False)
    parser.add_argument('--lr', required=False)
    parser.add_argument('--l2-loss', required=False)
    parser.add_argument('--cosface', required=False)
    parser.add_argument('--arcface', required=False)
    parser.add_argument('--scl', required=False)
    parser.add_argument('--store-penultimate', required=False)
    parser.add_argument('--penultimate-nodes', required=False)
    args = parser.parse_args()
    main_worker(args)


if __name__ == '__main__':
    main()
