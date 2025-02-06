import torch
import numpy as np
import torch.distributed as dist


def load_checkpoint(filename):
    """Load a checkpoint from file."""
    checkpoint = torch.load(filename, map_location='cpu')
    return checkpoint


def save_checkpoint(state, filename):
    """Save a checkpoint to file."""
    torch.save(state, filename)


def gather_dict_outputs_ddp(local_eval_train, rank, world_size):
    """
    Gather dictionary outputs from all processes in distributed training.
    """
    gathered_eval_train = {key: [] for key in local_eval_train.keys()}
    for key, local_data in local_eval_train.items():
        if np.isscalar(local_data):
            local_data_tensor = torch.tensor([local_data], dtype=torch.float32).to(rank)
            gathered_data = [torch.zeros_like(local_data_tensor) for _ in range(world_size)]
            dist.gather(local_data_tensor, gathered_data if rank == 0 else [], dst=0)
            if rank == 0:
                gathered_eval_train[key] = gathered_data[0].cpu().numpy()[0]
            continue

        if torch.is_tensor(local_data):
            local_data_tensor = local_data.to(rank)
        else:
            try:
                local_data_tensor = torch.tensor(local_data, dtype=torch.float32).to(rank)
            except (ValueError, TypeError):
                gathered_eval_train[key] = local_data
                continue

        gathered_data = [torch.zeros_like(local_data_tensor) for _ in range(world_size)]
        dist.gather(local_data_tensor, gathered_data if rank == 0 else [], dst=0)
        if rank == 0:
            gathered_data = torch.cat(gathered_data, dim=0)
            gathered_eval_train[key] = gathered_data.cpu().numpy()
    return gathered_eval_train if rank == 0 else None


