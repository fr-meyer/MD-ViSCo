# Standard library imports
import os

# Third-party imports
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def cleanup():
    """Cleanup DDP process group"""
    if dist.is_initialized():
        dist.destroy_process_group()

def print_memory_stats(rank, location):
    """Print GPU memory statistics"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(rank) / (1024 * 1024)
        cached = torch.cuda.memory_reserved(rank) / (1024 * 1024)
        print(f"Rank {rank} at {location}:")
        print(f"Allocated: {allocated:.2f}MB")
        print(f"Cached: {cached:.2f}MB")

def setup_ddp(rank: int, world_size: int, master_addr: str, master_port: str, ddp_backend: str = 'nccl'):
    """Setup DDP environment"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    try:
        if torch.cuda.is_available():
            backend = ddp_backend
            dist.init_process_group(backend, rank=rank, world_size=world_size)
            torch.cuda.set_device(rank)
        else:
            backend = 'gloo'
            dist.init_process_group(backend, rank=rank, world_size=world_size)
    except Exception as e:
        print(f"Failed to initialize process group with {backend}: {e}")
        if backend == 'nccl':
            print("Falling back to Gloo backend")
            if dist.is_initialized():
                dist.destroy_process_group()
            dist.init_process_group("gloo", rank=rank, world_size=world_size)

def create_ddp_model(model, rank: int, find_unused_parameters: bool = False, static_graph: bool = True):
    """Create DDP model with appropriate settings"""
    return DDP(
        model,
        device_ids=[rank],
        find_unused_parameters=find_unused_parameters,
        static_graph=static_graph
    )

def create_ddp_dataloaders(dataset, val_dataset, test_dataset, rank: int, world_size: int, batch_size: int, dataloader_settings: dict):
    """Create train/val dataloaders with DDP samplers
    
    Args:
        dataset: Training dataset
        val_dataset: Validation dataset
        rank: Process rank
        world_size: Number of processes
        batch_size: Batch size
        dataloader_settings: Additional dataloader settings
    """
    # Create samplers
    train_sampler = DistributedSampler(dataset, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, rank=rank, shuffle=False)

    # Create dataloaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        **dataloader_settings
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        **dataloader_settings
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        **dataloader_settings
    )
    return train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler