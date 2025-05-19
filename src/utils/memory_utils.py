"""
Memory Utilities

This module provides utility functions for memory management
and optimization in the Brain-Inspired Neural Network.
"""

import torch
import gc
import numpy as np

def optimize_memory_usage(model, device=None):
    """
    Optimize memory usage for the model.
    
    Args:
        model: The neural network model
        device: The device (CPU/GPU) the model is on
        
    Returns:
        model: The optimized model
    """
    # Clear CUDA cache
    if device is not None and device.type == 'cuda':
        torch.cuda.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Run garbage collection
    gc.collect()
    
    return model

def print_gpu_memory_status():
    """
    Print the current GPU memory usage status.
    
    Returns:
        dict: Dictionary with memory usage statistics
    """
    if not torch.cuda.is_available():
        print("CUDA not available")
        return {}
    
    # Get memory usage statistics
    stats = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # GB
        max_allocated = torch.cuda.max_memory_allocated(i) / (1024 ** 3)  # GB
        
        stats[f'cuda:{i}'] = {
            'allocated_GB': allocated,
            'reserved_GB': reserved,
            'max_allocated_GB': max_allocated
        }
        
        print(f"CUDA:{i} Memory: {allocated:.2f}GB allocated, "
              f"{reserved:.2f}GB reserved, {max_allocated:.2f}GB max allocated")
    
    return stats

def tensor_shape_fix(tensor, target_shape):
    """
    Fix tensor shape mismatches by reshaping or padding.
    
    Args:
        tensor (torch.Tensor): Input tensor
        target_shape (tuple): Target shape
        
    Returns:
        torch.Tensor: Reshaped or padded tensor
    """
    current_shape = tensor.shape
    
    # If shapes match, return original tensor
    if current_shape == target_shape:
        return tensor
    
    # Try reshaping if total elements match
    if tensor.numel() == np.prod(target_shape):
        return tensor.reshape(target_shape)
    
    # If reshaping not possible, try padding
    result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    
    # Determine common dimensions to copy
    copy_shape = [min(s1, s2) for s1, s2 in zip(current_shape, target_shape)]
    
    # Create slices for copying
    slices_src = tuple(slice(0, s) for s in copy_shape)
    slices_dst = tuple(slice(0, s) for s in copy_shape)
    
    # Copy data
    result[slices_dst] = tensor[slices_src]
    
    return result
