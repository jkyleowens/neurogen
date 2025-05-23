"""
Memory Utilities

This module provides utility functions for memory management
and optimization in the Brain-Inspired Neural Network.
"""

import torch
import gc
try:
    import cupy as cp
    USING_CUPY = True
    print("Using CuPy for memory operations")
    # Empty cupy cache as well
    cp.get_default_memory_pool().free_all_blocks()
except ImportError:
    import numpy as cp
    USING_CUPY = False

# Add efficient conversion functions
def efficient_cp_to_torch(cp_array, device='cuda', non_blocking=True):
    """
    Efficiently convert CuPy array to PyTorch tensor without redundant CPU transfers.
    
    Args:
        cp_array: CuPy array to convert
        device: Target PyTorch device (default: 'cuda')
        non_blocking: Whether to perform asynchronous transfer (default: True)
        
    Returns:
        torch.Tensor: PyTorch tensor on the specified device
    """
    if not USING_CUPY:
        # For NumPy arrays, use standard conversion
        return torch.from_numpy(cp_array).to(device=device, non_blocking=non_blocking)
    
    # For CuPy arrays, use the most efficient path based on target device
    if device == 'cuda' or (isinstance(device, torch.device) and device.type == 'cuda'):
        # Direct GPU-to-GPU transfer (zero copy when possible)
        # Use CUDA IPC or DLPack for zero-copy when array is contiguous
        if cp_array.flags.c_contiguous:
            try:
                # Use DLPack protocol for zero-copy GPU-to-GPU transfer
                return torch.utils.dlpack.from_dlpack(cp_array.toDlpack())
            except (AttributeError, RuntimeError):
                # Fallback to standard path if DLPack not available
                return torch.as_tensor(cp_array.get(), device=device, non_blocking=non_blocking)
        else:
            # Make contiguous first if needed
            return torch.as_tensor(cp.ascontiguousarray(cp_array).get(), 
                                  device=device, non_blocking=non_blocking)
    else:
        # For CPU tensor, we need to get the array to host memory first
        return torch.from_numpy(cp.asnumpy(cp_array)).to(device=device, non_blocking=non_blocking)

def efficient_torch_to_cp(torch_tensor, stream=None):
    """
    Efficiently convert PyTorch tensor to CuPy array without redundant transfers.
    
    Args:
        torch_tensor: PyTorch tensor to convert
        stream: CUDA stream to use for the transfer (default: None)
        
    Returns:
        cp_array: CuPy array
    """
    if not USING_CUPY:
        # For NumPy, detach and convert to NumPy
        return cp.array(torch_tensor.detach().cpu().numpy())
    
    # Ensure tensor is on CPU or CUDA for proper conversion
    if torch_tensor.device.type == 'cuda':
        # Direct GPU-to-GPU transfer (zero copy when possible)
        try:
            # Try to use DLPack protocol for zero-copy transfer
            if torch_tensor.is_contiguous():
                return cp.fromDlpack(torch.utils.dlpack.to_dlpack(torch_tensor))
            else:
                # Make contiguous first
                return cp.fromDlpack(torch.utils.dlpack.to_dlpack(torch_tensor.contiguous()))
        except (AttributeError, RuntimeError):
            # Fallback to standard path if DLPack not available
            return cp.asarray(torch_tensor.detach().cpu().numpy())
    else:
        # CPU tensor to CuPy array
        return cp.asarray(torch_tensor.detach().numpy())

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
        if USING_CUPY:
            cp.get_default_memory_pool().free_all_blocks()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        if USING_CUPY:
            cp.get_default_memory_pool().free_all_blocks()
    
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
    
    # Add CuPy memory stats if available
    if USING_CUPY:
        try:
            mem_pool = cp.get_default_memory_pool()
            used_bytes = mem_pool.used_bytes()
            total_bytes = mem_pool.total_bytes()
            used_gb = used_bytes / (1024 ** 3)
            total_gb = total_bytes / (1024 ** 3)
            print(f"CuPy Memory: {used_gb:.2f}GB used, {total_gb:.2f}GB total")
            stats['cupy'] = {
                'used_GB': used_gb,
                'total_GB': total_gb
            }
        except Exception as e:
            print(f"Error getting CuPy memory stats: {e}")
    
    return stats

def tensor_shape_fix(tensor, target_shape):
    """
    Optimized function to fix tensor shape mismatches by reshaping or padding.
    
    Args:
        tensor (torch.Tensor): Input tensor
        target_shape (tuple): Target shape
        
    Returns:
        torch.Tensor: Reshaped or padded tensor
    """
    current_shape = tensor.shape
    
    # Fast path: If shapes match, return original tensor
    if current_shape == target_shape:
        return tensor
    
    # Calculate tensor size once
    tensor_numel = tensor.numel()
    target_numel = torch.prod(torch.tensor(target_shape, device=tensor.device)).item()
    
    # Fast path: Try reshape if element counts match
    if tensor_numel == target_numel:
        # Try view for contiguous tensors first (zero-copy operation, much faster)
        if tensor.is_contiguous():
            try:
                return tensor.view(target_shape)
            except RuntimeError:
                # View failed, try reshape
                return tensor.reshape(target_shape)
        else:
            # For non-contiguous tensors, make contiguous first for better performance
            return tensor.contiguous().view(target_shape)
    
    # Handle dimension mismatch with minimal memory operations
    # Create result tensor with pinned memory if on CUDA for faster transfers
    pin_memory = tensor.is_cuda
    
    # Reuse existing allocation if possible to reduce memory pressure
    result = torch.zeros(target_shape, 
                        dtype=tensor.dtype, 
                        device=tensor.device,
                        pin_memory=pin_memory)
    
    # For efficiency, handle common dimension patterns
    if len(current_shape) == len(target_shape):
        # Batch dimension handling (efficient for batched tensors)
        # Determine common dimensions to copy
        copy_shape = tuple(min(s1, s2) for s1, s2 in zip(current_shape, target_shape))
        
        # Use native slice indexing for optimal performance
        slices = tuple(slice(0, s) for s in copy_shape)
        result[slices] = tensor[slices]
    else:
        # For tensors with different dimensions, use adaptive copying
        # Flatten both tensors, copy common elements, then reshape
        flat_size = min(tensor_numel, target_numel)
        result.view(-1)[:flat_size] = tensor.view(-1)[:flat_size]
    
    return result

def optimize_batch_size(model, input_shape, target_batch_size=128, min_batch_size=8, 
                   max_memory_usage=0.8, device=None):
    """
    Dynamically optimize batch size based on available GPU memory.
    
    Args:
        model: PyTorch model to test
        input_shape: Shape of a single input item (excluding batch dim)
        target_batch_size: Desired batch size to try
        min_batch_size: Minimum acceptable batch size
        max_memory_usage: Maximum fraction of GPU memory to use (0-1)
        device: Device to test on
        
    Returns:
        int: Recommended batch size
    """
    if not torch.cuda.is_available():
        print("CUDA not available, using default batch size")
        return target_batch_size
    
    if device is None:
        device = torch.device('cuda')
    
    # Free all unused memory first
    torch.cuda.empty_cache()
    if USING_CUPY:
        cp.get_default_memory_pool().free_all_blocks()
    
    # Get total GPU memory
    total_memory = torch.cuda.get_device_properties(device).total_memory
    
    # Binary search for optimal batch size
    low, high = min_batch_size, target_batch_size
    optimal_batch_size = min_batch_size
    
    while low <= high:
        mid = (low + high) // 2
        try:
            # Try creating a batch of this size
            x = torch.randn((mid,) + tuple(input_shape), device=device)
            
            # Run a forward pass to check memory usage
            with torch.no_grad():
                _ = model(x)
            
            # Check memory usage after forward pass
            allocated_memory = torch.cuda.memory_allocated(device)
            memory_usage_fraction = allocated_memory / total_memory
            
            print(f"Batch size {mid}: Memory usage {memory_usage_fraction:.2%}")
            
            if memory_usage_fraction <= max_memory_usage:
                # This batch size works, try a larger one
                optimal_batch_size = mid
                low = mid + 1
            else:
                # Too much memory used, try smaller batch size
                high = mid - 1
            
            # Clean up
            del x
            torch.cuda.empty_cache()
            if USING_CUPY:
                cp.get_default_memory_pool().free_all_blocks()
                
        except RuntimeError:
            # Out of memory, try smaller batch size
            print(f"Batch size {mid} caused OOM, trying smaller")
            high = mid - 1
            torch.cuda.empty_cache()
            if USING_CUPY:
                cp.get_default_memory_pool().free_all_blocks()
    
    # Add a small safety margin
    recommended_batch_size = max(min_batch_size, int(optimal_batch_size * 0.95))
    print(f"Recommended batch size: {recommended_batch_size}")
    return recommended_batch_size

def profile_memory_usage(model=None, input_shape=None, batch_size=32, detailed=False):
    """
    Profile GPU memory usage with detailed breakdown of allocations.
    
    Args:
        model: Optional PyTorch model to profile
        input_shape: Shape for test input if model provided
        batch_size: Batch size for test if model provided
        detailed: Whether to show detailed per-tensor breakdown
        
    Returns:
        dict: Memory usage statistics
    """
    if not torch.cuda.is_available():
        print("CUDA not available")
        return {}
    
    print("\n===== MEMORY USAGE PROFILE =====")
    
    # Get basic memory stats
    stats = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # GB
        max_allocated = torch.cuda.max_memory_allocated(i) / (1024 ** 3)  # GB
        
        stats[f'cuda:{i}'] = {
            'allocated_GB': allocated,
            'reserved_GB': reserved,
            'max_allocated_GB': max_allocated,
            'utilization': allocated / reserved if reserved > 0 else 0
        }
        
        print(f"CUDA:{i} Memory: {allocated:.2f}GB allocated, "
              f"{reserved:.2f}GB reserved, {max_allocated:.2f}GB max allocated "
              f"({stats[f'cuda:{i}']['utilization']:.1%} utilization)")
    
    # Add CuPy memory stats if available
    if USING_CUPY:
        try:
            mem_pool = cp.get_default_memory_pool()
            used_bytes = mem_pool.used_bytes()
            total_bytes = mem_pool.total_bytes()
            used_gb = used_bytes / (1024 ** 3)
            total_gb = total_bytes / (1024 ** 3)
            print(f"CuPy Memory: {used_gb:.2f}GB used, {total_gb:.2f}GB total")
            stats['cupy'] = {
                'used_GB': used_gb,
                'total_GB': total_gb
            }
        except Exception as e:
            print(f"Error getting CuPy memory stats: {e}")
    
    # Add advanced memory profiling
    if detailed:
        try:
            import torch.autograd.profiler as profiler
            
            # Get top tensors by size
            if hasattr(torch.cuda, 'memory_snapshot'):
                snapshot = torch.cuda.memory_snapshot()
                segments = sorted(snapshot, key=lambda x: x.get('size', 0), reverse=True)
                print("\nTop 10 memory allocations:")
                for i, segment in enumerate(segments[:10]):
                    size_mb = segment.get('size', 0) / (1024 * 1024)
                    state = segment.get('state', 'unknown')
                    print(f"{i+1}. {size_mb:.2f}MB - {state}")
            
            # Optional: profile model if provided
            if model is not None and input_shape is not None:
                device = next(model.parameters()).device
                x = torch.randn((batch_size,) + tuple(input_shape), device=device)
                
                print("\nProfiling model execution...")
                with profiler.profile(use_cuda=True) as prof:
                    with torch.no_grad():
                        _ = model(x)
                
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                stats['model_profile'] = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
                
                del x
                torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"Error in detailed profiling: {e}")
    
    print("==============================\n")
    return stats

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
        if USING_CUPY:
            cp.get_default_memory_pool().free_all_blocks()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        if USING_CUPY:
            cp.get_default_memory_pool().free_all_blocks()
    
    # Run garbage collection
    gc.collect()
    
    return model
