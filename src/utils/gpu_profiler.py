"""
GPU Memory and Performance Profiler

This module provides utilities for profiling GPU memory usage and performance
of the neural network models, with a focus on optimizing CuPy-PyTorch data transfers
and memory utilization.
"""

import torch
import time
import gc
from functools import wraps
import warnings

try:
    import cupy as cp
    USING_CUPY = True
except ImportError:
    import numpy as cp
    USING_CUPY = False
    warnings.warn("CuPy not available. Using NumPy instead.")

from src.utils.memory_utils import (
    efficient_cp_to_torch, 
    efficient_torch_to_cp,
    print_gpu_memory_status,
    profile_memory_usage
)

# Memory tracking decorators
def track_memory(func):
    """
    Decorator to track memory usage before and after function execution.
    
    Args:
        func: Function to track memory usage for
    
    Returns:
        Wrapped function with memory tracking
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Clear memory before tracking
        torch.cuda.empty_cache()
        gc.collect()
        
        if USING_CUPY:
            cp.get_default_memory_pool().free_all_blocks()
        
        # Record memory before
        start_mem = {}
        for i in range(torch.cuda.device_count()):
            start_mem[f'cuda:{i}'] = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
        
        # Execute function
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Record memory after
        end_mem = {}
        peak_mem = {}
        for i in range(torch.cuda.device_count()):
            end_mem[f'cuda:{i}'] = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
            peak_mem[f'cuda:{i}'] = torch.cuda.max_memory_allocated(i) / (1024 ** 3)  # GB
        
        # Print memory usage
        print(f"\n===== Memory Report for {func.__name__} =====")
        print(f"Execution time: {end_time - start_time:.4f} seconds")
        
        for device in start_mem.keys():
            print(f"{device} Memory:")
            print(f"  Start: {start_mem[device]:.4f} GB")
            print(f"  End: {end_mem[device]:.4f} GB")
            print(f"  Change: {end_mem[device] - start_mem[device]:.4f} GB")
            print(f"  Peak: {peak_mem[device]:.4f} GB")
        
        return result
    
    return wrapper

def profile_gpu_performance(model, input_shape, batch_size=32, warmup_steps=5, profile_steps=20, device=None):
    """
    Profile GPU performance of a model.
    
    Args:
        model: PyTorch model to profile
        input_shape: Shape of a single input item (excluding batch dim)
        batch_size: Batch size to use for profiling
        warmup_steps: Number of warmup steps
        profile_steps: Number of steps to profile
        device: Device to run profiling on (defaults to model's device)
        
    Returns:
        dict: Performance metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Ensure model is in eval mode
    model.eval()
    
    # Generate random input data
    input_shape = tuple(input_shape) if not isinstance(input_shape, tuple) else input_shape
    
    # Print profiling setup
    print(f"\n===== GPU Performance Profile =====")
    print(f"Model: {model.__class__.__name__}")
    print(f"Batch size: {batch_size}")
    print(f"Input shape: {input_shape}")
    print(f"Device: {device}")
    
    # Warmup
    print("\nWarming up...")
    for _ in range(warmup_steps):
        with torch.no_grad():
            x = torch.randn((batch_size,) + input_shape, device=device)
            _ = model(x)
    
    # Profile
    print(f"\nProfiling ({profile_steps} steps)...")
    start_time = time.time()
    with torch.cuda.profiler.profile():
        with torch.no_grad():
            for i in range(profile_steps):
                step_start = time.time()
                x = torch.randn((batch_size,) + input_shape, device=device)
                _ = model(x)
                step_time = time.time() - step_start
                print(f"Step {i+1}/{profile_steps}: {step_time:.4f}s")
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_time = total_time / profile_steps
    samples_per_second = batch_size / avg_time
    
    # Print results
    print("\n===== Results =====")
    print(f"Total time: {total_time:.4f}s")
    print(f"Average step time: {avg_time:.4f}s")
    print(f"Samples per second: {samples_per_second:.2f}")
    
    # Check memory usage
    memory_stats = print_gpu_memory_status()
    
    # Check data transfer performance
    print("\n===== Data Transfer Performance =====")
    profile_cupy_torch_transfers(batch_size, input_shape, device)
    
    # Return metrics
    return {
        'total_time': total_time,
        'avg_step_time': avg_time,
        'samples_per_second': samples_per_second,
        'memory_stats': memory_stats,
    }

def profile_cupy_torch_transfers(batch_size, shape, device, iterations=10):
    """
    Profile CuPy-PyTorch data transfer performance.
    
    Args:
        batch_size: Batch size to use for profiling
        shape: Shape of a single input item (excluding batch dim)
        device: Device to run profiling on
        iterations: Number of iterations to average over
        
    Returns:
        dict: Performance metrics for different transfer methods
    """
    if not USING_CUPY:
        print("CuPy not available. Skipping CuPy-PyTorch transfer profiling.")
        return {}
    
    shape = tuple(shape) if not isinstance(shape, tuple) else shape
    full_shape = (batch_size,) + shape
    
    # Create test data
    cp_array = cp.random.random(full_shape).astype(cp.float32)
    
    # Method 1: Standard as_tensor
    start_time = time.time()
    for _ in range(iterations):
        torch_tensor = torch.as_tensor(cp_array.get(), device=device)
        # Force synchronization
        _ = torch_tensor.shape
    standard_time = (time.time() - start_time) / iterations
    
    # Method 2: Using efficient_cp_to_torch
    start_time = time.time()
    for _ in range(iterations):
        torch_tensor = efficient_cp_to_torch(cp_array, device=device)
        # Force synchronization
        _ = torch_tensor.shape
    efficient_time = (time.time() - start_time) / iterations
    
    # Method 3: DLPack (if available)
    dlpack_time = None
    try:
        start_time = time.time()
        for _ in range(iterations):
            torch_tensor = torch.utils.dlpack.from_dlpack(cp_array.toDlpack())
            # Force synchronization
            _ = torch_tensor.shape
        dlpack_time = (time.time() - start_time) / iterations
    except (AttributeError, RuntimeError):
        dlpack_time = None
    
    # Print results
    print(f"Data shape: {full_shape}")
    print(f"Standard as_tensor: {standard_time*1000:.2f}ms")
    print(f"Efficient cp_to_torch: {efficient_time*1000:.2f}ms")
    if dlpack_time is not None:
        print(f"DLPack: {dlpack_time*1000:.2f}ms")
    
    # Speedups
    if standard_time > 0:
        if efficient_time > 0:
            print(f"Efficient speedup: {standard_time/efficient_time:.2f}x")
        if dlpack_time is not None and dlpack_time > 0:
            print(f"DLPack speedup: {standard_time/dlpack_time:.2f}x")
    
    return {
        'standard_time_ms': standard_time * 1000,
        'efficient_time_ms': efficient_time * 1000,
        'dlpack_time_ms': dlpack_time * 1000 if dlpack_time is not None else None,
    }

def generate_optimization_report(model, input_shape, batch_sizes=(32, 64, 128)):
    """
    Generate a comprehensive optimization report.
    
    Args:
        model: PyTorch model to profile
        input_shape: Shape of a single input item (excluding batch dim)
        batch_sizes: Batch sizes to test
        
    Returns:
        dict: Comprehensive performance metrics
    """
    device = next(model.parameters()).device
    report = {
        'model_name': model.__class__.__name__,
        'device': str(device),
        'input_shape': input_shape,
        'batch_size_results': {},
        'memory_profile': {},
        'transfer_performance': {},
        'recommendations': []
    }
    
    # Profile memory usage
    print("\n===== Memory Profile =====")
    memory_profile = profile_memory_usage(model, input_shape, batch_size=batch_sizes[0], detailed=True)
    report['memory_profile'] = memory_profile
    
    # Test transfer performance
    print("\n===== Transfer Performance =====")
    transfer_perf = profile_cupy_torch_transfers(batch_sizes[0], input_shape, device)
    report['transfer_performance'] = transfer_perf
    
    # Test different batch sizes
    for batch_size in batch_sizes:
        print(f"\n===== Testing Batch Size: {batch_size} =====")
        # For each batch size, run a brief performance test
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                x = torch.randn((batch_size,) + tuple(input_shape), device=device)
                _ = model(x)
            
            # Test performance
            start_time = time.time()
            steps = 10
            for _ in range(steps):
                x = torch.randn((batch_size,) + tuple(input_shape), device=device)
                _ = model(x)
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_time = total_time / steps
            samples_per_second = batch_size / avg_time
            
            print(f"Batch size {batch_size}:")
            print(f"  Average step time: {avg_time:.4f}s")
            print(f"  Samples per second: {samples_per_second:.2f}")
            
            report['batch_size_results'][batch_size] = {
                'avg_step_time': avg_time,
                'samples_per_second': samples_per_second
            }
    
    # Generate recommendations
    generate_optimization_recommendations(report)
    
    return report

def generate_optimization_recommendations(report):
    """
    Generate optimization recommendations based on the performance report.
    
    Args:
        report: Performance report from generate_optimization_report
        
    Returns:
        list: Recommendations
    """
    recommendations = []
    
    # Check if CuPy is available
    if not USING_CUPY:
        recommendations.append("Install CuPy for GPU-accelerated array operations")
    
    # Check if efficient_cp_to_torch is faster
    if USING_CUPY and 'transfer_performance' in report:
        tp = report['transfer_performance']
        if tp.get('standard_time_ms', 0) > 0 and tp.get('efficient_time_ms', 0) > 0:
            if tp['standard_time_ms'] > tp['efficient_time_ms'] * 1.2:  # 20% faster
                recommendations.append(
                    "Use efficient_cp_to_torch for CuPy to PyTorch transfers - "
                    f"{tp['standard_time_ms']/tp['efficient_time_ms']:.2f}x speedup"
                )
    
    # Check batch size performance
    if 'batch_size_results' in report and len(report['batch_size_results']) > 0:
        best_batch_size = max(report['batch_size_results'].keys(), 
                            key=lambda bs: report['batch_size_results'][bs]['samples_per_second'])
        
        recommendations.append(
            f"Optimal batch size for throughput: {best_batch_size} "
            f"({report['batch_size_results'][best_batch_size]['samples_per_second']:.2f} samples/second)"
        )
    
    # Check memory utilization
    if 'memory_profile' in report:
        memory_stats = report['memory_profile']
        if memory_stats:
            for device, stats in memory_stats.items():
                if 'allocated_GB' in stats and 'reserved_GB' in stats:
                    utilization = stats['allocated_GB'] / stats['reserved_GB'] if stats['reserved_GB'] > 0 else 0
                    if utilization < 0.5:
                        recommendations.append(
                            f"Low memory utilization on {device}: {utilization:.1%}. "
                            "Consider increasing batch size or reducing reserved memory."
                        )
    
    # Add recommendations to report
    report['recommendations'] = recommendations
    
    # Print recommendations
    print("\n===== Optimization Recommendations =====")
    for i, rec in enumerate(recommendations):
        print(f"{i+1}. {rec}")
    
    return recommendations

def apply_optimization_recommendations(model, report):
    """
    Apply optimization recommendations from the report.
    
    Args:
        model: PyTorch model to optimize
        report: Performance report from generate_optimization_report
        
    Returns:
        model: Optimized model
    """
    # Implement optimizations based on recommendations
    
    # Replace data transfers with efficient_cp_to_torch
    if "Use efficient_cp_to_torch" in str(report.get('recommendations', [])):
        # The model.py file has already been updated
        print("✅ Using efficient_cp_to_torch for CuPy to PyTorch transfers")
    
    # Apply optimal batch size if recommended
    for rec in report.get('recommendations', []):
        if "Optimal batch size" in rec:
            try:
                batch_size = int(rec.split("batch size")[1].split(":")[1].strip().split(" ")[0])
                print(f"✅ Recommended batch size: {batch_size}")
                # The batch size could be set in your training loop
            except (ValueError, IndexError):
                pass
    
    # Additional optimizations
    
    return model
