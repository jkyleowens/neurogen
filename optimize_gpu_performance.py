#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPU Performance Optimization for Neural Networks

This script performs a comprehensive performance audit and optimization
of GPU acceleration for neural network training, focusing on:
1. Memory optimization
2. CuPy-PyTorch data transfer efficiency
3. Batch normalization and RNN performance
4. Batch size optimization
5. Kernel launch overhead reduction

Usage:
    python optimize_gpu_performance.py --model-path models/checkpoints/best_model.pt
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import warnings
import json

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import cupy as cp
    USING_CUPY = True
except ImportError:
    import numpy as cp
    USING_CUPY = False
    warnings.warn("CuPy not available. Performance optimization will be limited.")

# Import project modules
from src.model import BrainInspiredNN
from src.utils.memory_utils import (
    optimize_memory_usage, print_gpu_memory_status, 
    profile_memory_usage, optimize_batch_size,
    efficient_cp_to_torch, efficient_torch_to_cp
)

def parse_args():
    parser = argparse.ArgumentParser(description="Neural Network GPU Performance Optimization")
    parser.add_argument("--model-path", type=str, default="neurogen/models/checkpoints/best_model.pt",
                      help="Path to model checkpoint to optimize")
    parser.add_argument("--config-path", type=str, default="config/config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--batch-size", type=int, default=32,
                      help="Starting batch size for optimization")
    parser.add_argument("--input-size", type=int, default=None,
                      help="Input size (will be inferred from model if not provided)")
    parser.add_argument("--output-dir", type=str, default="results/performance_optimization",
                      help="Directory to save optimization results")
    parser.add_argument("--detailed", action="store_true",
                      help="Run detailed performance analysis (slower)")
    parser.add_argument("--test-iterations", type=int, default=100,
                      help="Number of iterations for performance testing")
    return parser.parse_args()

def load_model(model_path, device):
    """Load model from checkpoint."""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            # Try to extract config if available
            config = checkpoint.get('config', {})
            
            # Create model using config extracted from checkpoint
            model = BrainInspiredNN(config).to(device)
            model.load_state_dict(model_state)
            
            print(f"Model loaded from {model_path}")
            return model, config
        elif 'state_dict' in checkpoint:
            model_state = checkpoint['state_dict']
            config = checkpoint.get('config', {})
            
            model = BrainInspiredNN(config).to(device)
            model.load_state_dict(model_state)
            
            print(f"Model loaded from {model_path}")
            return model, config
        elif isinstance(checkpoint, dict) and 'config' in checkpoint:
            # Looks like the whole model is saved directly
            model = checkpoint
            config = checkpoint.get('config', {})
            
            print(f"Full model loaded from {model_path}")
            return model, config
        else:
            # Try to directly load as a model
            model = checkpoint
            # Extract a default config based on model attributes
            config = {
                'model': {
                    'input_size': getattr(model, 'input_size', None),
                    'hidden_size': getattr(model, 'hidden_size', None),
                    'output_size': getattr(model, 'output_size', None)
                }
            }
            print(f"Model loaded, no config found in checkpoint")
            return model, config
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, {}

def test_cupy_torch_transfer(iterations=1000, tensor_size=(1000, 1000)):
    """Test and optimize CuPy-PyTorch data transfer."""
    if not USING_CUPY:
        print("CuPy not available, skipping transfer tests")
        return {}
    
    results = {}
    device = torch.device('cuda')
    
    # Create test data
    torch_tensor = torch.rand(tensor_size, device=device)
    cp_array = cp.random.rand(*tensor_size)
    
    # Test 1: Standard CuPy to PyTorch
    start_time = time.time()
    for _ in range(iterations):
        standard_tensor = torch.as_tensor(cp_array.get(), device=device)
    standard_time = time.time() - start_time
    del standard_tensor
    torch.cuda.empty_cache()
    
    # Test 2: Efficient CuPy to PyTorch
    start_time = time.time()
    for _ in range(iterations):
        efficient_tensor = efficient_cp_to_torch(cp_array, device=device)
    efficient_time = time.time() - start_time
    del efficient_tensor
    torch.cuda.empty_cache()
    
    # Test 3: Standard PyTorch to CuPy
    start_time = time.time()
    for _ in range(iterations):
        standard_array = cp.array(torch_tensor.cpu().numpy())
    standard_reverse_time = time.time() - start_time
    del standard_array
    
    # Test 4: Efficient PyTorch to CuPy
    start_time = time.time()
    for _ in range(iterations):
        efficient_array = efficient_torch_to_cp(torch_tensor)
    efficient_reverse_time = time.time() - start_time
    del efficient_array
    
    # Record results
    results['cupy_to_torch'] = {
        'standard_time': standard_time,
        'efficient_time': efficient_time,
        'speedup': standard_time / efficient_time if efficient_time > 0 else float('inf')
    }
    
    results['torch_to_cupy'] = {
        'standard_time': standard_reverse_time,
        'efficient_time': efficient_reverse_time,
        'speedup': standard_reverse_time / efficient_reverse_time if efficient_reverse_time > 0 else float('inf')
    }
    
    print("\n===== CuPy-PyTorch Transfer Performance =====")
    print(f"CuPy to PyTorch (standard): {standard_time:.4f}s")
    print(f"CuPy to PyTorch (efficient): {efficient_time:.4f}s")
    print(f"Speedup: {results['cupy_to_torch']['speedup']:.2f}x")
    
    print(f"\nPyTorch to CuPy (standard): {standard_reverse_time:.4f}s")
    print(f"PyTorch to CuPy (efficient): {efficient_reverse_time:.4f}s")
    print(f"Speedup: {results['torch_to_cupy']['speedup']:.2f}x")
    
    return results

def optimize_model(model, batch_size=32, input_size=None, iterations=100):
    """Perform multi-step model optimization."""
    if model is None:
        return {}
    
    # Get device and basic model information
    device = next(model.parameters()).device
    model_input_size = getattr(model, 'input_size', input_size)
    
    if model_input_size is None:
        print("Input size not specified and couldn't be inferred from model")
        return {}
    
    results = {}
    
    # Step 1: Memory optimization
    print("\n===== Memory Optimization =====")
    model = optimize_memory_usage(model, device)
    
    # Step 2: Batch size optimization
    print("\n===== Batch Size Optimization =====")
    optimal_batch_size = optimize_batch_size(
        model=model,
        input_shape=(model_input_size,),
        target_batch_size=batch_size * 2,
        min_batch_size=batch_size // 2
    )
    results['optimal_batch_size'] = optimal_batch_size
    
    # Step 3: Performance benchmarking
    print("\n===== Performance Benchmarking =====")
    test_batch_sizes = [
        max(1, optimal_batch_size // 2),
        optimal_batch_size,
        min(optimal_batch_size * 2, 1024)  # Cap at 1024 to avoid OOM
    ]
    
    perf_results = {}
    model.eval()  # Ensure model is in eval mode for consistent timing
    
    for bs in test_batch_sizes:
        print(f"\nTesting batch size: {bs}")
        # Warmup
        x = torch.randn((bs, model_input_size), device=device)
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        
        # Timed test
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                x = torch.randn((bs, model_input_size), device=device)
                _ = model(x)
                torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        # Calculate performance metrics
        total_samples = bs * iterations
        samples_per_sec = total_samples / elapsed
        ms_per_batch = (elapsed / iterations) * 1000
        
        perf_results[bs] = {
            'batch_size': bs,
            'total_time_seconds': elapsed,
            'samples_per_second': samples_per_sec,
            'ms_per_batch': ms_per_batch
        }
        
        print(f"  Time: {elapsed:.2f}s total, {ms_per_batch:.2f}ms/batch")
        print(f"  Performance: {samples_per_sec:.1f} samples/second")
        
        # Clean up
        del x
        torch.cuda.empty_cache()
    
    results['performance_benchmarks'] = perf_results
    
    # Step 4: Find optimal throughput batch size
    best_batch_size = max(perf_results.keys(), key=lambda bs: perf_results[bs]['samples_per_second'])
    results['recommended_batch_size'] = best_batch_size
    print(f"\nRecommended batch size for maximum throughput: {best_batch_size}")
    print(f"Performance: {perf_results[best_batch_size]['samples_per_second']:.1f} samples/second")
    
    return results

def profile_gradients(model, batch_size=32, input_size=None, iterations=10):
    """Profile gradient computation performance."""
    if model is None:
        return {}
    
    # Get device and basic model information
    device = next(model.parameters()).device
    model_input_size = getattr(model, 'input_size', input_size)
    model_output_size = getattr(model, 'output_size', 1)
    
    if model_input_size is None:
        print("Input size not specified and couldn't be inferred from model")
        return {}
    
    print("\n===== Gradient Computation Profiling =====")
    
    # Create loss function similar to what's used in training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Put model in training mode
    model.train()
    
    # Warmup
    x = torch.randn((batch_size, model_input_size), device=device)
    target = torch.randn((batch_size, model_output_size), device=device)
    output = model(x)
    # Handle possible shape differences
    if output.shape != target.shape:
        if output.dim() == 3 and target.dim() == 2:
            output = output[:, -1, :]
        min_features = min(output.shape[-1], target.shape[-1])
        output = output[..., :min_features]
        target = target[..., :min_features]
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    
    # Timing array
    times = []
    
    for _ in range(iterations):
        x = torch.randn((batch_size, model_input_size), device=device)
        target = torch.randn((batch_size, model_output_size), device=device)
        
        # Forward pass
        start_time = time.time()
        output = model(x)
        if output.shape != target.shape:
            if output.dim() == 3 and target.dim() == 2:
                output = output[:, -1, :]
            min_features = min(output.shape[-1], target.shape[-1])
            output = output[..., :min_features]
            target = target[..., :min_features]
        forward_time = time.time() - start_time
        
        # Loss calculation
        start_time = time.time()
        loss = criterion(output, target)
        loss_time = time.time() - start_time
        
        # Backward pass
        start_time = time.time()
        optimizer.zero_grad()
        loss.backward()
        backward_time = time.time() - start_time
        
        # Optimizer step
        start_time = time.time()
        optimizer.step()
        optim_time = time.time() - start_time
        
        # Total iteration time
        total_time = forward_time + loss_time + backward_time + optim_time
        
        times.append({
            'forward_time': forward_time,
            'loss_time': loss_time,
            'backward_time': backward_time,
            'optim_time': optim_time,
            'total_time': total_time
        })
        
        torch.cuda.synchronize()
    
    # Calculate averages
    avg_forward = sum(t['forward_time'] for t in times) / len(times)
    avg_loss = sum(t['loss_time'] for t in times) / len(times)
    avg_backward = sum(t['backward_time'] for t in times) / len(times)
    avg_optim = sum(t['optim_time'] for t in times) / len(times)
    avg_total = sum(t['total_time'] for t in times) / len(times)
    
    print(f"Average times for batch size {batch_size}:")
    print(f"  Forward pass: {avg_forward*1000:.2f}ms ({avg_forward/avg_total*100:.1f}%)")
    print(f"  Loss calculation: {avg_loss*1000:.2f}ms ({avg_loss/avg_total*100:.1f}%)")
    print(f"  Backward pass: {avg_backward*1000:.2f}ms ({avg_backward/avg_total*100:.1f}%)")
    print(f"  Optimizer step: {avg_optim*1000:.2f}ms ({avg_optim/avg_total*100:.1f}%)")
    print(f"  Total iteration: {avg_total*1000:.2f}ms")
    
    result = {
        'batch_size': batch_size,
        'forward_ms': avg_forward * 1000,
        'loss_ms': avg_loss * 1000,
        'backward_ms': avg_backward * 1000,
        'optim_ms': avg_optim * 1000,
        'total_ms': avg_total * 1000,
        'samples_per_second': batch_size / avg_total
    }
    
    return result

def generate_performance_report(results, output_dir="results/performance_optimization"):
    """Generate comprehensive performance report."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw results as JSON
    with open(os.path.join(output_dir, "performance_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Create markdown report
    report_path = os.path.join(output_dir, "performance_report.md")
    with open(report_path, "w") as f:
        f.write("# Neural Network GPU Performance Optimization Report\n\n")
        
        # System information
        f.write("## System Information\n\n")
        f.write("| Component | Details |\n")
        f.write("|:----------|:--------|\n")
        
        if torch.cuda.is_available():
            f.write(f"| GPU | {torch.cuda.get_device_name(0)} |\n")
            f.write(f"| CUDA Version | {torch.version.cuda} |\n")
        else:
            f.write("| GPU | Not available |\n")
        
        f.write(f"| PyTorch Version | {torch.__version__} |\n")
        f.write(f"| CuPy Available | {USING_CUPY} |\n\n")
        
        # Memory optimization
        if 'memory_profile' in results:
            f.write("## Memory Usage\n\n")
            for device, stats in results['memory_profile'].items():
                if "cuda" in device:
                    f.write(f"### {device.upper()}\n\n")
                    f.write(f"- Allocated: {stats['allocated_GB']:.2f} GB\n")
                    f.write(f"- Reserved: {stats['reserved_GB']:.2f} GB\n")
                    f.write(f"- Max Allocated: {stats['max_allocated_GB']:.2f} GB\n")
                    f.write(f"- Utilization: {stats['utilization']*100:.1f}%\n\n")
        
        # Batch size optimization
        if 'optimal_batch_size' in results:
            f.write("## Batch Size Optimization\n\n")
            f.write(f"- Optimal memory-based batch size: {results['optimal_batch_size']}\n")
            if 'recommended_batch_size' in results:
                f.write(f"- Recommended performance-based batch size: {results['recommended_batch_size']}\n\n")
        
        # Performance benchmarks
        if 'performance_benchmarks' in results:
            f.write("## Performance Benchmarks\n\n")
            f.write("| Batch Size | Samples/Second | ms/batch |\n")
            f.write("|:----------:|:-------------:|:--------:|\n")
            
            for bs, data in results['performance_benchmarks'].items():
                f.write(f"| {bs} | {data['samples_per_second']:.1f} | {data['ms_per_batch']:.2f} |\n")
            f.write("\n")
        
        # CuPy-PyTorch transfer optimization
        if 'cupy_torch_transfer' in results:
            f.write("## CuPy-PyTorch Transfer Optimization\n\n")
            
            cp_to_torch = results['cupy_torch_transfer']['cupy_to_torch']
            torch_to_cp = results['cupy_torch_transfer']['torch_to_cupy']
            
            f.write("### CuPy to PyTorch\n\n")
            f.write(f"- Standard method: {cp_to_torch['standard_time']:.4f}s\n")
            f.write(f"- Optimized method: {cp_to_torch['efficient_time']:.4f}s\n")
            f.write(f"- Speedup: {cp_to_torch['speedup']:.2f}x\n\n")
            
            f.write("### PyTorch to CuPy\n\n")
            f.write(f"- Standard method: {torch_to_cp['standard_time']:.4f}s\n")
            f.write(f"- Optimized method: {torch_to_cp['efficient_time']:.4f}s\n")
            f.write(f"- Speedup: {torch_to_cp['speedup']:.2f}x\n\n")
        
        # Training profile
        if 'gradient_profile' in results:
            f.write("## Training Performance Profile\n\n")
            profile = results['gradient_profile']
            
            f.write(f"- Batch size: {profile['batch_size']}\n")
            f.write(f"- Forward pass: {profile['forward_ms']:.2f}ms\n")
            f.write(f"- Loss calculation: {profile['loss_ms']:.2f}ms\n")
            f.write(f"- Backward pass: {profile['backward_ms']:.2f}ms\n")
            f.write(f"- Optimizer step: {profile['optim_ms']:.2f}ms\n")
            f.write(f"- Total iteration time: {profile['total_ms']:.2f}ms\n")
            f.write(f"- Throughput: {profile['samples_per_second']:.1f} samples/second\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        # Always recommend the efficient transfer functions
        f.write("1. **Use efficient CuPy-PyTorch transfer functions** - Implement the optimized transfer functions to avoid unnecessary CPU round-trips.\n\n")
        
        if 'recommended_batch_size' in results:
            f.write(f"2. **Set batch size to {results['recommended_batch_size']}** - This provides the best throughput based on performance benchmarks.\n\n")
        
        # Add analysis-dependent recommendations
        if 'gradient_profile' in results:
            profile = results['gradient_profile']
            
            # If backward pass is taking a lot of time
            if profile['backward_ms'] > profile['forward_ms'] * 1.5:
                f.write("3. **Optimize backward pass** - The backward pass is taking significantly longer than the forward pass. Consider:\n")
                f.write("   - Using `torch.compile()` for your model\n")
                f.write("   - Applying gradient checkpointing for large models\n")
                f.write("   - Reviewing complex operations in your model\n\n")
            
            # If optimizer step is slow
            if profile['optim_ms'] > profile['total_ms'] * 0.2:  # >20% of time
                f.write("4. **Optimize optimizer operations** - The optimizer step is taking a significant portion of training time. Consider:\n")
                f.write("   - Using a simpler optimizer like SGD for initial training\n")
                f.write("   - Applying optimizer state sharding for distributed training\n")
                f.write("   - Using mixed precision training with `torch.amp`\n\n")
        
        if 'memory_profile' in results and any('cuda' in k for k in results['memory_profile']):
            device = next(k for k in results['memory_profile'] if 'cuda' in k)
            stats = results['memory_profile'][device]
            
            # If memory utilization is low
            if stats['utilization'] < 0.5:
                f.write("5. **Increase batch size further** - GPU memory utilization is low. You can likely increase the batch size to improve throughput.\n\n")
            # If memory allocation is near capacity
            elif stats['allocated_GB'] > stats['reserved_GB'] * 0.9:
                f.write("5. **Implement memory optimizations** - GPU memory usage is near capacity. Consider:\n")
                f.write("   - Enabling gradient checkpointing\n")
                f.write("   - Using mixed precision training\n")
                f.write("   - Reviewing model size and complexity\n\n")
    
    print(f"Performance report generated at {report_path}")
    return report_path

def main():
    """Main execution function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_model(args.model_path, device)
    
    # Collect all results
    results = {}
    
    # Step 1: Memory profiling
    print("\n===== Memory Profile =====")
    input_size = args.input_size or getattr(model, 'input_size', None)
    memory_profile = profile_memory_usage(
        model=model, 
        input_shape=(input_size,) if input_size else None,
        batch_size=args.batch_size,
        detailed=args.detailed
    )
    results['memory_profile'] = memory_profile
    
    # Step 2: CuPy-Torch transfer optimization testing
    if USING_CUPY:
        cupy_torch_results = test_cupy_torch_transfer(
            iterations=args.test_iterations // 10,  # Reduce iterations for transfer test
            tensor_size=(1000, 1000)  # Medium size tensor
        )
        results['cupy_torch_transfer'] = cupy_torch_results
    
    # Step 3: Model performance optimization
    if model is not None and input_size is not None:
        optimization_results = optimize_model(
            model=model,
            batch_size=args.batch_size,
            input_size=input_size,
            iterations=args.test_iterations
        )
        results.update(optimization_results)
        
        # Step 4: Training performance profiling with optimal batch size
        recommended_bs = results.get('recommended_batch_size', args.batch_size)
        gradient_profile = profile_gradients(
            model=model,
            batch_size=recommended_bs,
            input_size=input_size,
            iterations=args.test_iterations // 2  # Reduce iterations for gradient profiling
        )
        results['gradient_profile'] = gradient_profile
    
    # Generate comprehensive report
    generate_performance_report(results, args.output_dir)
    
    print("\nPerformance optimization complete! See report for details and recommendations.")

if __name__ == "__main__":
    main()
