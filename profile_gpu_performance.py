#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU Performance Analysis and Optimization Script

This script performs a comprehensive analysis of GPU performance for the neural network
trading system, focusing on:

1. CuPy-PyTorch data transfer efficiency
2. Memory allocation and utilization
3. Batch size optimization
4. Model architecture performance

Run this script to get a complete report on GPU performance and optimization recommendations.
"""

import os
import sys
import torch
import argparse
import json
from datetime import datetime
import warnings

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cupy as cp
    USING_CUPY = True
except ImportError:
    import numpy as cp
    USING_CUPY = False
    warnings.warn("CuPy not available. Some profiling features will be limited.")

# Import model and utilities
from src.model import BrainInspiredNN
from src.utils.memory_utils import (
    efficient_cp_to_torch,
    efficient_torch_to_cp,
    optimize_batch_size,
    profile_memory_usage
)
from src.utils.gpu_profiler import (
    profile_gpu_performance,
    profile_cupy_torch_transfers,
    generate_optimization_report,
    apply_optimization_recommendations
)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='GPU Performance Analysis and Optimization')
    parser.add_argument('--model-path', type=str, default="neurogen/models/checkpoints/best_model.pt",
                      help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for profiling')
    parser.add_argument('--test-batch-sizes', type=str, default="32,64,128",
                      help='Comma-separated batch sizes to test')
    parser.add_argument('--input-size', type=int, default=None,
                      help='Input size (will be inferred from model if not specified)')
    parser.add_argument('--output-dir', type=str, default='results/performance_optimization',
                      help='Directory to save results')
    parser.add_argument('--apply-optimizations', action='store_true',
                      help='Apply recommended optimizations')
    parser.add_argument('--verbose', action='store_true',
                      help='Print verbose output')
    return parser.parse_args()

def load_model(checkpoint_path, device='cuda'):
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model and config from checkpoint
        if 'model_state_dict' in checkpoint:
            # Standard checkpoint format
            model_state = checkpoint['model_state_dict']
            config = checkpoint.get('config', {})
            
            model = BrainInspiredNN(config).to(device)
            model.load_state_dict(model_state)
        elif isinstance(checkpoint, dict) and 'config' in checkpoint:
            # Direct model storage
            config = checkpoint.get('config', {})
            model = BrainInspiredNN(config).to(device)
        else:
            # Try loading directly
            model = checkpoint
        
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        
        # Create a fallback small model for testing
        print("Creating fallback model for profiling...")
        
        config = {
            'model': {
                'input_size': 5,
                'hidden_size': 32,
                'output_size': 1
            },
            'controller': {
                'num_layers': 1,
                'persistent_memory_size': 16
            }
        }
        
        model = BrainInspiredNN(config).to(device)
        print("Fallback model created.")
        
        return model

def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Determine input size if not specified
    input_size = args.input_size or getattr(model, 'input_size', 5)
    print(f"Using input size: {input_size}")
    
    # Parse test batch sizes
    test_batch_sizes = [int(x) for x in args.test_batch_sizes.split(',')]
    print(f"Testing batch sizes: {test_batch_sizes}")
    
    # Perform basic GPU info check
    print("\n===== GPU Information =====")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            print(f"GPU {i}: {gpu_name}, {total_mem:.2f} GB total memory")
    else:
        print("No CUDA-compatible GPU detected")
    
    # Print CuPy status
    print("\n===== CuPy Status =====")
    if USING_CUPY:
        print(f"CuPy version: {cp.__version__}")
        
        # Get CuPy memory pool info
        try:
            mem_pool = cp.get_default_memory_pool()
            used_bytes = mem_pool.used_bytes()
            total_bytes = mem_pool.total_bytes()
            print(f"CuPy memory pool: {used_bytes/(1024**2):.2f} MB used, {total_bytes/(1024**2):.2f} MB total")
        except Exception as e:
            print(f"Error getting CuPy memory pool info: {e}")
    else:
        print("CuPy not available. Using NumPy for array operations.")
        print("For better performance, install CuPy: pip install cupy")
    
    # Test data transfer efficiency
    print("\n===== Data Transfer Efficiency =====")
    transfer_profile = profile_cupy_torch_transfers(args.batch_size, (input_size,), device)
    
    # Profile memory usage
    print("\n===== Memory Usage =====")
    memory_profile = profile_memory_usage(model, (input_size,), args.batch_size, detailed=True)
    
    # Test batch size efficiency
    print("\n===== Batch Size Optimization =====")
    optimal_batch_size = optimize_batch_size(model, (input_size,), args.batch_size, max_memory_usage=0.8)
    print(f"Optimal batch size: {optimal_batch_size}")
    
    # Generate full optimization report
    print("\n===== Generating Optimization Report =====")
    report = generate_optimization_report(model, (input_size,), test_batch_sizes)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(args.output_dir, f"performance_report_{timestamp}.json")
    
    # Convert any non-serializable values to strings
    serializable_report = json.loads(json.dumps(report, default=str))
    
    with open(result_file, 'w') as f:
        json.dump(serializable_report, f, indent=2)
    
    # Generate markdown report for human readability
    markdown_file = os.path.join(args.output_dir, f"performance_report_{timestamp}.md")
    
    with open(markdown_file, 'w') as f:
        f.write("# GPU Performance Optimization Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## GPU Information\n\n")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                f.write(f"- GPU {i}: {gpu_name}, {total_mem:.2f} GB total memory\n")
        else:
            f.write("- No CUDA-compatible GPU detected\n")
        
        f.write("\n## CuPy Status\n\n")
        if USING_CUPY:
            f.write(f"- CuPy version: {cp.__version__}\n")
            try:
                mem_pool = cp.get_default_memory_pool()
                used_bytes = mem_pool.used_bytes()
                total_bytes = mem_pool.total_bytes()
                f.write(f"- CuPy memory pool: {used_bytes/(1024**2):.2f} MB used, {total_bytes/(1024**2):.2f} MB total\n")
            except Exception:
                pass
        else:
            f.write("- CuPy not available. Using NumPy for array operations.\n")
            f.write("- For better performance, install CuPy: pip install cupy\n")
        
        f.write("\n## Data Transfer Efficiency\n\n")
        f.write("| Method | Time (ms) | Speedup |\n")
        f.write("|--------|-----------|--------|\n")
        
        standard_time = transfer_profile.get('standard_time_ms', 0)
        if standard_time > 0:
            f.write(f"| Standard as_tensor | {standard_time:.2f} | 1.00x |\n")
            
            efficient_time = transfer_profile.get('efficient_time_ms', 0)
            if efficient_time > 0:
                speedup = standard_time / efficient_time
                f.write(f"| Efficient transfer | {efficient_time:.2f} | {speedup:.2f}x |\n")
            
            dlpack_time = transfer_profile.get('dlpack_time_ms')
            if dlpack_time is not None and dlpack_time > 0:
                speedup = standard_time / dlpack_time
                f.write(f"| DLPack | {dlpack_time:.2f} | {speedup:.2f}x |\n")
        
        f.write("\n## Batch Size Performance\n\n")
        f.write("| Batch Size | Step Time (s) | Samples/Second |\n")
        f.write("|------------|---------------|----------------|\n")
        
        for bs, metrics in report.get('batch_size_results', {}).items():
            avg_time = metrics.get('avg_step_time', 0)
            samples_per_sec = metrics.get('samples_per_second', 0)
            f.write(f"| {bs} | {avg_time:.4f} | {samples_per_sec:.2f} |\n")
        
        f.write("\n## Memory Usage\n\n")
        for device_name, stats in memory_profile.items():
            f.write(f"### {device_name}\n\n")
            f.write(f"- Allocated: {stats.get('allocated_GB', 0):.2f} GB\n")
            f.write(f"- Reserved: {stats.get('reserved_GB', 0):.2f} GB\n")
            f.write(f"- Max Allocated: {stats.get('max_allocated_GB', 0):.2f} GB\n")
        
        f.write("\n## Optimization Recommendations\n\n")
        for i, rec in enumerate(report.get('recommendations', [])):
            f.write(f"{i+1}. {rec}\n")
    
    print(f"Reports saved to {result_file} and {markdown_file}")
    
    # Apply optimizations if requested
    if args.apply_optimizations:
        print("\n===== Applying Optimizations =====")
        model = apply_optimization_recommendations(model, report)
        
        # Save optimized model
        optimized_model_path = os.path.join(args.output_dir, f"optimized_model_{timestamp}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': getattr(model, 'config', {}),
            'optimization_report': serializable_report
        }, optimized_model_path)
        
        print(f"Optimized model saved to {optimized_model_path}")
    
    print("\nPerformance analysis complete!")

if __name__ == "__main__":
    main()
