#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Manual test for shape error fixes.

This script tests the shape error fixing utilities in a real training scenario.
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model and fixes
from src.model import BrainInspiredNN
from src.utils.shape_error_fix import reshape_output_for_loss, apply_fixes

def create_dummy_data(batch_size=32, seq_len=10, input_size=10, output_size=1):
    """Create dummy data for testing."""
    # Create data with different shapes
    x_2d = torch.randn(batch_size, input_size)
    y_2d = torch.randn(batch_size, output_size)
    
    x_3d = torch.randn(batch_size, seq_len, input_size)
    y_3d = torch.randn(batch_size, seq_len, output_size)
    
    return (x_2d, y_2d), (x_3d, y_3d)

def test_shape_fixes():
    """Test shape fixes with a real model."""
    print("\n===== Testing Shape Error Fixes =====")
    
    # Create model config
    config = {
        'model': {
            'input_size': 10,
            'hidden_size': 32,
            'output_size': 1,
            'use_bio_gru': True
        },
        'controller': {
            'persistent_memory_size': 16,
            'num_layers': 2,
            'dropout': 0.1
        },
        'training': {
            'learning_rate': 0.001
        }
    }
    
    # Create test data
    (x_2d, y_2d), (x_3d, y_3d) = create_dummy_data(
        batch_size=8, 
        input_size=config['model']['input_size'],
        output_size=config['model']['output_size']
    )
    
    # Create model
    print("\n1. Creating model...")
    model = BrainInspiredNN(config)
    
    # Test without fixes
    try:
        print("\n2. Testing model without fixes...")
        output_2d = model(x_2d)
        print(f"2D input: {x_2d.shape} -> output: {output_2d.shape}")
        
        output_3d = model(x_3d)
        print(f"3D input: {x_3d.shape} -> output: {output_3d.shape}")
    except Exception as e:
        print(f"Error without fixes: {e}")
    
    # Apply fixes
    print("\n3. Applying shape fixes...")
    model = apply_fixes(model)
    if hasattr(model, 'configure_shape_awareness'):
        model.configure_shape_awareness(target_shape=(config['model']['output_size'],), auto_adjust=True)
    
    # Test with fixes
    try:
        print("\n4. Testing model with fixes...")
        output_2d = model(x_2d)
        
        if isinstance(output_2d, tuple):
            actual_output = output_2d[0]
            print(f"2D input: {x_2d.shape} -> output tuple with shape: {actual_output.shape}")
            
            # Test if loss calculation would work with tuple output
            output_for_loss = reshape_output_for_loss(output_2d, y_2d)
            print(f"Reshaped for loss (from tuple): {output_for_loss.shape}, target: {y_2d.shape}")
            
            # Test MSE loss
            loss = torch.nn.MSELoss()(output_for_loss, y_2d)
            print(f"Loss calculation successful: {loss.item()}")
        else:
            print(f"2D input: {x_2d.shape} -> output: {output_2d.shape}")
            
            # Test if loss calculation would work
            output_for_loss = reshape_output_for_loss(output_2d, y_2d)
            print(f"Reshaped for loss: {output_for_loss.shape}, target: {y_2d.shape}")
            
            # Test MSE loss
            loss = torch.nn.MSELoss()(output_for_loss, y_2d)
            print(f"Loss calculation successful: {loss.item()}")
        
        # Test with 3D input
        output_3d = model(x_3d)
        
        if isinstance(output_3d, tuple):
            actual_output = output_3d[0]
            print(f"3D input: {x_3d.shape} -> output tuple with shape: {actual_output.shape}")
            
            # Test if loss calculation would work with tuple output (different dimensions)
            reshaped_output = reshape_output_for_loss(output_3d, y_2d)
            print(f"3D input -> 2D target reshape: {reshaped_output.shape}, target: {y_2d.shape}")
            
            # Calculate loss with 2D target
            loss = torch.nn.MSELoss()(reshaped_output, y_2d)
            print(f"Loss calculation (3D->2D) successful: {loss.item()}")
            
            # Also test with matching dimensions
            reshaped_output_3d = reshape_output_for_loss(output_3d, y_3d)
            print(f"3D input -> 3D target reshape: {reshaped_output_3d.shape}, target: {y_3d.shape}")
            loss_3d = torch.nn.MSELoss()(reshaped_output_3d, y_3d)
            print(f"Loss calculation (3D->3D) successful: {loss_3d.item()}")
        else:
            print(f"3D input: {x_3d.shape} -> output: {output_3d.shape}")
            
            # Test reshaping with 2D target
            reshaped_output = reshape_output_for_loss(output_3d, y_2d)
            print(f"3D input -> 2D target reshape: {reshaped_output.shape}, target: {y_2d.shape}")
            
            # Calculate loss
            loss = torch.nn.MSELoss()(reshaped_output, y_2d)
            print(f"Loss calculation successful: {loss.item()}")
            
    except Exception as e:
        print(f"Error with fixes: {e}")
        import traceback
        traceback.print_exc()

    print("\n===== Shape Error Fix Testing Complete =====")

if __name__ == "__main__":
    test_shape_fixes()
