#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for shape error fixes.

This module tests the shape error fixing utilities to ensure they correctly
handle the shape mismatches between model outputs and targets.
"""

import os
import sys
import unittest
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model and fixes
from src.model import BrainInspiredNN
from src.utils.shape_error_fix import reshape_output_for_loss, apply_fixes


class TestShapeFixes(unittest.TestCase):
    """Test the shape error fixes."""
    
    def test_reshape_output_for_loss(self):
        """Test the reshape_output_for_loss function."""
        # Test case 1: 3D output, 2D target
        output = torch.randn(32, 10, 32)  # [batch, seq, features]
        target = torch.randn(32, 1)       # [batch, 1]
        
        fixed_output = reshape_output_for_loss(output, target)
        self.assertEqual(fixed_output.shape, target.shape, 
                         f"Failed to reshape 3D output {output.shape} to match 2D target {target.shape}")
        
        # Test case 2: 2D output with more features than target
        output = torch.randn(32, 32)      # [batch, features]
        target = torch.randn(32, 1)       # [batch, 1]
        
        fixed_output = reshape_output_for_loss(output, target)
        self.assertEqual(fixed_output.shape, target.shape,
                         f"Failed to reshape 2D output {output.shape} to match target {target.shape}")
        
        # Test case 3: Already matching shapes
        output = torch.randn(32, 1)       # [batch, 1]
        target = torch.randn(32, 1)       # [batch, 1]
        
        fixed_output = reshape_output_for_loss(output, target)
        self.assertEqual(fixed_output.shape, target.shape,
                         "Should not change shape when already matching")
        
        # Test case 4: Different batch sizes
        output = torch.randn(32, 5)       # [batch=32, features]
        target = torch.randn(16, 5)       # [batch=16, features]
        
        fixed_output = reshape_output_for_loss(output, target)
        self.assertEqual(fixed_output.shape[1:], target.shape[1:],
                         "Should at least match feature dimensions")
    
    def test_model_with_fixes(self):
        """Test a model with the shape fixes applied."""
        # Create a minimal config
        config = {
            'model': {
                'input_size': 10,
                'hidden_size': 20,
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
        
        # Add debug prints
        print("\nStarting test_model_with_fixes()")
        print(f"Config: {config}")
        
        # Create model and apply fixes
        try:
            print("Creating BrainInspiredNN model")
            model = BrainInspiredNN(config)
            print(f"Model created: {model.__class__.__name__}")
            print("Applying fixes")
            model = apply_fixes(model)
            print("Fixes applied successfully")
            
            # Test with various input shapes
            batch_size = 8
            seq_len = 5
            print(f"Using batch_size={batch_size}, seq_len={seq_len}")
        except Exception as e:
            print(f"Error in model creation or fixing: {e}")
            raise
        
        # Test 2D input
        try:
            print("Creating 2D input tensor")
            x = torch.randn(batch_size, config['model']['input_size'])
            print(f"Input shape: {x.shape}")
            
            print("Running model forward pass")
            model_output = model(x)
            print(f"Model output type: {type(model_output)}")
            
            # Handle tuple output from BioGRU
            if isinstance(model_output, tuple):
                print(f"Output is a tuple with lengths: {len(model_output)}")
                output = model_output[0]  # Extract actual output from tuple
                print(f"Extracted output shape: {output.shape}")
            else:
                output = model_output
                print(f"Output shape: {output.shape}")
        except Exception as e:
            print(f"Error with 2D input: {e}")
            import traceback
            traceback.print_exc()
            raise
            
        self.assertEqual(output.shape, (batch_size, config['model']['output_size']),
                        f"2D input: Expected output shape {(batch_size, config['model']['output_size'])}, "
                        f"got {output.shape}")
        
        # Test 3D input
        x = torch.randn(batch_size, seq_len, config['model']['input_size'])
        model_output = model(x)
        
        # Handle tuple output from BioGRU
        if isinstance(model_output, tuple):
            output = model_output[0]  # Extract actual output from tuple
        else:
            output = model_output
            
        # For 3D input, we expect either 2D or 3D output with last dimension matching output_size
        self.assertEqual(output.shape[-1], config['model']['output_size'],
                        f"3D input: Expected last dimension {config['model']['output_size']}, "
                        f"got {output.shape}")


if __name__ == "__main__":
    unittest.main()
