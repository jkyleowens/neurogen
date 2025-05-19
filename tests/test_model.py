#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the Brain-Inspired Neural Network model.

This module contains unit tests for the BrainInspiredNN model
and its components.
"""

import sys
import os
import unittest
import torch
import numpy as np
import yaml

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import the model and components
from src.model import BrainInspiredNN
from src.controller.persistent_gru import PersistentGRUController
from src.neuromodulator.reward_modulator import RewardModulator
from src.utils.memory_utils import tensor_shape_fix

class TestPersistentGRUController(unittest.TestCase):
    """Tests for the PersistentGRUController class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_size = 10
        self.hidden_size = 20
        self.output_size = 5
        self.batch_size = 8
        self.seq_length = 15
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create controller
        self.controller = PersistentGRUController(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size
        ).to(self.device)
    
    def test_init_hidden(self):
        """Test initialization of hidden state."""
        hidden = self.controller.init_hidden(self.batch_size, self.device)
        
        # Check shape
        self.assertEqual(hidden.shape, (1, self.batch_size, self.hidden_size))
        
        # Check device
        self.assertEqual(hidden.device, self.device)
        
        # Check that persistent memory was initialized
        self.assertIsNotNone(self.controller.persistent_memory)
        self.assertEqual(
            self.controller.persistent_memory.shape,
            (1, self.batch_size, self.hidden_size)
        )
    
    def test_forward(self):
        """Test forward pass."""
        # Create input tensor
        x = torch.randn(
            self.batch_size, self.seq_length, self.input_size,
            device=self.device
        )
        
        # Initialize hidden state
        hidden = self.controller.init_hidden(self.batch_size, self.device)
        
        # Forward pass
        output, new_hidden = self.controller(x, hidden)
        
        # Check output shape
        self.assertEqual(
            output.shape,
            (self.batch_size, self.seq_length, self.output_size)
        )
        
        # Check hidden state shape
        self.assertEqual(
            new_hidden.shape,
            (1, self.batch_size, self.hidden_size)
        )

class TestRewardModulator(unittest.TestCase):
    """Tests for the RewardModulator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_size = 10
        self.hidden_size = 20
        self.num_modulators = 4
        self.batch_size = 8
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create modulator
        self.modulator = RewardModulator(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_modulators=self.num_modulators
        ).to(self.device)
    
    def test_init_modulators(self):
        """Test initialization of modulator levels."""
        self.modulator.init_modulators(self.batch_size, self.device)
        
        # Check shape
        self.assertEqual(
            self.modulator.modulator_levels.shape,
            (self.batch_size, self.num_modulators)
        )
        
        # Check device
        self.assertEqual(self.modulator.modulator_levels.device, self.device)
        
        # Check initial values
        self.assertTrue(torch.all(self.modulator.modulator_levels == 0.5))
    
    def test_forward(self):
        """Test forward pass."""
        # Create input tensor
        x = torch.randn(self.batch_size, self.input_size, device=self.device)
        reward = torch.randn(self.batch_size, 1, device=self.device)
        
        # Initialize modulator levels
        self.modulator.init_modulators(self.batch_size, self.device)
        
        # Forward pass
        levels = self.modulator(x, reward)
        
        # Check output shape
        self.assertEqual(levels.shape, (self.batch_size, self.num_modulators))
        
        # Check values are between 0 and 1
        self.assertTrue(torch.all(levels >= 0) and torch.all(levels <= 1))

class TestMemoryUtils(unittest.TestCase):
    """Tests for memory utility functions."""
    
    def test_tensor_shape_fix(self):
        """Test tensor shape fixing."""
        # Create tensor
        tensor = torch.randn(10, 20)
        
        # Test with matching shape
        fixed_tensor = tensor_shape_fix(tensor, (10, 20))
        self.assertEqual(fixed_tensor.shape, (10, 20))
        self.assertTrue(torch.all(fixed_tensor == tensor))
        
        # Test with reshapable shape
        fixed_tensor = tensor_shape_fix(tensor, (20, 10))
        self.assertEqual(fixed_tensor.shape, (20, 10))
        
        # Test with padding
        fixed_tensor = tensor_shape_fix(tensor, (15, 25))
        self.assertEqual(fixed_tensor.shape, (15, 25))
        
        # Check that original data was preserved
        self.assertTrue(torch.all(fixed_tensor[:10, :20] == tensor))

if __name__ == '__main__':
    unittest.main()
