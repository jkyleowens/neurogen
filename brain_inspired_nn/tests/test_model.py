"""
Test script for the Brain-Inspired Neural Network model.

This script tests the basic functionality of the model components.
"""

import sys
import os
import torch
import numpy as np

# Add the parent directory to the path so we can import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller.persistent_gru import PersistentGRUController
from src.neuromodulator.neuromodulator import NeuromodulatorSystem
from src.model import BrainInspiredNN


def test_persistent_gru_controller():
    """Test the PersistentGRUController."""
    print("Testing PersistentGRUController...")
    
    # Create a controller
    input_size = 64
    hidden_size = 128
    output_size = 32
    persistent_memory_size = 64
    num_layers = 2
    
    controller = PersistentGRUController(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        persistent_memory_size=persistent_memory_size
    )
    
    # Create a random input
    batch_size = 4
    seq_length = 10
    x = torch.randn(batch_size, seq_length, input_size)
    
    # Forward pass
    outputs, hidden_states, persistent_memories = controller(x)
    
    # Check output shape
    assert outputs.shape == (batch_size, seq_length, output_size)
    assert len(hidden_states) == seq_length
    assert len(persistent_memories) == seq_length
    
    print("PersistentGRUController test passed!")


def test_neuromodulator_system():
    """Test the NeuromodulatorSystem."""
    print("Testing NeuromodulatorSystem...")
    
    # Create a neuromodulator system
    input_size = 64
    hidden_size = 128
    
    neuromodulator = NeuromodulatorSystem(
        input_size=input_size,
        hidden_size=hidden_size
    )
    
    # Create random inputs
    batch_size = 4
    x = torch.randn(batch_size, input_size)
    hidden = torch.randn(batch_size, hidden_size)
    reward = torch.randn(batch_size, 1)
    
    # Forward pass
    modulated_hidden, neurotransmitter_levels = neuromodulator(x, hidden, reward)
    
    # Check output shape
    assert modulated_hidden.shape == (batch_size, hidden_size)
    assert all(level.shape == (batch_size, 1) for level in neurotransmitter_levels.values())
    
    print("NeuromodulatorSystem test passed!")


def test_brain_inspired_nn():
    """Test the BrainInspiredNN."""
    print("Testing BrainInspiredNN...")
    
    # Create a model
    input_size = 64
    hidden_size = 128
    output_size = 32
    
    model = BrainInspiredNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size
    )
    
    # Create random inputs
    batch_size = 4
    seq_length = 10
    x = torch.randn(batch_size, seq_length, input_size)
    
    # Forward pass
    outputs, predicted_rewards = model(x)
    
    # Check output shape
    assert outputs.shape == (batch_size, seq_length, output_size)
    assert predicted_rewards.shape == (batch_size, seq_length, 1)
    
    print("BrainInspiredNN test passed!")


def main():
    """Run all tests."""
    print("Running tests for Brain-Inspired Neural Network...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    test_persistent_gru_controller()
    test_neuromodulator_system()
    test_brain_inspired_nn()
    
    print("All tests passed!")


if __name__ == "__main__":
    main()
