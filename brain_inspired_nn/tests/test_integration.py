"""
Integration Test for Brain-Inspired Neural Network

This script tests the integration of all components of the brain-inspired neural network
with simplified mocks to avoid dimension mismatches.
"""

import sys
import os
import torch
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller.persistent_gru import PersistentGRUController
from src.neuromodulator.neuromodulator import NeuromodulatorSystem
from src.model import BrainInspiredNN


class MockInputProcessor(torch.nn.Module):
    """Mock input processor that simply passes through the input."""
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, x, adapt=True):
        # Return the input unchanged and a mock metadata dictionary
        return x, {'type_probs': torch.ones(4), 'attention_weights': torch.ones(x.shape[0], x.shape[1])}
    
    def reset_adaptation(self):
        pass


class MockLLMInterface:
    """Mock LLM interface that returns dummy responses."""
    def __init__(self, *args, **kwargs):
        pass
    
    def connect(self):
        return True
    
    def get_response(self, prompt, streaming=False, callback=None, **kwargs):
        return "This is a mock LLM response."
    
    def model_output_to_prompt(self, model_output):
        return "Mock prompt from model output."
    
    def train_with_llm_feedback(self, model, inputs, targets, optimizer, loss_fn, epochs=1):
        return {"loss_history": [], "feedback_history": []}


def test_model_integration():
    """Test the integration of controller and neuromodulator components."""
    print("Testing model integration...")
    
    # Create model components
    input_size = 64
    hidden_size = 128
    output_size = 32
    
    # Create controller
    controller = PersistentGRUController(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=hidden_size
    )
    
    # Create neuromodulator
    neuromodulator = NeuromodulatorSystem(
        input_size=input_size,
        hidden_size=hidden_size
    )
    
    # Create random input
    batch_size = 4
    seq_length = 10
    x = torch.randn(batch_size, seq_length, input_size)
    
    # Process through controller
    controller_outputs, hidden_states, persistent_memories = controller(x)
    
    # Process through neuromodulator
    last_hidden = controller_outputs[:, -1, :]
    modulated_output, neurotransmitter_levels = neuromodulator(
        x[:, -1, :], last_hidden, torch.randn(batch_size, 1)
    )
    
    # Check shapes
    assert controller_outputs.shape == (batch_size, seq_length, hidden_size)
    assert modulated_output.shape == (batch_size, hidden_size)
    
    print("Model integration test passed!")


@patch('src.model.InputProcessor', MockInputProcessor)
@patch('src.model.LLMInterface', MockLLMInterface)
def test_full_model_integration():
    """Test the full BrainInspiredNN model with mocked components."""
    print("Testing full model integration...")
    
    # Create model
    input_size = 64
    hidden_size = 128
    output_size = 32
    
    model = BrainInspiredNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size
    )
    
    # Create random input
    batch_size = 4
    seq_length = 10
    x = torch.randn(batch_size, seq_length, input_size)
    
    # Forward pass
    outputs, predicted_rewards = model(x)
    
    # Check shapes
    assert outputs.shape == (batch_size, seq_length, output_size)
    assert predicted_rewards.shape == (batch_size, seq_length, 1)
    
    # Check that internal states are set
    assert model.hidden_states is not None
    assert model.persistent_memories is not None
    assert model.neurotransmitter_levels is not None
    assert model.input_metadata is not None
    
    # Reset states and check
    model.reset_states()
    assert model.hidden_states is None
    assert model.persistent_memories is None
    assert model.neurotransmitter_levels is None
    assert model.input_metadata is None
    
    print("Full model integration test passed!")


@patch('src.model.InputProcessor', MockInputProcessor)
@patch('src.model.LLMInterface', MockLLMInterface)
def test_reward_based_learning():
    """Test that the model can learn from reward signals."""
    print("Testing reward-based learning...")
    
    # Create model
    input_size = 64
    hidden_size = 128
    output_size = 32
    
    model = BrainInspiredNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size
    )
    
    # Create random input
    batch_size = 4
    seq_length = 10
    x = torch.randn(batch_size, seq_length, input_size)
    
    # Forward pass with positive reward
    positive_reward = torch.ones(batch_size, seq_length, 1)
    outputs1, _ = model(x, external_reward=positive_reward)
    
    # Get connection strengths
    connections1 = model.get_dynamic_connections()
    
    # Reset model
    model.reset_states()
    
    # Forward pass with negative reward
    negative_reward = -torch.ones(batch_size, seq_length, 1)
    outputs2, _ = model(x, external_reward=negative_reward)
    
    # Get connection strengths
    connections2 = model.get_dynamic_connections()
    
    # Check that connection strengths are different
    # This verifies that the model adapts based on reward signals
    for c1, c2 in zip(connections1, connections2):
        assert not torch.allclose(c1, c2)
    
    print("Reward-based learning test passed!")


@patch('src.model.InputProcessor', MockInputProcessor)
@patch('src.model.LLMInterface', MockLLMInterface)
def test_persistent_memory():
    """Test that persistent memory affects model behavior over time."""
    print("Testing persistent memory...")
    
    # Create model
    input_size = 64
    hidden_size = 128
    output_size = 32
    
    model = BrainInspiredNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        persistent_memory_size=64
    )
    
    # Create random inputs
    batch_size = 4
    seq_length = 10
    x1 = torch.randn(batch_size, seq_length, input_size)
    x2 = torch.randn(batch_size, seq_length, input_size)
    
    # First forward pass
    outputs1, _ = model(x1)
    
    # Second forward pass with different input but same persistent memory
    outputs2, _ = model(x2)
    
    # Reset persistent memory
    model.reset_states()
    
    # Third forward pass with same input as second but reset memory
    outputs3, _ = model(x2)
    
    # Check that outputs2 and outputs3 are different due to persistent memory effects
    assert not torch.allclose(outputs2, outputs3)
    
    print("Persistent memory test passed!")


def main():
    """Run all integration tests."""
    print("Running integration tests for Brain-Inspired Neural Network...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    test_model_integration()
    test_full_model_integration()
    test_reward_based_learning()
    test_persistent_memory()
    
    print("All integration tests passed!")


if __name__ == "__main__":
    main()
