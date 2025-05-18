
"""
Test script for the neuromodulator system.

This script tests the functionality of the neuromodulator system and its integration
with the persistent GRU controller.
"""

import torch
import unittest
import sys
import os

# Add the parent directory to the path so we can import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller.persistent_gru import PersistentGRUController
from src.neuromodulator.neuromodulator import NeuromodulatorSystem, RewardPredictor, NeurotransmitterPool, DynamicSynapseModel


class TestNeuromodulatorSystem(unittest.TestCase):
    """Test cases for the neuromodulator system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.seq_length = 5
        self.input_size = 16
        self.hidden_size = 32
        self.output_size = 8
        self.persistent_memory_size = 64
        
        # Create test inputs
        self.inputs = torch.randn(self.batch_size, self.seq_length, self.input_size)
        self.rewards = torch.randn(self.batch_size, self.seq_length, 1)
        
        # Create controller
        self.controller = PersistentGRUController(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            persistent_memory_size=self.persistent_memory_size
        )
        
        # Create neuromodulator system
        self.neuromodulator = NeuromodulatorSystem(
            input_size=self.input_size,
            hidden_size=self.hidden_size
        )
        
        # Create reward predictor
        self.reward_predictor = RewardPredictor(
            state_size=self.hidden_size,
            action_size=self.output_size,
            hidden_size=self.hidden_size
        )
    
    def test_neurotransmitter_pool(self):
        """Test the neurotransmitter pool functionality."""
        pool = NeurotransmitterPool()
        
        # Test initial state
        self.assertEqual(pool.get_stored_amount(), pool.baseline_level)
        self.assertEqual(pool.get_synaptic_level(), 0.0)
        
        # Test release
        release_signal = torch.tensor(1.0)
        released = pool.release(release_signal)
        self.assertGreater(released.item(), 0.0)
        self.assertLess(pool.get_stored_amount(), pool.baseline_level)
        self.assertGreater(pool.get_synaptic_level(), 0.0)
        
        # Test update
        old_synaptic = pool.get_synaptic_level()
        pool.update()
        # Synaptic level should decrease due to reuptake and decay
        self.assertLess(pool.get_synaptic_level(), old_synaptic)
        
        # Test reset
        pool.reset()
        self.assertEqual(pool.get_stored_amount(), pool.baseline_level)
        self.assertEqual(pool.get_synaptic_level(), 0.0)
    
    def test_dynamic_synapse_model(self):
        """Test the dynamic synapse model functionality."""
        model = DynamicSynapseModel(self.input_size, self.hidden_size)
        
        # Test forward pass
        hidden = torch.randn(self.batch_size, self.hidden_size)
        output = model(hidden)
        self.assertEqual(output.shape, hidden.shape)
        
        # Test connection update
        weights_before = model.get_synaptic_weights().clone()
        reward = torch.tensor(1.0)
        model.update_connections(hidden, reward)
        weights_after = model.get_synaptic_weights()
        # Weights should change after update
        self.assertFalse(torch.allclose(weights_before, weights_after))
        
        # Test reset
        model.reset_connections()
        weights_reset = model.get_synaptic_weights()
        self.assertTrue(torch.all(weights_reset == 1.0))
    
    def test_neuromodulator_system(self):
        """Test the neuromodulator system functionality."""
        # Test forward pass
        hidden = torch.randn(self.batch_size, self.hidden_size)
        input_data = torch.randn(self.batch_size, self.input_size)
        reward = torch.randn(self.batch_size, 1)
        
        modulated_hidden, neurotransmitter_levels = self.neuromodulator(input_data, hidden, reward)
        
        # Check output shapes
        self.assertEqual(modulated_hidden.shape, hidden.shape)
        self.assertEqual(len(neurotransmitter_levels), 4)  # Four neurotransmitters
        
        # Check neurotransmitter levels
        for nt_name, nt_level in neurotransmitter_levels.items():
            self.assertEqual(nt_level.shape, (self.batch_size, 1))
        
        # Test reset
        self.neuromodulator.reset()
        nt_levels = self.neuromodulator.get_neurotransmitter_levels()
        for nt_name, nt_level in nt_levels.items():
            self.assertEqual(nt_level, 0.0)
    
    def test_reward_predictor(self):
        """Test the reward predictor functionality."""
        # Test forward pass
        state = torch.randn(self.batch_size, self.hidden_size)
        action = torch.randn(self.batch_size, self.output_size)
        
        predicted_reward = self.reward_predictor(state, action)
        
        # Check output shape
        self.assertEqual(predicted_reward.shape, (self.batch_size, 1))
        
        # Test TD error computation
        next_state = torch.randn(self.batch_size, self.hidden_size)
        reward = torch.randn(self.batch_size, 1)
        done = torch.zeros(self.batch_size, 1)
        
        td_error = self.reward_predictor.compute_td_error(state, action, reward, next_state, done)
        
        # Check TD error shape
        self.assertEqual(td_error.shape, (self.batch_size, 1))
    
    def test_integration_with_controller(self):
        """Test the integration of the neuromodulator system with the controller."""
        # Initialize hidden states and persistent memories
        hidden, persistent_memory = self.controller.init_hidden(self.batch_size, self.inputs.device)
        
        # Process inputs through controller
        outputs, hidden_states, persistent_memories = self.controller(self.inputs)
        
        # Check that controller outputs have the expected shape
        self.assertEqual(outputs.shape, (self.batch_size, self.seq_length, self.output_size))
        
        # Process the last hidden state through the neuromodulator
        last_hidden = hidden_states[-1][-1]  # Last layer, last time step
        input_data = self.inputs[:, -1, :]  # Last time step input
        reward = self.rewards[:, -1, :]  # Last time step reward
        
        modulated_hidden, neurotransmitter_levels = self.neuromodulator(input_data, last_hidden, reward)
        
        # Check that modulated hidden state has the expected shape
        self.assertEqual(modulated_hidden.shape, last_hidden.shape)
        
        # Test that we can get neurotransmitter levels
        nt_levels = self.neuromodulator.get_neurotransmitter_levels()
        self.assertEqual(len(nt_levels), 4)  # Four neurotransmitters
        
        # Test that we can get synaptic weights
        synaptic_weights = self.neuromodulator.get_synaptic_weights()
        self.assertEqual(synaptic_weights.shape, (self.hidden_size, self.hidden_size))


if __name__ == '__main__':
    unittest.main()
