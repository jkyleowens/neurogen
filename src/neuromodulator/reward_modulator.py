"""
Reward-Based Neuromodulator

This module implements a neuromodulation system based on reward signals
for the Brain-Inspired Neural Network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RewardModulator(nn.Module):
    """
    A neuromodulation system based on reward signals.
    
    This system adjusts the learning rate and other parameters
    based on the reward signals received from the environment.
    """
    
    def __init__(self, hidden_size, dopamine_scale=1.0, serotonin_scale=1.0, 
                 norepinephrine_scale=1.0, acetylcholine_scale=1.0, reward_decay=0.95):
        """
        Initialize the Reward Modulator.
        
        Args:
            hidden_size (int): Size of hidden state
            dopamine_scale (float): Scale factor for dopamine
            serotonin_scale (float): Scale factor for serotonin
            norepinephrine_scale (float): Scale factor for norepinephrine
            acetylcholine_scale (float): Scale factor for acetylcholine
            reward_decay (float): Decay rate for reward signals
        """
        super(RewardModulator, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_modulators = 4  # dopamine, serotonin, norepinephrine, acetylcholine
        
        # Scale factors for each neurotransmitter
        self.scales = {
            'dopamine': dopamine_scale,
            'serotonin': serotonin_scale,
            'norepinephrine': norepinephrine_scale,
            'acetylcholine': acetylcholine_scale
        }
        
        self.reward_decay = reward_decay
        
        # Network to compute modulator levels
        self.modulator_network = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),  # +1 for reward signal
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, self.num_modulators),
            nn.Sigmoid()  # Normalize modulator levels between 0 and 1
        )
        
        # Initial modulator levels
        self.modulator_levels = None
        
        # Decay rates for each modulator
        self.decay_rates = nn.Parameter(
            torch.tensor([0.9, 0.8, 0.7, 0.95][:self.num_modulators], 
                         dtype=torch.float32)
        )
    
    def init_levels(self, batch_size, device):
        """
        Initialize neurotransmitter levels.
        
        Args:
            batch_size (int): Batch size
            device (torch.device): Device to create tensors on
            
        Returns:
            torch.Tensor: Initialized neurotransmitter levels
        """
        self.modulator_levels = torch.ones(
            batch_size, self.num_modulators, 
            device=device
        ) * 0.5  # Initialize at moderate levels
        
        return self.modulator_levels
    
    def forward(self, x, reward, current_levels=None):
        """
        Update modulator levels based on input and reward.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, hidden_size)
            reward (torch.Tensor): Reward signal of shape (batch_size, 1)
            current_levels (torch.Tensor, optional): Current neurotransmitter levels
            
        Returns:
            tuple: (modulated_output, updated_levels)
        """
        batch_size = x.size(0)
        
        # Initialize levels if not provided
        if current_levels is None:
            if self.modulator_levels is None:
                self.modulator_levels = self.init_levels(batch_size, x.device)
            current_levels = self.modulator_levels
        
        # Ensure reward is properly shaped
        if reward.dim() == 1:
            reward = reward.unsqueeze(1)
            
        # Concatenate input and reward
        combined_input = torch.cat([x[:, :self.hidden_size], reward], dim=1)
        
        # Compute new modulator levels
        new_levels = self.modulator_network(combined_input)
        
        # Apply decay to current levels
        decayed_levels = current_levels * self.reward_decay
        
        # Apply adaptive blending based on reward magnitude
        # Higher reward = more weight to new levels to increase adaptability
        reward_magnitude = torch.abs(reward).mean()
        adaptive_blend_rate = torch.clamp(0.3 + 0.3 * reward_magnitude, 0.2, 0.5)
        updated_levels = (1 - adaptive_blend_rate) * decayed_levels + adaptive_blend_rate * new_levels
        
        # Apply regularization to prevent extreme values that lead to overfitting
        # This ensures modulator values stay in a reasonable range and prevents dominance
        updated_levels = torch.clamp(updated_levels, 0.2, 0.8)
        self.modulator_levels = updated_levels
        
        # Apply modulation to the input
        # Each neurotransmitter affects different aspects of the signal
        dopamine = updated_levels[:, 0].unsqueeze(1) * self.scales['dopamine']
        serotonin = updated_levels[:, 1].unsqueeze(1) * self.scales['serotonin']
        norepinephrine = updated_levels[:, 2].unsqueeze(1) * self.scales['norepinephrine']
        acetylcholine = updated_levels[:, 3].unsqueeze(1) * self.scales['acetylcholine']
        
        # Modulate the input signal with balanced coefficients to reduce overfitting
        # - Dopamine: Enhances signal strength but can lead to overfitting if too strong
        # - Serotonin: Regulates signal stability and acts as regularizer
        # - Norepinephrine: Increases signal-to-noise ratio, helps focus on important features
        # - Acetylcholine: Enhances signal precision, but needs balance
        
        # Calculate anti-overfitting factor based on modulator levels
        # When dopamine is high (reward-seeking), increase regularization
        regularization_factor = torch.sigmoid(dopamine.mean() * 5 - 2.0)
        
        modulated_output = (
            x * (1.0 + 0.15 * dopamine) +  # Reduced dopamine influence to prevent overfitting
            (0.1 + 0.05 * regularization_factor) * serotonin * torch.tanh(x) +  # Increase stabilization when needed
            0.1 * norepinephrine * torch.sign(x) * torch.abs(x) +  # Maintain signal-to-noise ratio
            0.1 * acetylcholine * torch.sigmoid(x) +  # Maintain precision
            0.05 * torch.randn_like(x) * regularization_factor  # Add small noise when regularization needed
        )
        
        return modulated_output, updated_levels
    
    def get_learning_rate_modifier(self):
        """
        Get a modifier for the learning rate based on modulator levels.
        
        Returns:
            torch.Tensor: Learning rate modifier
        """
        if self.modulator_levels is None:
            return 1.0
        
        # Use the first modulator as the learning rate modifier
        return torch.mean(self.modulator_levels[:, 0])
    
    def get_exploration_rate(self):
        """
        Get the exploration rate based on modulator levels.
        
        Returns:
            torch.Tensor: Exploration rate
        """
        if self.modulator_levels is None:
            return 0.1
        
        # Use the second modulator as the exploration rate
        return torch.mean(self.modulator_levels[:, 1])
    
    def get_levels(self, levels=None):
        """
        Get the current neurotransmitter levels.
        
        Args:
            levels (torch.Tensor, optional): Levels to return, defaults to self.modulator_levels
            
        Returns:
            dict: Dictionary of neurotransmitter levels
        """
        if levels is None:
            levels = self.modulator_levels
            
        if levels is None:
            return {
                'dopamine': 0.5,
                'serotonin': 0.5,
                'norepinephrine': 0.5,
                'acetylcholine': 0.5
            }
        
        # Map tensor indices to neurotransmitter names
        neurotransmitters = ['dopamine', 'serotonin', 'norepinephrine', 'acetylcholine']
        
        # Calculate mean levels across batch dimension
        mean_levels = torch.mean(levels, dim=0)
        
        # Create dictionary of levels
        result = {}
        for i, name in enumerate(neurotransmitters[:self.num_modulators]):
            result[name] = mean_levels[i].item() * self.scales[name]
            
        return result
