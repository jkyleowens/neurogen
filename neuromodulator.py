
"""
Neuromodulator System

This module implements a biologically-inspired neuromodulator system that modulates
neural activity based on reward signals, similar to how neurotransmitters like
dopamine, serotonin, norepinephrine, and acetylcholine function in the brain.

The implementation includes:
1. A system for modeling different neurotransmitters and their effects
2. A mechanism for outputting finite amounts of neurotransmitters based on reward signals
3. A dynamic connection mechanism that adjusts synaptic strengths based on neurotransmitter levels
4. Methods for integrating with the persistent GRU controller
5. Support for reward prediction and error calculation

Biological inspiration:
- Dopamine: Primarily involved in reward prediction and reinforcement learning
- Serotonin: Regulates mood, behavioral inhibition, and social behavior
- Norepinephrine: Modulates attention, arousal, and vigilance
- Acetylcholine: Influences learning, memory, and attentional focus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NeurotransmitterPool(nn.Module):
    """
    A biologically-inspired neurotransmitter pool that models the finite resources
    and dynamics of neurotransmitter release, reuptake, and synthesis.
    
    Biological inspiration:
    - Neurotransmitters are stored in vesicles with finite capacity
    - Release is triggered by neural activity and modulated by various factors
    - Reuptake mechanisms recycle neurotransmitters from the synaptic cleft
    - Synthesis pathways replenish neurotransmitter stores over time
    """
    
    def __init__(self, 
                 pool_size=100.0, 
                 baseline_level=20.0,
                 synthesis_rate=0.1,
                 reuptake_rate=0.3,
                 decay_rate=0.05,
                 release_threshold=0.2):
        """
        Initialize the Neurotransmitter Pool.
        
        Args:
            pool_size (float): Maximum capacity of the neurotransmitter pool
            baseline_level (float): Baseline level of neurotransmitter
            synthesis_rate (float): Rate at which neurotransmitter is synthesized
            reuptake_rate (float): Rate at which released neurotransmitter is recycled
            decay_rate (float): Rate at which neurotransmitter decays in the synaptic cleft
            release_threshold (float): Threshold for neurotransmitter release
        """
        super(NeurotransmitterPool, self).__init__()
        
        self.pool_size = pool_size
        self.baseline_level = baseline_level
        self.synthesis_rate = synthesis_rate
        self.reuptake_rate = reuptake_rate
        self.decay_rate = decay_rate
        self.release_threshold = release_threshold
        
        # Register buffers for persistent state
        self.register_buffer('stored_amount', torch.tensor(baseline_level))
        self.register_buffer('released_amount', torch.tensor(0.0))
        self.register_buffer('synaptic_level', torch.tensor(0.0))
        self.register_buffer('cumulative_release', torch.tensor(0.0))
        
    def release(self, release_signal):
        """
        Release neurotransmitter based on the release signal.
        
        Args:
            release_signal (torch.Tensor): Signal triggering neurotransmitter release
            
        Returns:
            torch.Tensor: Amount of neurotransmitter released
        """
        # Handle batch processing - take mean of release signal
        if release_signal.dim() > 1:
            release_signal = release_signal.mean()
            
        # Calculate release amount based on signal and available neurotransmitter
        release_probability = torch.sigmoid(release_signal - self.release_threshold)
        potential_release = self.stored_amount * release_probability
        
        # Ensure we don't release more than what's available
        actual_release = torch.min(potential_release, self.stored_amount)
        
        # Update stored and released amounts
        self.stored_amount = self.stored_amount - actual_release
        self.released_amount = actual_release
        self.cumulative_release = self.cumulative_release + actual_release
        
        # Update synaptic level
        self.synaptic_level = self.synaptic_level + actual_release
        
        return actual_release
    
    def update(self):
        """
        Update the neurotransmitter pool state, including synthesis, reuptake, and decay.
        
        Returns:
            float: Current synaptic level of neurotransmitter
        """
        # Synthesize new neurotransmitter
        synthesis_amount = self.synthesis_rate * (self.pool_size - self.stored_amount)
        self.stored_amount = self.stored_amount + synthesis_amount
        
        # Reuptake from synaptic cleft
        reuptake_amount = self.reuptake_rate * self.synaptic_level
        self.stored_amount = torch.min(self.stored_amount + reuptake_amount, torch.tensor(self.pool_size))
        
        # Decay in synaptic cleft
        decay_amount = self.decay_rate * self.synaptic_level
        self.synaptic_level = self.synaptic_level - reuptake_amount - decay_amount
        
        # Ensure non-negative values
        self.stored_amount = torch.max(self.stored_amount, torch.tensor(0.0))
        self.synaptic_level = torch.max(self.synaptic_level, torch.tensor(0.0))
        
        # Return as float for easier batch handling
        return self.synaptic_level.item()
    
    def get_synaptic_level(self):
        """Get the current synaptic level of neurotransmitter."""
        return self.synaptic_level.item()
    
    def get_stored_amount(self):
        """Get the current stored amount of neurotransmitter."""
        return self.stored_amount.item()
    
    def get_cumulative_release(self):
        """Get the cumulative amount of neurotransmitter released."""
        return self.cumulative_release.item()
    
    def reset(self):
        """Reset the neurotransmitter pool to baseline levels."""
        self.stored_amount.fill_(self.baseline_level)
        self.released_amount.fill_(0.0)
        self.synaptic_level.fill_(0.0)
        self.cumulative_release.fill_(0.0)


class DynamicSynapseModel(nn.Module):
    """
    A model of dynamic synaptic connections that adjust their strengths based on
    neurotransmitter levels and neural activity.
    
    Biological inspiration:
    - Synaptic strength is modulated by neurotransmitter levels
    - Hebbian plasticity strengthens connections between co-active neurons
    - Homeostatic scaling maintains overall network stability
    - Neuromodulators like dopamine and acetylcholine gate plasticity
    """
    
    def __init__(self, input_size, hidden_size, 
                 hebbian_lr=0.01, 
                 homeostatic_factor=0.1,
                 connection_decay=0.99):
        """
        Initialize the Dynamic Synapse Model.
        
        Args:
            input_size (int): Size of the input features
            hidden_size (int): Size of the hidden state
            hebbian_lr (float): Learning rate for Hebbian plasticity
            homeostatic_factor (float): Factor for homeostatic scaling
            connection_decay (float): Decay rate for synaptic connections
        """
        super(DynamicSynapseModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hebbian_lr = hebbian_lr
        self.homeostatic_factor = homeostatic_factor
        self.connection_decay = connection_decay
        
        # Initialize synaptic weights
        self.register_buffer('synaptic_weights', torch.ones(hidden_size, hidden_size))
        self.register_buffer('activity_history', torch.zeros(hidden_size))
        self.register_buffer('connection_history', torch.zeros(hidden_size, hidden_size))
        
        # Learnable parameters for neuromodulatory influence
        self.dopamine_influence = nn.Parameter(torch.ones(1) * 0.5)
        self.serotonin_influence = nn.Parameter(torch.ones(1) * 0.3)
        self.norepinephrine_influence = nn.Parameter(torch.ones(1) * 0.4)
        self.acetylcholine_influence = nn.Parameter(torch.ones(1) * 0.6)
        
    def forward(self, hidden, neurotransmitters=None):
        """
        Forward pass of the Dynamic Synapse Model.
        
        Args:
            hidden (torch.Tensor): Hidden state tensor of shape (batch_size, hidden_size)
            neurotransmitters (dict, optional): Dictionary of neurotransmitter levels
                
        Returns:
            torch.Tensor: Modulated hidden state
        """
        batch_size = hidden.size(0)
        
        # Update activity history
        current_activity = torch.mean(torch.abs(hidden), dim=0)
        self.activity_history = self.activity_history * self.connection_decay + current_activity * (1 - self.connection_decay)
        
        # Apply synaptic weights if batch size is 1
        # For larger batches, we use the weights as a modulation factor
        if batch_size == 1:
            modulated_hidden = torch.matmul(hidden, self.synaptic_weights)
        else:
            # Use weights as a modulation factor for each unit
            modulation = torch.mean(self.synaptic_weights, dim=1).unsqueeze(0).expand(batch_size, -1)
            modulated_hidden = hidden * modulation
        
        return modulated_hidden
    
    def update_connections(self, hidden, reward=None, neurotransmitters=None):
        """
        Update synaptic connections based on activity, reward, and neurotransmitter levels.
        
        Args:
            hidden (torch.Tensor): Hidden state tensor
            reward (torch.Tensor, optional): Reward signal
            neurotransmitters (dict, optional): Dictionary of neurotransmitter levels
                
        Returns:
            torch.Tensor: Updated synaptic weights
        """
        batch_size = hidden.size(0)
        
        # For batched inputs, use the mean activity
        if batch_size > 1:
            hidden_activity = torch.mean(hidden, dim=0)
        else:
            hidden_activity = hidden.squeeze(0)
        
        # Compute Hebbian updates (outer product of activity vectors)
        activity_outer = torch.outer(hidden_activity, hidden_activity)
        
        # Decay existing connections
        self.connection_history = self.connection_history * self.connection_decay
        
        # Apply Hebbian learning
        hebbian_update = self.hebbian_lr * activity_outer
        
        # Apply homeostatic scaling to prevent runaway dynamics
        homeostatic_scaling = self.homeostatic_factor * (1.0 - self.activity_history.unsqueeze(1))
        
        # Combine updates
        connection_update = hebbian_update * homeostatic_scaling
        
        # Apply neuromodulatory influence if provided
        if neurotransmitters is not None:
            # Extract neurotransmitter levels
            dopamine = neurotransmitters.get('dopamine', torch.tensor(0.0))
            serotonin = neurotransmitters.get('serotonin', torch.tensor(0.0))
            norepinephrine = neurotransmitters.get('norepinephrine', torch.tensor(0.0))
            acetylcholine = neurotransmitters.get('acetylcholine', torch.tensor(0.0))
            
            # Dopamine gates reward-based learning
            if reward is not None:
                dopamine_effect = self.dopamine_influence * dopamine * reward.mean()
            else:
                dopamine_effect = self.dopamine_influence * dopamine
            
            # Serotonin modulates overall plasticity
            serotonin_effect = self.serotonin_influence * serotonin
            
            # Norepinephrine enhances signal-to-noise ratio
            norepinephrine_effect = self.norepinephrine_influence * norepinephrine
            
            # Acetylcholine gates attention-based learning
            acetylcholine_effect = self.acetylcholine_influence * acetylcholine
            
            # Combine neuromodulatory effects
            neuromod_factor = (1.0 + dopamine_effect + serotonin_effect + 
                              norepinephrine_effect + acetylcholine_effect)
            
            # Apply neuromodulatory scaling
            connection_update = connection_update * neuromod_factor
        
        # Update connection history
        self.connection_history = self.connection_history + connection_update.detach()
        
        # Update synaptic weights
        self.synaptic_weights = torch.sigmoid(self.connection_history)
        
        return self.synaptic_weights
    
    def get_synaptic_weights(self):
        """Get the current synaptic weights."""
        return self.synaptic_weights
    
    def reset_connections(self):
        """Reset synaptic connections to initial state."""
        self.synaptic_weights.fill_(1.0)
        self.activity_history.fill_(0.0)
        self.connection_history.fill_(0.0)


class NeuromodulatorSystem(nn.Module):
    """
    A biologically-inspired neuromodulator system that modulates neural activity
    based on reward signals.
    
    This system models the effects of different neurotransmitters:
    - Dopamine: Reward prediction and reinforcement learning
    - Serotonin: Mood regulation and behavioral inhibition
    - Norepinephrine: Attention and arousal
    - Acetylcholine: Learning and memory
    
    The system includes:
    1. Neurotransmitter pools with finite resources
    2. Dynamic release mechanisms based on neural activity and reward
    3. Integration with the persistent GRU controller
    4. Support for reward prediction and error calculation
    """
    
    def __init__(self, input_size, hidden_size, 
                 dopamine_scale=1.0, serotonin_scale=0.8, 
                 norepinephrine_scale=0.6, acetylcholine_scale=0.7,
                 reward_decay=0.95):
        """
        Initialize the Neuromodulator System.
        
        Args:
            input_size (int): Size of the input features
            hidden_size (int): Size of the hidden state
            dopamine_scale (float): Scaling factor for dopamine effects
            serotonin_scale (float): Scaling factor for serotonin effects
            norepinephrine_scale (float): Scaling factor for norepinephrine effects
            acetylcholine_scale (float): Scaling factor for acetylcholine effects
            reward_decay (float): Decay factor for reward signals
        """
        super(NeuromodulatorSystem, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Scaling factors for different neurotransmitters
        self.dopamine_scale = dopamine_scale
        self.serotonin_scale = serotonin_scale
        self.norepinephrine_scale = norepinephrine_scale
        self.acetylcholine_scale = acetylcholine_scale
        self.reward_decay = reward_decay
        
        # Create neurotransmitter pools with different characteristics
        # Dopamine: Moderate baseline, fast reuptake, sensitive to reward
        self.dopamine_pool = NeurotransmitterPool(
            pool_size=100.0,
            baseline_level=30.0,
            synthesis_rate=0.05,
            reuptake_rate=0.4,
            decay_rate=0.1,
            release_threshold=0.3
        )
        
        # Serotonin: High baseline, slow reuptake, less sensitive to immediate reward
        self.serotonin_pool = NeurotransmitterPool(
            pool_size=150.0,
            baseline_level=50.0,
            synthesis_rate=0.03,
            reuptake_rate=0.2,
            decay_rate=0.05,
            release_threshold=0.4
        )
        
        # Norepinephrine: Low baseline, fast reuptake, sensitive to novelty/surprise
        self.norepinephrine_pool = NeurotransmitterPool(
            pool_size=80.0,
            baseline_level=20.0,
            synthesis_rate=0.08,
            reuptake_rate=0.5,
            decay_rate=0.15,
            release_threshold=0.2
        )
        
        # Acetylcholine: Moderate baseline, moderate reuptake, sensitive to attention
        self.acetylcholine_pool = NeurotransmitterPool(
            pool_size=120.0,
            baseline_level=40.0,
            synthesis_rate=0.06,
            reuptake_rate=0.3,
            decay_rate=0.08,
            release_threshold=0.25
        )
        
        # Neural networks for predicting neurotransmitter release signals
        self.dopamine_network = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        self.serotonin_network = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        self.norepinephrine_network = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        self.acetylcholine_network = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Modulation networks - transform neurotransmitter levels to modulation signals
        self.dopamine_modulation = nn.Linear(1, hidden_size)
        self.serotonin_modulation = nn.Linear(1, hidden_size)
        self.norepinephrine_modulation = nn.Linear(1, hidden_size)
        self.acetylcholine_modulation = nn.Linear(1, hidden_size)
        
        # Dynamic synapse model for adjusting connection strengths
        self.dynamic_synapse = DynamicSynapseModel(input_size, hidden_size)
        
        # Reward prediction error tracking
        self.register_buffer('predicted_reward', torch.zeros(1))
        self.register_buffer('reward_prediction_error', torch.zeros(1))
        
        # Initialize parameters
        self.init_parameters()
        
    def init_parameters(self):
        """Initialize the parameters with appropriate distributions."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'modulation' in name:
                    # Initialize modulation weights with small values
                    nn.init.normal_(param, mean=0.0, std=0.01)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, hidden, reward=None):
        """
        Forward pass of the Neuromodulator System.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            hidden (torch.Tensor): Hidden state tensor of shape (batch_size, hidden_size)
            reward (torch.Tensor, optional): Reward signal of shape (batch_size, 1)
            
        Returns:
            tuple: (modulated_hidden, neurotransmitter_levels)
        """
        batch_size = x.size(0)
        
        # Combine input and hidden state
        combined = torch.cat((x, hidden), dim=1)
        
        # Calculate reward prediction error if reward is provided
        if reward is not None:
            self.reward_prediction_error = reward.mean() - self.predicted_reward
            self.predicted_reward = self.predicted_reward * self.reward_decay + reward.mean() * (1 - self.reward_decay)
        
        # Predict neurotransmitter release signals
        dopamine_signal = self.dopamine_network(combined)
        serotonin_signal = self.serotonin_network(combined)
        norepinephrine_signal = self.norepinephrine_network(combined)
        acetylcholine_signal = self.acetylcholine_network(combined)
        
        # Modulate release signals based on reward and prediction error
        if reward is not None:
            # Dopamine release is strongly influenced by reward prediction error
            dopamine_signal = dopamine_signal + self.reward_prediction_error.abs() * 0.5
            
            # Serotonin is influenced by positive rewards
            serotonin_signal = serotonin_signal + torch.relu(reward.mean()) * 0.3
            
            # Norepinephrine is influenced by surprise (absolute prediction error)
            norepinephrine_signal = norepinephrine_signal + self.reward_prediction_error.abs() * 0.4
            
            # Acetylcholine is influenced by the presence of any reward signal
            acetylcholine_signal = acetylcholine_signal + (reward != 0).float().mean() * 0.2
        
        # Release neurotransmitters from pools
        dopamine_release = self.dopamine_pool.release(dopamine_signal)
        serotonin_release = self.serotonin_pool.release(serotonin_signal)
        norepinephrine_release = self.norepinephrine_pool.release(norepinephrine_signal)
        acetylcholine_release = self.acetylcholine_pool.release(acetylcholine_signal)
        
        # Update neurotransmitter pools
        dopamine_level = self.dopamine_pool.update()
        serotonin_level = self.serotonin_pool.update()
        norepinephrine_level = self.norepinephrine_pool.update()
        acetylcholine_level = self.acetylcholine_pool.update()
        
        # Create batch tensors for neurotransmitter levels
        dopamine_batch = torch.ones(batch_size, 1, device=hidden.device) * dopamine_level
        serotonin_batch = torch.ones(batch_size, 1, device=hidden.device) * serotonin_level
        norepinephrine_batch = torch.ones(batch_size, 1, device=hidden.device) * norepinephrine_level
        acetylcholine_batch = torch.ones(batch_size, 1, device=hidden.device) * acetylcholine_level
        
        dopamine_mod = self.dopamine_modulation(dopamine_batch) * self.dopamine_scale
        serotonin_mod = self.serotonin_modulation(serotonin_batch) * self.serotonin_scale
        norepinephrine_mod = self.norepinephrine_modulation(norepinephrine_batch) * self.norepinephrine_scale
        acetylcholine_mod = self.acetylcholine_modulation(acetylcholine_batch) * self.acetylcholine_scale
        
        # Combine modulations
        modulation = torch.sigmoid(dopamine_mod + serotonin_mod + norepinephrine_mod + acetylcholine_mod)
        
        # Apply modulation to hidden state
        modulated_hidden = hidden * modulation
        
        # Update dynamic synaptic connections
        neurotransmitter_dict = {
            'dopamine': dopamine_level,
            'serotonin': serotonin_level,
            'norepinephrine': norepinephrine_level,
            'acetylcholine': acetylcholine_level
        }
        
        self.dynamic_synapse.update_connections(hidden, reward, neurotransmitter_dict)
        
        # Apply dynamic synaptic modulation
        final_hidden = self.dynamic_synapse(modulated_hidden, neurotransmitter_dict)
        
        # Collect neurotransmitter levels for return
        neurotransmitter_levels = {
            'dopamine': dopamine_batch,
            'serotonin': serotonin_batch,
            'norepinephrine': norepinephrine_batch,
            'acetylcholine': acetylcholine_batch
        }
        
        return final_hidden, neurotransmitter_levels
    
    def get_neurotransmitter_levels(self):
        """
        Get the current neurotransmitter levels.
        
        Returns:
            dict: Dictionary of neurotransmitter levels
        """
        return {
            'dopamine': self.dopamine_pool.get_synaptic_level(),
            'serotonin': self.serotonin_pool.get_synaptic_level(),
            'norepinephrine': self.norepinephrine_pool.get_synaptic_level(),
            'acetylcholine': self.acetylcholine_pool.get_synaptic_level()
        }
    
    def get_neurotransmitter_stores(self):
        """
        Get the current neurotransmitter store levels.
        
        Returns:
            dict: Dictionary of neurotransmitter store levels
        """
        return {
            'dopamine': self.dopamine_pool.get_stored_amount(),
            'serotonin': self.serotonin_pool.get_stored_amount(),
            'norepinephrine': self.norepinephrine_pool.get_stored_amount(),
            'acetylcholine': self.acetylcholine_pool.get_stored_amount()
        }
    
    def get_cumulative_release(self):
        """
        Get the cumulative neurotransmitter release.
        
        Returns:
            dict: Dictionary of cumulative neurotransmitter release
        """
        return {
            'dopamine': self.dopamine_pool.get_cumulative_release(),
            'serotonin': self.serotonin_pool.get_cumulative_release(),
            'norepinephrine': self.norepinephrine_pool.get_cumulative_release(),
            'acetylcholine': self.acetylcholine_pool.get_cumulative_release()
        }
    
    def get_synaptic_weights(self):
        """
        Get the current synaptic weights.
        
        Returns:
            torch.Tensor: Synaptic weight matrix
        """
        return self.dynamic_synapse.get_synaptic_weights()
    
    def get_reward_prediction_error(self):
        """
        Get the current reward prediction error.
        
        Returns:
            torch.Tensor: Reward prediction error
        """
        return self.reward_prediction_error
    
    def reset(self):
        """Reset the neuromodulator system to initial state."""
        self.dopamine_pool.reset()
        self.serotonin_pool.reset()
        self.norepinephrine_pool.reset()
        self.acetylcholine_pool.reset()
        self.dynamic_synapse.reset_connections()
        self.predicted_reward.fill_(0.0)
        self.reward_prediction_error.fill_(0.0)


class EnhancedRewardPredictor(nn.Module):
    """
    Enhanced reward prediction system with internal GRU for temporal modeling
    and bidirectional feedback connections with the neuromodulator system.
    """
    
    def __init__(self, state_size, action_size, hidden_size, memory_size=64):
        super(EnhancedRewardPredictor, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        
        # Input projection
        self.input_projection = nn.Linear(state_size + action_size, hidden_size)
        
        # Internal GRU for temporal reward modeling
        self.reward_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )
        
        # Prediction layers
        self.prediction_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Neuromodulator processing
        self.neuromodulator_integration = nn.Linear(4, hidden_size)  # 4 neurotransmitters
        
        # Error signal generation
        self.error_projection = nn.Sequential(
            nn.Linear(2, hidden_size // 2),  # Predicted and actual reward
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 4)  # Feedback to 4 neurotransmitters
        )
        
        # Memory for maintaining reward history
        self.register_buffer('reward_memory', torch.zeros(memory_size))
        self.register_buffer('prediction_error_history', torch.zeros(memory_size))
    
    def forward(self, state, action, neuromodulator_levels, actual_reward=None, hidden=None):
        """
        Forward pass with recursive reward prediction and error computation.
        
        Args:
            state (torch.Tensor): Current state tensor
            action (torch.Tensor): Action tensor
            neuromodulator_levels (dict): Current levels of neuromodulators
            actual_reward (torch.Tensor, optional): Actual observed reward
            hidden (torch.Tensor, optional): Previous hidden state
            
        Returns:
            tuple: (predicted_reward, prediction_error, updated_hidden_state)
        """
        batch_size = state.size(0)
        
        # Combine state and action
        combined_input = torch.cat([state, action], dim=1)
        projected_input = self.input_projection(combined_input)
        
        # Integrate neuromodulator information
        neuromod_tensor = torch.cat([
            neuromodulator_levels['dopamine'],
            neuromodulator_levels['serotonin'],
            neuromodulator_levels['norepinephrine'],
            neuromodulator_levels['acetylcholine']
        ], dim=1)
        
        neuromod_signal = self.neuromodulator_integration(neuromod_tensor)
        
        # Combine with projected input
        gru_input = projected_input + neuromod_signal
        
        # Process through GRU
        if hidden is None:
            gru_output, new_hidden = self.reward_gru(gru_input.unsqueeze(1))
        else:
            gru_output, new_hidden = self.reward_gru(gru_input.unsqueeze(1), hidden)
        
        # Generate reward prediction
        predicted_reward = self.prediction_layer(gru_output.squeeze(1))
        
        # If actual reward is provided, compute prediction error
        prediction_error = None
        if actual_reward is not None:
            # Compute prediction error
            prediction_error = actual_reward - predicted_reward
            
            # Update error history
            self.prediction_error_history = torch.cat([
                self.prediction_error_history[1:],
                prediction_error.mean().unsqueeze(0)
            ])
            
            # Generate error feedback signal for neuromodulators
            error_input = torch.cat([predicted_reward, actual_reward], dim=1)
            neuromodulator_feedback = self.error_projection(error_input)
        
        # Update reward memory
        self.reward_memory = torch.cat([
            self.reward_memory[1:],
            predicted_reward.mean().unsqueeze(0)
        ])
        
        return predicted_reward, prediction_error, new_hidden, neuromodulator_feedback if actual_reward is not None else None