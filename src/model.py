"""
Brain-Inspired Neural Network Model

This module implements the main BrainInspiredNN class that integrates
the controller, neuromodulator, and other components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import from project structure
from src.controller.persistent_gru import PersistentGRUController
from bio_gru import BioGRU  # Biological GRU implementation
from src.utils.memory_utils import optimize_memory_usage, print_gpu_memory_status
from src.utils.reset_model_state import reset_model_state

class BrainInspiredNN(nn.Module):
    """
    A neural network model inspired by brain functionality.
    
    This model uses a GRU-based controller as its central component,
    with a neuromodulation system based on reward signals.
    """
    
    def __init__(self, 
                 input_size=None,
                 hidden_size=None,
                 output_size=None,
                 persistent_memory_size=None,
                 num_layers=None,
                 dropout=None,
                 dopamine_scale=None,
                 serotonin_scale=None,
                 norepinephrine_scale=None,
                 acetylcholine_scale=None,
                 reward_decay=None,
                 config=None):
        """Initialize the Brain-Inspired Neural Network."""
        super(BrainInspiredNN, self).__init__()
        # Allow passing config dict as first positional argument
        if config is None and isinstance(input_size, dict):
            config = input_size
        # Use nested config fields if config provided
        if config is not None:
            self.config = config
            # Model params
            model_conf = config.get('model', {})
            self.input_size = model_conf.get('input_size', 10)
            self.hidden_size = model_conf.get('hidden_size', 128)
            self.output_size = model_conf.get('output_size', 1)
            # Controller params
            ctrl_conf = config.get('controller', {})
            self.persistent_memory_size = ctrl_conf.get('persistent_memory_size', 64)
            self.num_layers = ctrl_conf.get('num_layers', 2)
            self.dropout = ctrl_conf.get('dropout', 0.2)
            # Neuromodulator params
            neu_conf = config.get('neuromodulator', {})
            self.dopamine_scale = neu_conf.get('dopamine_scale', 1.0)
            self.serotonin_scale = neu_conf.get('serotonin_scale', 1.0)
            self.norepinephrine_scale = neu_conf.get('norepinephrine_scale', 1.0)
            self.acetylcholine_scale = neu_conf.get('acetylcholine_scale', 1.0)
            self.reward_decay = neu_conf.get('reward_decay', 0.95)
        else:
            # Use explicit parameters
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.persistent_memory_size = persistent_memory_size
            self.num_layers = num_layers
            self.dropout = dropout
            self.dopamine_scale = dopamine_scale
            self.serotonin_scale = serotonin_scale
            self.norepinephrine_scale = norepinephrine_scale
            self.acetylcholine_scale = acetylcholine_scale
            self.reward_decay = reward_decay
            self.config = config or {}

        # Initialize controller (choose bio GRU if configured)
        if self.config.get('model', {}).get('use_bio_gru', False):
            # Use the biological GRU
            self.controller = BioGRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=self.output_size
            )
        else:
            # Use standard persistent GRU
            self.controller = PersistentGRUController(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                persistent_memory_size=self.persistent_memory_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            )
        
        # Output projection and regularization
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        self.dropout_layer = nn.Dropout(self.dropout)

        # Health and pruning parameters for neuron death
        health_conf = config.get('neuron_health', {}) if config else {}
        self.health_decay = health_conf.get('health_decay', 0.99)
        self.death_threshold = health_conf.get('death_threshold', 0.1)
        # Initialize neuron health and mask buffers
        self.register_buffer('neuron_health', torch.ones(self.hidden_size))
        self.register_buffer('neuron_mask', torch.ones(self.hidden_size))
        
        # Learning rate for neuromodulator-driven weight updates (from training config)
        self.learning_rate = self.config.get('training', {}).get('learning_rate', 1e-3)

        # Initialize hidden states
        self.hidden = None
        self.neurotransmitter_levels = None
        
    def forward(self, x, reward=None):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            reward (torch.Tensor, optional): Reward signal for neuromodulation
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Modify forward pass to support feedback-driven learning
        # Apply controller (biological or persistent) and retain hidden state
        if self.config.get('model', {}).get('use_bio_gru', False):
            # BioGRU returns (output_seq, hidden_states)
            controller_output, new_hidden = self.controller(x, self.hidden)
            self.hidden = new_hidden
        else:
            # PersistentGRUController returns (output_seq, hidden_dict)
            controller_output, hidden_dict = self.controller(x, self.hidden)
            self.hidden = hidden_dict

        # If output is sequence (batch, seq, hidden), take last time step
        if controller_output.dim() == 3:
            final_hidden = controller_output[:, -1, :]
        else:
            final_hidden = controller_output

        # Enhanced neuromodulator-driven feedback adjustment with anti-overfitting mechanisms
        if reward is not None:
            # Calculate reward absolute magnitude and sign separately
            reward_magnitude = torch.abs(reward)
            reward_sign = torch.sign(reward)
            
            # Apply non-linear transformation to prevent excessive feedback on large rewards
            # This helps prevent the model from overcommitting to spurious large rewards
            adjusted_reward = reward_sign * torch.tanh(reward_magnitude * 0.5)
            
            # Dynamic feedback strength that reduces with continued high rewards (anti-overfitting)
            # This prevents the model from becoming too specialized/confident
            if hasattr(self, '_prev_rewards'):
                # Track recent reward history
                self._prev_rewards.append(adjusted_reward.item())
                if len(self._prev_rewards) > 10:  # Keep history limited
                    self._prev_rewards.pop(0)
                
                # Calculate average recent reward and variance
                avg_reward = sum(self._prev_rewards) / len(self._prev_rewards)
                reward_variance = sum((r - avg_reward)**2 for r in self._prev_rewards) / len(self._prev_rewards)
                
                # Reduce feedback strength when rewards are consistently high with low variance
                # (indicating potential overfitting to a specific pattern)
                consistency_factor = 1.0
                if avg_reward > 0.5 and reward_variance < 0.1:
                    consistency_factor = 0.5  # Reduce feedback to prevent overfitting
                
                feedback_signal = final_hidden * adjusted_reward * consistency_factor
            else:
                # First reward, initialize history
                self._prev_rewards = [adjusted_reward.item()]
                feedback_signal = final_hidden * adjusted_reward
            
            # Apply feedback signal
            final_hidden = final_hidden + feedback_signal

        # Update neuron health based on feedback magnitude with improved balance
        if reward is not None:
            with torch.no_grad():
                # Calculate effect but with diminishing returns for very large feedback
                effect = torch.tanh(torch.mean(torch.abs(feedback_signal), dim=0))
                
                # Introduce homeostasis - neurons that are consistently active get less health boost
                if hasattr(self, '_neuron_activity'):
                    # Update activity tracking (exponential moving average)
                    self._neuron_activity = 0.9 * self._neuron_activity + 0.1 * (final_hidden > 0.5).float()
                    
                    # Neurons with balanced activity (not too high or low) get health boost
                    # This encourages diversity in neural population
                    activity_modifier = 1.0 - torch.abs(self._neuron_activity - 0.5) * 0.5
                    effect = effect * activity_modifier
                else:
                    # First update, initialize tracking
                    self._neuron_activity = 0.1 * (final_hidden > 0.5).float()
                
                # Update health with more balanced decay
                self.neuron_health = self.neuron_health * self.health_decay + effect * (1 - self.health_decay)
                
                # Update mask: neurons below threshold die
                self.neuron_mask = (self.neuron_health > self.death_threshold).float()

        # Apply neuron mask to prune dead neurons
        final_hidden = final_hidden * self.neuron_mask

        # Store features for potential weight updates
        self._last_features = final_hidden.detach()

        # Apply dropout then project to output
        out_hidden = self.dropout_layer(final_hidden)
        output = self.output_layer(out_hidden)

        # Optionally update weights if reward provided
        if reward is not None:
            self.update_weights(reward)
        return output

    def update_weights(self, reward):
        """
        Update output layer weights based on last features and reward signal
        with anti-overfitting mechanisms.
        """
        with torch.no_grad():
            # Process reward for more stable learning
            if isinstance(reward, (int, float)):
                reward_value = reward
            else:
                # If tensor, get scalar value
                reward_value = reward.item() if reward.numel() == 1 else reward.mean().item()
                
            # Apply adaptive learning rate based on reward history
            if not hasattr(self, '_reward_history'):
                self._reward_history = [abs(reward_value)]
                effective_lr = self.learning_rate
            else:
                # Track reward history for adaptive learning
                self._reward_history.append(abs(reward_value))
                if len(self._reward_history) > 20:
                    self._reward_history.pop(0)
                
                # Calculate reward statistics
                avg_reward = sum(self._reward_history) / len(self._reward_history)
                max_reward = max(self._reward_history)
                
                # Reduce learning rate when rewards are consistently high (preventing overfitting)
                lr_scale = 1.0
                if avg_reward > 0.5 * max_reward and avg_reward > 0.2:
                    # Rewards are consistently high, reduce learning rate to prevent overfitting
                    lr_scale = 0.5
                
                effective_lr = self.learning_rate * lr_scale
            
            # Calculate update with feature normalization to prevent domination by active neurons
            normalized_features = self._last_features / (self._last_features.norm(dim=1, keepdim=True) + 1e-6)
            delta = (normalized_features * reward).mean(dim=0)
            
            # Apply L2 regularization (weight decay) to prevent overfitting
            weight_decay = 0.001 * self.output_layer.weight.data
            
            # Update weights with regularization
            self.output_layer.weight.data += effective_lr * delta.unsqueeze(0) - weight_decay
            self.output_layer.bias.data += effective_lr * reward_value * 0.1  # Smaller updates for bias

    def init_hidden(self, batch_size, device):
        """
        Initialize hidden states for the model.
        
        Args:
            batch_size (int): Batch size
            device (torch.device): Device to create tensors on
            
        Returns:
            dict: Dictionary of initialized hidden states
        """
        self.hidden = self.controller.init_hidden(batch_size, device)
        self.neurotransmitter_levels = self.neuromodulator.init_levels(batch_size, device)
        return {'hidden': self.hidden, 'neurotransmitter_levels': self.neurotransmitter_levels}
    
    def get_neurotransmitter_levels(self):
        """
        Get the current neurotransmitter levels.
        
        Returns:
            dict: Dictionary of neurotransmitter levels
        """
        if self.neurotransmitter_levels is None:
            return None
        return self.neuromodulator.get_levels(self.neurotransmitter_levels)
    
    def reset_state(self):
        """
        Reset the model's internal state.
        
        Returns:
            self: The model instance with reset state
        """
        self.hidden = None
        self.neurotransmitter_levels = None
        return self
    
    @staticmethod
    def setup_model(config, input_shape):
        """
        Set up a model instance based on configuration.
        """
        input_size = input_shape[-1] if len(input_shape) > 1 else input_shape[0]
        
        # Get model parameters from config
        model_config = config.get('model', {})
        hidden_size = model_config.get('hidden_size', 128)
        # Default output_size to input_size (feature count) if not explicitly set
        output_size = model_config.get('output_size', input_size)
        
        # Get controller parameters
        controller_config = config.get('controller', {})
        persistent_memory_size = controller_config.get('persistent_memory_size', 64)
        num_layers = controller_config.get('num_layers', 2)
        dropout = controller_config.get('dropout', 0.2)
        
        # Get neuromodulator parameters
        neuromod_config = config.get('neuromodulator', {})
        dopamine_scale = neuromod_config.get('dopamine_scale', 1.0)
        serotonin_scale = neuromod_config.get('serotonin_scale', 1.0)
        norepinephrine_scale = neuromod_config.get('norepinephrine_scale', 1.0)
        acetylcholine_scale = neuromod_config.get('acetylcholine_scale', 1.0)
        reward_decay = neuromod_config.get('reward_decay', 0.95)
        
        model = BrainInspiredNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            persistent_memory_size=persistent_memory_size,
            num_layers=num_layers,
            dropout=dropout,
            dopamine_scale=dopamine_scale,
            serotonin_scale=serotonin_scale,
            norepinephrine_scale=norepinephrine_scale,
            acetylcholine_scale=acetylcholine_scale,
            reward_decay=reward_decay
        )
        
        # Optimize memory usage if specified in config
        if config.get('optimize_memory', False):
            model = optimize_memory_usage(model)
            
        return model
    
    @staticmethod
    def preprocess_data(stock_data, config):
        """
        Preprocess stock data for the model.
        
        Args:
            stock_data (pandas.DataFrame): Raw stock data
            config (dict): Configuration parameters
            
        Returns:
            tuple: Processed data ready for model input
        """
        try:
            # Extract features
            features = []
            for feature in config.get('features', ['Open', 'High', 'Low', 'Close', 'Volume']):
                if feature in stock_data.columns:
                    features.append(stock_data[feature].values)
            
            # Stack features
            X = np.column_stack(features)
            
            # Normalize data
            if config.get('normalize', True):
                for i in range(X.shape[1]):
                    mean = np.mean(X[:, i])
                    std = np.std(X[:, i])
                    if std > 0:
                        X[:, i] = (X[:, i] - mean) / std
            
            # Create target variable (next day's close price)
            y = stock_data['Close'].shift(-1).values[:-1]
            X = X[:-1]  # Remove last row to match y length
            
            return X, y
        except Exception as e:
            print(f"Error in data preprocessing: {e}")
            return None, None
