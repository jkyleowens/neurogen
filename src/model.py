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
from src.neuromodulator.reward_modulator import RewardModulator
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

        # Initialize controller (GRU-based)
        self.controller = PersistentGRUController(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            persistent_memory_size=self.persistent_memory_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        # Initialize neuromodulator
        self.neuromodulator = RewardModulator(
            hidden_size=self.hidden_size,
            dopamine_scale=self.dopamine_scale,
            serotonin_scale=self.serotonin_scale,
            norepinephrine_scale=self.norepinephrine_scale,
            acetylcholine_scale=self.acetylcholine_scale,
            reward_decay=self.reward_decay
        )
        
        # Output layer
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        # Dropout for regularization
        self.dropout_layer = nn.Dropout(self.dropout)
        # Batch normalization layer
        self.batch_norm = nn.BatchNorm1d(self.hidden_size)
        
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
        # Apply controller and retain hidden state
        controller_output, hidden_dict = self.controller(x, self.hidden)
        self.hidden = hidden_dict

        # Apply batch normalization
        if controller_output.dim() == 3:
            # Reshape for batch normalization (batch, hidden, seq)
            controller_output = controller_output.permute(0, 2, 1)
            controller_output = self.batch_norm(controller_output)
            controller_output = controller_output.permute(0, 2, 1)
        else:
            controller_output = self.batch_norm(controller_output)

        # Apply neuromodulation if reward is provided
        if reward is not None:
            controller_output, self.neurotransmitter_levels = self.neuromodulator(
                controller_output, reward, self.neurotransmitter_levels
            )
        
        # If output is sequence (batch, seq, hidden), take last time step
        if controller_output.dim() == 3:
            final_hidden = controller_output[:, -1, :]
        else:
            final_hidden = controller_output
        
        # Apply dropout then project to output
        out_hidden = self.dropout_layer(final_hidden)
        output = self.output_layer(out_hidden)
        return output
    
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
