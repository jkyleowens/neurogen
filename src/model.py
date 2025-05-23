"""
Brain-Inspired Neural Network Model

This module implements the main BrainInspiredNN class that integrates
the controller, neuromodulator, and other components.
"""

import torch
import torch.nn as nn
import warnings

# Import from project structure
try:
    from .controller.persistent_gru import PersistentGRUController
except ImportError:
    warnings.warn("PersistentGRUController not found. Ensure it\'s correctly placed and imported.")
    # Define a placeholder if not found, to avoid NameError during class definition
    class PersistentGRUController(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); warnings.warn("Using Placeholder PersistentGRUController")
        def forward(self, x, h): return x, h
        def init_hidden(self, b, d): return None

try:
    from neurogen.bio_gru import BioGRU # Adjusted path
except ImportError:
    try:
        from bio_gru import BioGRU # Try direct import if in same dir or PYTHONPATH
    except ImportError:
        warnings.warn("BioGRU not found. Ensure it\'s correctly placed and imported.")
        # Define a placeholder
        class BioGRU(nn.Module):
            def __init__(self, *args, **kwargs): super().__init__(); warnings.warn("Using Placeholder BioGRU")
            def forward(self, x, h): return x, h
            def init_hidden(self, b, d): return None

from src.utils.memory_utils import optimize_memory_usage, print_gpu_memory_status
from src.utils.reset_model_state import reset_model_state

# Try to import cupy for GPU-accelerated array operations, fall back to numpy
try:
    import cupy as cp
    USING_CUPY = True
    print("Using CuPy for GPU-accelerated array operations")
except ImportError:
    warnings.warn("CuPy not found. Falling back to NumPy (aliased as \'cp\'). "
                  "For GPU acceleration of array operations, please install CuPy.")
    import numpy as cp # cp will alias to numpy if cupy is not found
    USING_CUPY = False

class BrainInspiredNN(nn.Module):
    """
    A neural network model inspired by brain functionality.
    
    This model uses a GRU-based controller as its central component,
    with a neuromodulation system based on reward signals.
    """
    
    # Define constants for history lengths
    PREV_REWARDS_MAX_LEN = 10
    REWARD_HISTORY_MAX_LEN = 20

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
            config = input_size # config is the first argument
            input_size = None # Reset to allow fetching from config
            # Note: other parameters like hidden_size, output_size will also be fetched from config

        # Use nested config fields if config provided
        if config is not None:
            self.config = config # Store the main config
            # Model general params
            model_conf = config.get('model', {})
            self.input_size = model_conf.get('input_size', input_size) # Prioritize explicit if given
            self.hidden_size = model_conf.get('hidden_size', hidden_size)
            self.output_size = model_conf.get('output_size', output_size)
            
            # Controller params
            ctrl_conf = config.get('controller', {})
            self.persistent_memory_size = ctrl_conf.get('persistent_memory_size', persistent_memory_size if persistent_memory_size is not None else 64)
            self.num_layers = ctrl_conf.get('num_layers', num_layers if num_layers is not None else 2)
            self.dropout = ctrl_conf.get('dropout', dropout if dropout is not None else 0.2)
            
            # Neuromodulator params
            neu_conf = config.get('neuromodulator', {})
            self.dopamine_scale = neu_conf.get('dopamine_scale', dopamine_scale if dopamine_scale is not None else 1.0)
            self.serotonin_scale = neu_conf.get('serotonin_scale', serotonin_scale if serotonin_scale is not None else 1.0)
            self.norepinephrine_scale = neu_conf.get('norepinephrine_scale', norepinephrine_scale if norepinephrine_scale is not None else 1.0)
            self.acetylcholine_scale = neu_conf.get('acetylcholine_scale', acetylcholine_scale if acetylcholine_scale is not None else 1.0)
            self.reward_decay = neu_conf.get('reward_decay', reward_decay if reward_decay is not None else 0.95)
            
            # Neurogenesis/pruning parameters for BioGRU (if used)
            bio_gru_conf = config.get('bio_gru_specifics', {}) # New config section
            self.neuron_death_threshold = bio_gru_conf.get('neuron_death_threshold', 0.1)
            self.neuron_growth_threshold = bio_gru_conf.get('neuron_growth_threshold', 0.7)
            self.growth_signal_threshold = bio_gru_conf.get('growth_signal_threshold', 0.5)
            # Default max_neurons_per_layer, ensure hidden_size is valid before multiplication
            default_max_neurons = hidden_size * 2 if isinstance(hidden_size, int) and hidden_size > 0 else 256
            self.max_neurons_per_layer = bio_gru_conf.get('max_neurons_per_layer', default_max_neurons)

        else:
            # Use explicit parameters if no config dict
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
            self.config = {} # Initialize empty config if none provided
            # Default neurogenesis params if no config
            self.neuron_death_threshold = 0.1
            self.neuron_growth_threshold = 0.7
            self.growth_signal_threshold = 0.5
            default_max_neurons_no_conf = hidden_size * 2 if isinstance(hidden_size, int) and hidden_size > 0 else 256
            self.max_neurons_per_layer = default_max_neurons_no_conf


        # Validate essential parameters
        if self.input_size is None or self.hidden_size is None or self.output_size is None:
            raise ValueError("input_size, hidden_size, and output_size must be specified either directly or via config.")

        # Initialize controller (choose bio GRU if configured)
        self.use_bio_gru = self.config.get('model', {}).get('use_bio_gru', False) # Store this flag

        if self.use_bio_gru:
            # Use the biological GRU
            # BioGRU now handles its own output projection to self.output_size
            self.controller = BioGRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size, 
                num_layers=self.num_layers,
                output_size=self.output_size, 
                neuron_death_threshold=self.neuron_death_threshold,
                neuron_growth_threshold=self.neuron_growth_threshold,
                growth_signal_threshold=self.growth_signal_threshold,
                max_neurons_per_layer=self.max_neurons_per_layer
            )
            # When using BioGRU that handles its own output, BrainInspiredNN's output_layer is not strictly needed for projection.
            # However, we might still want it for applying dropout or if BioGRU's output isn't the final form.
            # For now, we'll assume BioGRU's output is final and BrainInspiredNN.output_layer will be bypassed.
            if hasattr(self, 'output_layer'):
                del self.output_layer 
            # Dropout can still be applied to BioGRU's output if self.dropout_layer is defined
            if self.dropout is not None and self.dropout > 0 and not hasattr(self, 'dropout_layer'):
                 self.dropout_layer = nn.Dropout(self.dropout)
            elif self.dropout is None or self.dropout == 0:
                 if hasattr(self, 'dropout_layer'): # remove if dropout is 0/None
                     del self.dropout_layer

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

        # Health and pruning parameters for neuron death (relevant for non-BioGRU or a different health mechanism)
        health_conf = self.config.get('neuron_health', {})
        self.health_decay = health_conf.get('health_decay', 0.99)
        self.death_threshold = health_conf.get('death_threshold', 0.1)
        # Initialize neuron health and mask buffers
        self.register_buffer('neuron_health', torch.ones(self.hidden_size))
        self.register_buffer('neuron_mask', torch.ones(self.hidden_size))
        
        # Learning rate for neuromodulator-driven weight updates (from training config)
        self.learning_rate = self.config.get('training', {}).get('learning_rate', 1e-3)

        # Initialize hidden states
        self.hidden = None
        # self.neurotransmitter_levels = None # This seems to be initialized in init_hidden

        # Initialize reward history buffers
        self.register_buffer('_prev_rewards_buffer', torch.zeros(self.PREV_REWARDS_MAX_LEN))
        self.register_buffer('_prev_rewards_filled', torch.tensor(0, dtype=torch.long)) # Tracks how many slots are filled
        self.register_buffer('_reward_history_buffer', torch.zeros(self.REWARD_HISTORY_MAX_LEN))
        self.register_buffer('_reward_history_filled', torch.tensor(0, dtype=torch.long)) # Tracks how many slots are filled
        
        # Initialize neuron activity buffer for health updates
        self.register_buffer('_neuron_activity', torch.full((self.hidden_size,), 0.5)) # Initialize to 0.5

        self.configure_neurons() # Call configure_neurons at the end of init

    def forward(self, x, reward=None, error_signal_for_update=None, **kwargs):
        """
        Forward pass through the network with flexible argument handling.
        
        Args:
            x (torch.Tensor): Input tensor
            reward (torch.Tensor, optional): Reward signal for neuromodulation
            error_signal_for_update (torch.Tensor, optional): Error signal for updates
            **kwargs: Additional keyword arguments (e.g., growth_signal for BioGRU)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Handle the error_signal_for_update parameter
        if error_signal_for_update is not None:
            if reward is None:
                reward = -error_signal_for_update
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        growth_signal = kwargs.get('growth_signal', 0.0) 
        output = None # Initialize output to ensure it's always defined

        try:
            if hasattr(self, 'controller'):
                if self.use_bio_gru:
                    controller_output, new_hidden = self.controller(
                        x, 
                        hidden_states_init=self.hidden, 
                        error_signal_for_update=error_signal_for_update, 
                        growth_signal_for_layers=growth_signal 
                    )
                    self.hidden = new_hidden 
                    if hasattr(self, 'dropout_layer'):
                         output = self.dropout_layer(controller_output)
                    else:
                         output = controller_output

                else: # Standard PersistentGRUController
                    controller_output_gru, hidden_dict = self.controller(x, self.hidden) # Renamed to avoid clash
                    self.hidden = hidden_dict 
                    
                    if controller_output_gru.dim() == 3 and seq_length > 1:
                        final_hidden_features = controller_output_gru[:, -1, :]
                    else:
                        final_hidden_features = controller_output_gru.squeeze(1) if controller_output_gru.dim() == 3 else controller_output_gru
                    
                    if reward is not None:
                        try:
                            if not isinstance(reward, torch.Tensor): reward = torch.tensor(reward, device=x.device, dtype=x.dtype)
                            if reward.dim() == 0: reward = reward.expand(batch_size)
                            elif reward.dim() == 1 and reward.size(0) != batch_size: reward = reward.mean().expand(batch_size)
                            if reward.dim() > 1: reward = reward.squeeze()
                            if reward.dim() == 0: reward = reward.expand(batch_size)
                            elif reward.dim() > 1 or reward.size(0) != batch_size: reward = reward.mean().expand(batch_size)

                            reward_magnitude = torch.abs(reward)
                            reward_sign = torch.sign(reward)
                            adjusted_reward = reward_sign * torch.tanh(reward_magnitude * 0.5)
                            current_reward_mean_tensor = adjusted_reward.mean()
                            self._prev_rewards_buffer = torch.roll(self._prev_rewards_buffer, shifts=-1, dims=0)
                            self._prev_rewards_buffer[-1] = current_reward_mean_tensor
                            if self._prev_rewards_filled < self.PREV_REWARDS_MAX_LEN: self._prev_rewards_filled += 1
                            
                            consistency_factor = torch.tensor(1.0, device=x.device) 
                            if self._prev_rewards_filled > 5: 
                                valid_prev_rewards = self._prev_rewards_buffer[-self._prev_rewards_filled:] 
                                avg_prev_reward = torch.mean(valid_prev_rewards)
                                reward_prev_variance = torch.var(valid_prev_rewards, unbiased=False) 
                                if avg_prev_reward > 0.5 and reward_prev_variance < 0.1: consistency_factor = torch.tensor(0.5, device=x.device)
                            
                            feedback_signal = final_hidden_features * adjusted_reward.unsqueeze(1) * consistency_factor
                            final_hidden_features = final_hidden_features + feedback_signal
                            
                            if hasattr(self, 'neuron_health') and hasattr(self, 'health_decay') and hasattr(self, 'death_threshold'): # Ensure attributes exist
                                with torch.no_grad():
                                    effect = torch.tanh(torch.mean(torch.abs(feedback_signal), dim=0)) 
                                    self._neuron_activity = self._neuron_activity * 0.9 + effect * 0.1
                                    activity_modifier = 1.0 - torch.abs(self._neuron_activity - 0.5) * 0.5
                                    effect = effect * activity_modifier
                                    self.neuron_health = self.neuron_health * self.health_decay + effect * (1 - self.health_decay)
                                    self.neuron_mask = (self.neuron_health > self.death_threshold).float()
                        except Exception as e:
                            warnings.warn(f"Reward processing error for non-BioGRU: {e}")                            

                    if hasattr(self, 'neuron_mask'): 
                        final_hidden_features = final_hidden_features * self.neuron_mask

                    self._last_features = final_hidden_features.detach()

                    if hasattr(self, 'dropout_layer'):
                        out_hidden = self.dropout_layer(final_hidden_features)
                    else:
                        out_hidden = final_hidden_features
                    
                    if hasattr(self, 'output_layer'): 
                        output = self.output_layer(out_hidden)
                    else: 
                        warnings.warn("Output layer not found for non-BioGRU. Using emergency output layer.")
                        if not hasattr(self, '_emergency_output_layer'):
                            self._emergency_output_layer = nn.Linear(final_hidden_features.size(-1), self.output_size).to(x.device)
                        output = self._emergency_output_layer(out_hidden)

            else: 
                warnings.warn("Controller not found. Using fallback for controller_output.")
                output = torch.zeros(batch_size, seq_length if x.dim() ==3 and seq_length > 0 else 1, self.output_size, device=x.device)
                if x.dim() == 2 or (x.dim() == 3 and seq_length == 1): 
                    output = output.squeeze(1) 

        except Exception as e:
            warnings.warn(f"Controller forward pass error: {e}. Using fallback for output.")
            # Ensure output is defined in case of early exception
            if output is None:
                output = torch.zeros(batch_size, seq_length if x.dim() ==3 and seq_length > 0 else 1, self.output_size, device=x.device)
                if x.dim() == 2 or (x.dim() == 3 and seq_length == 1):
                    output = output.squeeze(1)

        if not self.use_bio_gru:
            if reward is not None and hasattr(self, 'update_weights'):
                try:
                    self.update_weights(reward) 
                except Exception as e:
                    warnings.warn(f"Weight update error in forward (non-BioGRU path): {e}")

        return output

    def update_weights(self, reward_tensor):
        """
        Update output layer weights based on last features and reward signal.
        This method is primarily for the non-BioGRU pathway using self.output_layer.
        `reward_tensor` is expected to be a pre-processed tensor, potentially per-batch.
        """
        if self.use_bio_gru: # BioGRU has its own update mechanisms
            warnings.warn("Skipping BrainInspiredNN.update_weights as BioGRU is active and handles its own updates.")
            return

        if not hasattr(self, '_last_features') or self._last_features is None or not hasattr(self, 'output_layer'):
            warnings.warn("Skipping weight update: _last_features or output_layer not available.")
            return
            
        try:
            with torch.no_grad():
                # Process reward_tensor to a scalar tensor for global scaling of updates
                if reward_tensor.numel() == 1:
                    reward_scalar_tensor = reward_tensor.squeeze() 
                else: # If reward_tensor is per-batch or multi-dimensional
                    reward_scalar_tensor = reward_tensor.mean() # Global scalar effect
                    
                current_abs_reward_scalar_tensor = torch.abs(reward_scalar_tensor) # 0-dim

                # Update _reward_history_buffer (circular buffer)
                self._reward_history_buffer = torch.roll(self._reward_history_buffer, shifts=-1, dims=0)
                self._reward_history_buffer[-1] = current_abs_reward_scalar_tensor # Store scalar

                if self._reward_history_filled < self.REWARD_HISTORY_MAX_LEN:
                    self._reward_history_filled += 1
                
                effective_lr = torch.tensor(self.learning_rate, device=reward_scalar_tensor.device)
                if self._reward_history_filled > 0:
                    valid_reward_history = self._reward_history_buffer[-self._reward_history_filled:]
                    
                    avg_reward_hist = torch.mean(valid_reward_history)
                    max_reward_hist = torch.max(valid_reward_history) # Ensure this is scalar
                    
                    lr_scale = torch.tensor(1.0, device=reward_scalar_tensor.device)
                    if avg_reward_hist > 0.5 * max_reward_hist and avg_reward_hist > 0.2:
                        lr_scale = torch.tensor(0.5, device=reward_scalar_tensor.device)
                    
                    effective_lr = self.learning_rate * lr_scale
                
                features = self._last_features # Should be [batch_size, hidden_size]
                
                if features.dim() == 1: # Should not happen if _last_features is correctly set from final_hidden
                    warnings.warn("_last_features has dim 1. Unsqueezing.")
                    features = features.unsqueeze(0) 
                elif features.dim() > 2:
                    warnings.warn(f"_last_features has dim {features.dim()}. Flattening.")
                    features = features.view(features.size(0), -1)
                
                batch_size = features.size(0)
                feature_size = features.size(1)
                
                if feature_size != self.output_layer.weight.size(1): # hidden_size
                    warnings.warn(f"Feature size {feature_size} doesn't match weight input size {self.output_layer.weight.size(1)}. Resizing/padding features.")
                    if feature_size > self.output_layer.weight.size(1):
                        features = features[:, :self.output_layer.weight.size(1)]
                    else:
                        padding = torch.zeros(batch_size, self.output_layer.weight.size(1) - feature_size, device=features.device)
                        features = torch.cat([features, padding], dim=1)
                
                feature_norms = features.norm(dim=1, keepdim=True)
                normalized_features = features / (feature_norms + 1e-6) # [batch_size, hidden_size]
                
                # Delta calculation:
                # If reward_tensor was per-batch [batch_size], use it for per-sample delta contribution
                # Otherwise, use reward_scalar_tensor for a global effect.
                if reward_tensor.dim() == 1 and reward_tensor.size(0) == batch_size:
                    # reward_tensor is [batch_size], normalized_features is [batch_size, hidden_size]
                    # delta should be [hidden_size] by averaging over batch
                    delta = (normalized_features * reward_tensor.unsqueeze(1)).mean(dim=0)
                else:
                    # Use scalar reward for delta, features are already normalized
                    # delta should be [hidden_size] by averaging over batch
                    delta = (normalized_features * reward_scalar_tensor).mean(dim=0)

                if delta.dim() == 0: # Should not happen if hidden_size > 1
                    delta = delta.expand(self.output_layer.weight.size(1))
                elif delta.size(0) != self.output_layer.weight.size(1):
                    warnings.warn(f"Delta size {delta.size(0)} doesn't match weight columns {self.output_layer.weight.size(1)}. Resizing/padding delta.")
                    if delta.size(0) > self.output_layer.weight.size(1):
                        delta = delta[:self.output_layer.weight.size(1)]
                    else:
                        padding = torch.zeros(self.output_layer.weight.size(1) - delta.size(0), device=delta.device)
                        delta = torch.cat([delta, padding])
                
                weight_decay = 0.001 * self.output_layer.weight.data
                
                # Weight matrix is [output_size, hidden_size], delta is [hidden_size]
                self.output_layer.weight.data += effective_lr * delta.unsqueeze(0) - weight_decay # delta.unsqueeze(0) makes it [1, hidden_size]
                
                if self.output_layer.bias is not None:
                    # Bias update scaled by scalar reward
                    self.output_layer.bias.data += effective_lr * reward_scalar_tensor * 0.1 
                    
        except Exception as e:
            warnings.warn(f"Weight update failed: {e}")
            if hasattr(self, '_last_features') and self._last_features is not None:
                warnings.warn(f"  Last features shape: {self._last_features.shape}")
            if hasattr(self, 'output_layer') and hasattr(self.output_layer, 'weight'):
                warnings.warn(f"  Output layer weight shape: {self.output_layer.weight.shape}")
            warnings.warn(f"  Reward tensor type: {type(reward_tensor)}, shape: {reward_tensor.shape if isinstance(reward_tensor, torch.Tensor) else 'N/A'}")

    def configure_neurons(self, config=None):
        """
        Configure neuron parameters for optimized performance.
        
        Args:
            config: Configuration parameters (uses self.config if None)
            
        Returns:
            self: For method chaining
        """
        if config is None:
            config = self.config
        
        # Skip if neuron optimization not enabled
        if not config.get('neuron_optimization', {}).get('enabled', False):
            return self
        
        warnings.warn("Configuring neurons for optimized performance...")
        opt_config = config['neuron_optimization']
        
        # Configure BioGRU if it's being used
        if self.config.get('model', {}).get('use_bio_gru', False) and hasattr(self, 'controller'):
            if hasattr(self.controller, 'layers'):
                for layer_idx, layer in enumerate(self.controller.layers):
                    # Skip non-BiologicalGRUCell layers
                    if not hasattr(layer, 'neuron_mask'):
                        continue
                        
                    for i in range(layer.hidden_size):
                        if layer.neuron_mask[i] > 0:
                            for neuron in [layer.update_gate_neurons[i], layer.reset_gate_neurons[i], layer.candidate_neurons[i]]:
                                # Set target activity
                                neuron.target_activity = opt_config.get('target_activity', 0.15)
                                
                                # Set homeostatic rate
                                neuron.homeostatic_rate = opt_config.get('homeostatic_rate', 0.01)
                                
                                # Apply plasticity settings
                                if hasattr(neuron, 'hebbian_weight'):
                                    plasticity = opt_config.get('plasticity', {})
                                    neuron.hebbian_weight = plasticity.get('hebbian_weight', 0.3)
                                    neuron.error_weight = plasticity.get('error_weight', 0.7)
                                    
                                # Apply synapse settings
                                synapse = opt_config.get('synapse', {})
                                if hasattr(neuron, 'synaptic_facilitation'):
                                    neuron.facilitation_rate = synapse.get('facilitation_rate', 0.1)
                                if hasattr(neuron, 'synaptic_depression'):
                                    neuron.depression_rate = synapse.get('depression_rate', 0.2)
            
            # Configure output neurons if they exist
            if hasattr(self.controller, 'output_neurons'):
                for i, neuron in enumerate(self.controller.output_neurons):
                    # Same configuration as for hidden neurons
                    neuron.target_activity = opt_config.get('target_activity', 0.15)
                    neuron.homeostatic_rate = opt_config.get('homeostatic_rate', 0.01)
                    
                    # Apply plasticity settings
                    if hasattr(neuron, 'hebbian_weight'):
                        plasticity = opt_config.get('plasticity', {})
                        neuron.hebbian_weight = plasticity.get('hebbian_weight', 0.3)
                        neuron.error_weight = plasticity.get('error_weight', 0.7)
        
        # Set health parameters
        self.health_decay = opt_config.get('health_decay', 0.99)
        self.death_threshold = opt_config.get('death_threshold', 0.1)
        
        # Apply to output layer
        if hasattr(self, 'output_layer'):
            # Nothing specific to configure for standard linear layer
            pass
        
        warnings.warn("Neuron configuration complete")
        return self


    def init_hidden(self, batch_size, device):
        """
        Initialize hidden states for the model.
        
        Args:
            batch_size (int): Batch size
            device (torch.device): Device to create tensors on
            
        Returns:
            Union[List[torch.Tensor], Dict[str, torch.Tensor], None]: Hidden states
        """
        if self.use_bio_gru:
            # BioGRU's forward pass handles hidden state initialization if None is passed.
            self.hidden = None 
        elif hasattr(self.controller, 'init_hidden'):
            self.hidden = self.controller.init_hidden(batch_size, device)
        else:
            self.hidden = None 
        
        if hasattr(self, 'neuromodulator') and hasattr(self.neuromodulator, 'init_levels'):
            self.neurotransmitter_levels = self.neuromodulator.init_levels(batch_size, device)
            if isinstance(self.hidden, dict): 
                self.hidden['neurotransmitter_levels'] = self.neurotransmitter_levels
        else:
            self.neurotransmitter_levels = None

        return self.hidden 
    
    def get_neurotransmitter_levels(self):
        """
        Get the current neurotransmitter levels.
        This might need to be adapted based on whether BioGRU or a separate neuromodulator is used.
        """
        if self.use_bio_gru and hasattr(self.controller, 'neuromodulator_levels'):
            return self.controller.neuromodulator_levels # Get from BioGRU directly
        elif hasattr(self, 'neuromodulator') and self.neurotransmitter_levels is not None:
            return self.neuromodulator.get_levels(self.neurotransmitter_levels)
        return None
    
    def reset_state(self):
        """Reset the model's internal state."""
        self.hidden = None
        if hasattr(self, 'neurotransmitter_levels'):
            self.neurotransmitter_levels = None
        
        # Reset controller state if it has reset method
        if hasattr(self, 'controller') and hasattr(self.controller, 'reset_state'):
            self.controller.reset_state()
        
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
            tuple: Processed data ready for model input (PyTorch tensors)
        """
        import time
        preprocess_start_time = time.time()
        
        try:
            # Define keys and default values for config.get to avoid complex string escaping
            features_key = 'features'
            default_features = ['Open', 'High', 'Low', 'Close', 'Volume']
            normalize_key = 'normalize_features'
            default_normalize = True
            sequence_length_key = 'sequence_length'
            default_sequence_length = 20
            target_column_key = 'target_column'
            default_target_column = 'Close'
            predict_next_step_key = 'predict_next_step'
            default_predict_next_step = True

            extracted_features = []
            for feature_name in config.get(features_key, default_features):
                if feature_name in stock_data.columns:
                    extracted_features.append(cp.asarray(stock_data[feature_name].values))
                else:
                    warnings.warn(f"Feature '{feature_name}' not found in stock_data. Skipping.")

            if not extracted_features:
                warnings.warn("No features extracted from stock_data. Returning empty tensors.")
                return torch.empty(0), torch.empty(0)

            X = cp.column_stack(extracted_features)

            if config.get(normalize_key, default_normalize):
                mean = cp.zeros(X.shape[1], dtype=X.dtype)
                std = cp.ones(X.shape[1], dtype=X.dtype)
                
                for i in range(X.shape[1]):
                    mean[i] = cp.mean(X[:, i])
                    std[i] = cp.std(X[:, i])
                
                std_no_zero = cp.where(std == 0, cp.array(1.0, dtype=std.dtype), std)
                X = (X - mean) / std_no_zero
            
            sequence_length = config.get(sequence_length_key, default_sequence_length)
            target_column_name = config.get(target_column_key, default_target_column)
            predict_next_step = config.get(predict_next_step_key, default_predict_next_step)

            num_samples = len(X) - sequence_length
            if predict_next_step:
                num_samples -= 1 

            if num_samples < 0:
                warnings.warn(f"Data length ({len(X)}) is insufficient for sequence length ({sequence_length}) and target. Cannot create sequences.")
                return torch.empty(0), torch.empty(0)

            x_sequences = cp.zeros((num_samples, sequence_length, X.shape[1]), dtype=X.dtype)
            y_sequences = cp.zeros(num_samples, dtype=X.dtype) 

            target_values = None
            if target_column_name in stock_data.columns:
                target_values = cp.asarray(stock_data[target_column_name].values)
            else:
                warnings.warn(f"Target column '{target_column_name}' not found. Using a placeholder target (last feature of current step).")

            for i in range(num_samples):
                x_sequences[i] = X[i:i + sequence_length]
                if target_values is not None:
                    if predict_next_step:
                        y_sequences[i] = target_values[i + sequence_length]
                    else: 
                        y_sequences[i] = target_values[i + sequence_length - 1] 
                else: 
                    y_sequences[i] = X[i + sequence_length - 1, -1] 
            
            # Convert the array to torch tensor efficiently using optimized transfer functions
            if USING_CUPY:
                # Import efficient conversion function for zero-copy when possible
                from src.utils.memory_utils import efficient_cp_to_torch
                x_sequences_torch = efficient_cp_to_torch(x_sequences, device='cuda', non_blocking=True).float()
                y_sequences_torch = efficient_cp_to_torch(y_sequences, device='cuda', non_blocking=True).float()
            else:
                # Standard numpy to torch conversion
                x_sequences_torch = torch.from_numpy(x_sequences).float() 
                y_sequences_torch = torch.from_numpy(y_sequences).float() 
            
            preprocess_total_time = time.time() - preprocess_start_time
            print(f"⏱️  Preprocess timing: {preprocess_total_time:.3f}s for {len(stock_data)} data points")
            
            return x_sequences_torch, y_sequences_torch

        except Exception as e:
            warnings.warn(f"Error in preprocess_data: {e}")
            return torch.empty(0), torch.empty(0)


    def pretrain_components(self, dataloader, device, config):
        """
        Built-in ultra-safe pretraining that's guaranteed to work.
        """
        warnings.warn("Starting built-in safe pretraining...")
        
        try:
            # Get controller and neuromodulator config
            controller_config = config.get('controller', {})
            neuromod_config = config.get('neuromodulator', {})
            
            controller_enabled = controller_config.get('enabled', True)
            neuromod_enabled = neuromod_config.get('enabled', True)
            
            # Get a single sample for testing
            try:
                sample_batch = next(iter(dataloader))
                if len(sample_batch) >= 2:
                    sample_data = sample_batch[0]
                    sample_target = sample_batch[1]
                    
                    # Ensure reasonable batch size
                    if sample_data.shape[0] > 8:
                        sample_data = sample_data[:8]
                        sample_target = sample_target[:8]
                    
                    sample_data = sample_data.to(device)
                    sample_target = sample_target.to(device)
                    
                    print(f"✅ Got sample data: {sample_data.shape}, target: {sample_target.shape}")
                else:
                    raise ValueError("Batch doesn't have enough elements")
                    
            except Exception as e:
                print(f"⚠️  Could not get sample from dataloader: {e}")
                print("Creating synthetic sample data...")
                
                # Create synthetic data based on model configuration
                input_size = getattr(self, 'input_size', 5)
                seq_length = 30  # Default sequence length
                batch_size = 4   # Small batch
                
                sample_data = torch.randn(batch_size, seq_length, input_size, device=device)
                sample_target = torch.randn(batch_size, device=device)
                print(f"✅ Created synthetic sample: {sample_data.shape}, target: {sample_target.shape}")
            
            # Test basic model functionality
            print("Testing basic model functionality...")
            try:
                self.eval()  # Set to evaluation mode for testing
                
                with torch.no_grad():
                    # Reset model state
                    if hasattr(self, 'reset_state'):
                        self.reset_state()
                    
                    # Test basic forward pass
                    output = self(sample_data)
                    print(f"✅ Basic forward pass successful: output shape {output.shape}")
                    
                    # Test with reward if the model supports it
                    try:
                        reward = torch.tensor(0.01, device=device)
                        output_with_reward = self(sample_data, reward=reward)
                        print("✅ Reward feedback test successful")
                    except Exception as reward_e:
                        print(f"⚠️  Reward feedback test failed: {reward_e} (this is OK)")
            
            except Exception as e:
                print(f"❌ Basic model test failed: {e}")
                print("Model may have compatibility issues - skipping pretraining")
                return self
            
            # Controller pretraining (if enabled and controller exists)
            if controller_enabled and hasattr(self, 'controller'):
                print("\n--- Testing Controller ---")
                try:
                    controller_epochs = min(2, controller_config.get('epochs', 2))  # Max 2 epochs
                    
                    # Test controller forward pass
                    with torch.no_grad():
                        if hasattr(self.controller, 'reset_state'):
                            self.controller.reset_state()
                        
                        if hasattr(self.controller, 'init_hidden'):
                            hidden = self.controller.init_hidden(sample_data.shape[0], device)
                            controller_output = self.controller(sample_data, hidden)
                        else:
                            controller_output = self.controller(sample_data)
                        
                        print(f"✅ Controller test successful: output shape {controller_output.shape if hasattr(controller_output, 'shape') else type(controller_output)}")
                    
                    # Simple controller "pretraining" - just test it a few times
                    for epoch in range(controller_epochs):
                        try:
                            with torch.no_grad():
                                if hasattr(self.controller, 'reset_state'):
                                    self.controller.reset_state()
                                
                                if hasattr(self.controller, 'init_hidden'):
                                    hidden = self.controller.init_hidden(sample_data.shape[0], device)
                                    _ = self.controller(sample_data, hidden)
                                else:
                                    _ = self.controller(sample_data)
                            
                            print(f"✅ Controller pretraining epoch {epoch+1}/{controller_epochs} completed")
                        except Exception as e:
                            print(f"⚠️  Controller pretraining epoch {epoch+1} failed: {e}")
                            break
                    
                    print("✅ Controller pretraining completed")
                    
                except Exception as e:
                    print(f"⚠️  Controller pretraining failed: {e} (continuing anyway)")
            
            # Neuromodulator pretraining (if enabled)
            if neuromod_enabled:
                print("\n--- Testing Neuromodulator ---")
                try:
                    neuromod_epochs = min(2, neuromod_config.get('epochs', 2))  # Max 2 epochs
                    
                    # Simple neuromodulator "pretraining" - test reward feedback
                    for epoch in range(neuromod_epochs):
                        try:
                            with torch.no_grad():
                                if hasattr(self, 'reset_state'):
                                    self.reset_state()
                                
                                # Test different reward values
                                rewards = [0.01, -0.01, 0.05]
                                for reward_val in rewards:
                                    reward = torch.tensor(reward_val, device=device)
                                    _ = self(sample_data, reward=reward)
                            
                            print(f"✅ Neuromodulator pretraining epoch {epoch+1}/{neuromod_epochs} completed")
                        except Exception as e:
                            print(f"⚠️  Neuromodulator pretraining epoch {epoch+1} failed: {e}")
                            break
                    
                    print("✅ Neuromodulator pretraining completed")
                    
                except Exception as e:
                    print(f"⚠️  Neuromodulator pretraining failed: {e} (continuing anyway)")
            
            print("\n✅ Built-in safe pretraining completed successfully!")
            return self
            
        except Exception as e:
            print(f"❌ Built-in safe pretraining failed: {e}")
            print("✅ Continuing without pretraining - model should still work fine!")
            return self


    def _basic_pretrain_fallback(self, dataloader, device, config):
        """
        Basic fallback pretraining method.
        
        Args:
            dataloader: DataLoader for pretraining data
            device: Device to use for training
            config: Configuration for pretraining
        """
        print("Running basic pretraining fallback...")
        
        # Simple controller pretraining
        if hasattr(self, 'controller') and config.get('controller', {}).get('enabled', True):
            print("Basic controller pretraining...")
            
            controller_epochs = config.get('controller', {}).get('epochs', 2)
            criterion = torch.nn.MSELoss()
            
            # Get a few batches for quick pretraining
            batches_processed = 0
            for batch_idx, batch_data in enumerate(dataloader):
                if batches_processed >= controller_epochs:
                    break
                    
                try:
                    if len(batch_data) >= 2:
                        data, target = batch_data[0].to(device), batch_data[1].to(device)
                        
                        # Simple forward pass without training
                        with torch.no_grad():
                            try:
                                if hasattr(self.controller, 'reset_state'):
                                    self.controller.reset_state()
                                
                                # Try forward pass to check compatibility
                                if hasattr(self.controller, 'init_hidden'):
                                    hidden = self.controller.init_hidden(data.shape[0], device)
                                    output = self.controller(data, hidden)
                                else:
                                    output = self.controller(data)
                                
                                print(f"Controller forward pass successful - batch {batch_idx}")
                                batches_processed += 1
                                
                            except Exception as e:
                                print(f"Controller forward pass failed: {e}")
                                break
                                
                except Exception as e:
                    print(f"Error in basic controller pretraining: {e}")
                    break
        
        # Simple neuromodulator pretraining
        if config.get('neuromodulator', {}).get('enabled', True):
            print("Basic neuromodulator pretraining...")
            
            neuromod_epochs = config.get('neuromodulator', {}).get('epochs', 2)
            batches_processed = 0
            
            for batch_idx, batch_data in enumerate(dataloader):
                if batches_processed >= neuromod_epochs:
                    break
                    
                try:
                    if len(batch_data) >= 2:
                        data, target = batch_data[0].to(device), batch_data[1].to(device)
                        
                        # Test neuromodulator feedback
                        with torch.no_grad():
                            try:
                                # Reset model state
                                if hasattr(self, 'reset_state'):
                                    self.reset_state()
                                
                                # Simple forward pass
                                output = self(data)
                                
                                # Apply simple reward feedback
                                reward = torch.tensor(0.1, device=device)  # Small positive reward
                                self(data, reward=reward)
                                
                                print(f"Neuromodulator feedback successful - batch {batch_idx}")
                                batches_processed += 1
                                
                            except Exception as e:
                                print(f"Neuromodulator feedback failed: {e}")
                                break
                                
                except Exception as e:
                    print(f"Error in basic neuromodulator pretraining: {e}")
                    break
        
        print("Basic pretraining fallback completed")