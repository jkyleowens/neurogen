"""
Fixed BioGRU Implementation - Resolving Matrix Shape Mismatch Errors

Key fixes:
1. Added missing layer normalization modules in BiologicalGRUCell
2. Fixed tensor dimension handling in BioNeuron forward pass
3. Added missing _manage_neuron_lifecycle method
4. Improved error handling and shape validation
5. Fixed weight initialization and application
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import cupy as cp
    USING_CUPY = True
except ImportError:
    import numpy as cp
    USING_CUPY = False
from collections import deque
import math

class IonChannelSystem(nn.Module):
    """
    Simulates ion channel dynamics for neural activation.
    """
    def __init__(self, input_size):
        super().__init__()
        
        # Channel conductances (learnable)
        self.na_conductance = nn.Parameter(torch.ones(input_size) * 0.5)
        self.k_conductance = nn.Parameter(torch.ones(input_size) * 0.5)
        self.ca_conductance = nn.Parameter(torch.ones(input_size) * 0.2)
        
        # Channel states
        self.register_buffer('na_state', torch.zeros(input_size))
        self.register_buffer('k_state', torch.zeros(input_size))
        self.register_buffer('ca_state', torch.zeros(input_size))
        
        # Time constants
        self.na_tau = 0.1
        self.k_tau = 0.5
        self.ca_tau = 1.0
        
        # Reversal potentials
        self.na_reversal = 1.0
        self.k_reversal = -0.8
        self.ca_reversal = 1.0
        
        # Membrane state
        self.register_buffer('membrane_potential', torch.zeros(input_size))
        self.register_buffer('resting_potential', torch.ones(input_size) * -0.7)
        self.register_buffer('calcium_concentration', torch.zeros(input_size))
        
    def reset_states(self):
        """Reset all channel and membrane states"""
        self.na_state.zero_()
        self.k_state.zero_()
        self.ca_state.zero_()
        self.membrane_potential = self.resting_potential.clone()
        self.calcium_concentration.zero_()
    
    def forward(self, x, previous_activity=None, dt=0.1):
        """Simulate ion channel dynamics and calculate membrane potential."""
        batch_size = x.size(0)
        
        # Ensure input is 2D
        if x.dim() > 2:
            x = x.view(batch_size, -1)
        
        # Handle single output case
        if x.size(1) == 1:
            x = x.expand(-1, self.na_conductance.size(0))
        elif x.size(1) != self.na_conductance.size(0):
            # Project input to match conductance size
            x = F.linear(x, torch.ones(self.na_conductance.size(0), x.size(1), device=x.device) / x.size(1))
        
        # Expand membrane state for batch processing
        if self.membrane_potential.dim() == 1:
            membrane_potential = self.membrane_potential.unsqueeze(0).expand(batch_size, -1)
            na_state = self.na_state.unsqueeze(0).expand(batch_size, -1)
            k_state = self.k_state.unsqueeze(0).expand(batch_size, -1)
            ca_state = self.ca_state.unsqueeze(0).expand(batch_size, -1)
            calcium_concentration = self.calcium_concentration.unsqueeze(0).expand(batch_size, -1)
        else:
            membrane_potential = self.membrane_potential
            na_state = self.na_state
            k_state = self.k_state
            ca_state = self.ca_state
            calcium_concentration = self.calcium_concentration
        
        # Channel activation
        na_activation = torch.sigmoid((membrane_potential - (-0.5)) * 10)
        k_activation = torch.sigmoid((membrane_potential - 0.0) * 5)
        ca_activation = torch.sigmoid((membrane_potential - 0.2) * 8)
        
        # Update channel states
        na_state = na_state + dt * (na_activation - na_state) / self.na_tau
        k_state = k_state + dt * (k_activation - k_state) / self.k_tau
        ca_state = ca_state + dt * (ca_activation - ca_state) / self.ca_tau
        
        # Calculate currents
        na_current = self.na_conductance * na_state * (self.na_reversal - membrane_potential)
        k_current = self.k_conductance * k_state * (self.k_reversal - membrane_potential)
        ca_current = self.ca_conductance * ca_state * (self.ca_reversal - membrane_potential)
        
        # Input and leak currents
        input_current = x
        leak_current = 0.05 * (self.resting_potential - membrane_potential)
        
        # Adaptation current
        adaptation_current = torch.zeros_like(membrane_potential)
        if previous_activity is not None:
            adaptation_conductance = 0.1
            adaptation_reversal = -1.0
            adaptation_current = adaptation_conductance * previous_activity * (adaptation_reversal - membrane_potential)
        
        # Update membrane potential
        total_current = na_current + k_current + ca_current + leak_current + input_current + adaptation_current
        membrane_potential = membrane_potential + dt * total_current
        
        # Update calcium concentration
        calcium_influx = F.relu(ca_state * ca_current)
        calcium_decay = 0.1 * calcium_concentration
        calcium_concentration = calcium_concentration + dt * (calcium_influx - calcium_decay)
        
        # Store updated states (take mean across batch for single-valued buffers)
        self.membrane_potential = membrane_potential.mean(0)
        self.na_state = na_state.mean(0)
        self.k_state = k_state.mean(0)
        self.ca_state = ca_state.mean(0)
        self.calcium_concentration = calcium_concentration.mean(0)
        
        return membrane_potential.mean(1, keepdim=True)  # Return single output per batch item
    
    def get_calcium_level(self):
        """Return current calcium concentration"""
        return self.calcium_concentration.mean()
    
    def get_channel_states(self):
        """Return current state of all ion channels"""
        return {
            'na': self.na_state,
            'k': self.k_state,
            'ca': self.ca_state
        }


class BioNeuron(nn.Module):
    """
    Fixed biologically-inspired neuron with proper dimension handling.
    """
    def __init__(self, input_size, activation='adaptive', initial_health=1.0, health_decay_rate=0.001, activity_target=0.15): # Added health_decay_rate and activity_target
        super().__init__()
        
        self.input_size = input_size
        
        # Fixed: Single output neuron with proper weight dimensions
        self.weights = nn.Parameter(torch.randn(1, input_size) * 0.01)  # [1, input_size]
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Ion channel system (single output)
        self.ion_channels = IonChannelSystem(1)
        
        # Health and utility tracking
        self.register_buffer('health', torch.tensor(initial_health)) # Use initial_health
        self.health_decay_rate = health_decay_rate # Store decay rate
        self.activity_target = activity_target # Store activity target

        self.register_buffer('utility', torch.zeros(1))
        self.register_buffer('activation_history', torch.zeros(100)) # Consider making size configurable
        self.register_buffer('activation_count', torch.zeros(1))
        
        # Activation function
        self.activation_type = activation
        
        # STP (Short Term Plasticity) parameters - fixed dimensions
        self.register_buffer('synaptic_facilitation', torch.ones(input_size))
        self.register_buffer('synaptic_depression', torch.ones(input_size))
        
        # Homeostatic parameters
        self.target_activity = 0.1
        self.homeostatic_rate = 0.01
        
        # Storage for learning
        self.input_history = deque(maxlen=100)
        self.activation_records = deque(maxlen=100)
        self.error_history = deque(maxlen=100)
        
    def reset_state(self):
        """Reset neuron state"""
        self.ion_channels.reset_states()
        self.activation_history = torch.zeros_like(self.activation_history)
        self.activation_count = torch.zeros_like(self.activation_count)
        self.synaptic_facilitation = torch.ones_like(self.synaptic_facilitation)
        self.synaptic_depression = torch.ones_like(self.synaptic_depression)
    
    def forward(self, x, previous_activity=None):
        """
        Forward pass with proper dimension handling.
        """
        # Ensure input is 2D [batch_size, features]
        if x.dim() > 2:
            original_batch_size = x.size(0)
            # More careful reshaping to avoid potential errors
            # For 3D input [batch, seq, features], take the last timestep
            # For 4D or higher, flatten all dimensions after the first
            if x.dim() == 3:
                x = x[:, -1, :]  # Take the last sequence element
            else:
                x = x.reshape(original_batch_size, -1)  # Flatten all dimensions after batch
        elif x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        
        batch_size = x.size(0)
        
        # Handle input size mismatch
        if x.size(1) != self.input_size:
            if x.size(1) > self.input_size:
                # Truncate extra features
                x = x[:, :self.input_size]
            else:
                # Pad with zeros
                padding = torch.zeros(batch_size, self.input_size - x.size(1), device=x.device)
                x = torch.cat([x, padding], dim=1)
        
        # Store input for plasticity
        if self.training:
            self.input_history.append(x.detach().mean(0))
        
        # Apply short-term plasticity
        effective_weights = self.weights * self.synaptic_facilitation.unsqueeze(0) * self.synaptic_depression.unsqueeze(0)
        
        # Calculate weighted input: [batch_size, input_size] @ [input_size, 1] -> [batch_size, 1]
        weighted_input = F.linear(x, effective_weights, self.bias)
        
        # Pass through ion channels
        membrane_potential = self.ion_channels(weighted_input, previous_activity)
        
        # Apply activation function
        if self.activation_type == 'adaptive':
            ca_level = self.ion_channels.get_calcium_level()
            threshold = 0.2 + 0.3 * ca_level
            activation = torch.sigmoid((membrane_potential - threshold) * 5)
        else:
            activation = torch.sigmoid(membrane_potential)
        
        # Update activation history
        if self.training:
            act_val = activation.detach().mean().item()
            
            # Update history
            new_val = torch.tensor([act_val], device=self.activation_history.device, dtype=self.activation_history.dtype) # Ensure dtype match
            self.activation_history = torch.cat([self.activation_history[1:], new_val])
            self.activation_count += (act_val > self.activity_target) # Compare with activity_target
            
            # Store for learning
            self.activation_records.append(activation.detach().mean())
        
        # Update short-term plasticity
        if self.training:
            mean_act = activation.detach().mean()
            decay_factor = 0.95
            growth_factor = 0.05
            
            self.synaptic_facilitation = (self.synaptic_facilitation * decay_factor + 
                                        growth_factor * mean_act)
            self.synaptic_depression = (self.synaptic_depression * 0.98 + 
                                      0.02 * (1 - mean_act))
        
        return activation  # [batch_size, 1]
    
    def update_health(self, reward=None, activity_threshold=0.05): # activity_threshold seems unused, consider removing or using self.activity_target
        """Update neuron health based on utility and activity pattern."""
        recent_activity = self.activation_history[-20:].mean() # Use a configurable window size
        
        # Penalize neurons with very low or extremely high activity relative to target
        # More sensitive penalty for being far from target_activity
        activity_deviation = torch.abs(recent_activity - self.activity_target)
        # Health impact is scaled by how far off target the activity is.
        # A small deviation has less impact than a large one.
        # The 5.0 multiplier makes it more sensitive.
        activity_penalty = torch.exp(-activity_deviation * 5.0) # Closer to 1 is better
        
        # Reward stable activity patterns
        activity_variability = self.activation_history[-20:].std()
        stability_factor = torch.exp(-activity_variability * 10.0) # Higher value for more stability, increased sensitivity

        # Base health change on activity and stability
        base_health_change = 0.5 * activity_penalty + 0.5 * stability_factor

        # Incorporate reward signal if provided
        if reward is not None:
            reward_tensor = torch.tensor(reward, device=self.health.device, dtype=torch.float32) if not isinstance(reward, torch.Tensor) else reward.float()
            # Reward influences the direction and magnitude of health change
            # Sigmoid maps reward to a -0.5 to 0.5 range (approx for reward around 0)
            # This means reward can boost or reduce health gain/loss
            reward_impact = (torch.sigmoid(reward_tensor * 2.0) - 0.5) # Scaled reward impact
            health_change_factor = base_health_change * (1.0 + reward_impact) # Reward modulates the base change
        else:
            health_change_factor = base_health_change

        # Apply a decay factor and the calculated change factor
        # health_decay_rate determines how quickly health naturally degrades if not supported by activity/reward
        # (1.0 - health_change_factor) determines the positive impact; if health_change_factor is high, less decay.
        # The difference from 0.5 scales the change: (health_change_factor - 0.5)
        # If health_change_factor > 0.5, it's a positive impact, else negative.
        # The 0.01 is a learning rate for health changes.
        effective_change = (health_change_factor - 0.7) * 0.05 # Shifted so that ~0.7 factor means neutral change, scaled impact

        self.health = self.health * (1.0 - self.health_decay_rate) + effective_change
        
        # Update health with bounds
        self.health = torch.clamp(self.health, 0.0, 1.0)
        
        # Update utility (simple example: health * recent_activity)
        self.utility = self.health * recent_activity
        
        return self.health
    
    def update_weights(self, error_signal, learning_rate=0.01, regularization=0.0001):
        """Update weights using biologically-plausible local learning rules."""
        if len(self.input_history) < 10 or len(self.activation_records) < 10:
            return 0.0
        
        try:
            # Calculate average recent input and activation
            recent_inputs = torch.stack(list(self.input_history)[-10:]).mean(0)
            recent_activation = torch.stack(list(self.activation_records)[-10:]).mean()
            
            # Ensure error_signal is a scalar
            if hasattr(error_signal, 'mean'):
                error_magnitude = torch.abs(error_signal.mean().detach())
                error_sign = torch.sign(error_signal.mean().detach())
            else:
                error_magnitude = torch.abs(torch.tensor(error_signal))
                error_sign = torch.sign(torch.tensor(error_signal))
            
            # Hebbian update
            hebbian_update = recent_activation * recent_inputs
            
            # Error-driven component
            adaptive_lr = learning_rate * torch.sigmoid(error_magnitude * 3)
            error_update = -error_sign * recent_inputs * error_magnitude
            
            # Combine updates
            hebbian_weight = torch.clamp(1.0 - error_magnitude, 0.2, 0.8)
            error_weight = 1.0 - hebbian_weight
            
            total_update = (hebbian_update * hebbian_weight + error_update * error_weight).unsqueeze(0)
            
            # Apply regularization
            regularization_term = regularization * self.weights.data
            
            # Update weights
            self.weights.data += adaptive_lr * total_update - regularization_term
            
            # Homeostatic bias adjustment
            activity_diff = self.target_activity - recent_activation
            bias_update_rate = self.homeostatic_rate * (1.0 + error_magnitude)
            self.bias.data += bias_update_rate * activity_diff
            
            # Weight normalization
            self.normalize_synapses()
            
            return torch.abs(total_update).mean().item()
            
        except Exception as e:
            print(f"Error in weight update: {e}")
            return 0.0
    
    def normalize_synapses(self, method="l2"):
        """Normalize synapse weights."""
        if method == "l2":
            norm = torch.norm(self.weights, p=2)
            if norm > 1.0:
                self.weights.data = self.weights.data / norm
        elif method == "l1":
            norm = torch.sum(torch.abs(self.weights))
            if norm > self.input_size * 0.5:
                self.weights.data = self.weights.data / (norm / (self.input_size * 0.5))
        elif method == "max":
            max_val = torch.max(torch.abs(self.weights))
            if max_val > 1.0:
                self.weights.data = self.weights.data / max_val


class BiologicalGRUCell(nn.Module):
    """
    Fixed GRU cell with proper layer normalization and dimension handling.
    """
    def __init__(self, input_size, hidden_size, neuron_death_threshold=0.1, neuron_growth_threshold=0.7, growth_signal_threshold=0.5, max_neurons=None): # Added growth params
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size # This is now the initial/max hidden size
        self.neuron_death_threshold = neuron_death_threshold
        self.neuron_growth_threshold = neuron_growth_threshold # Health threshold for a neuron to be considered for seeding new neuron
        self.growth_signal_threshold = growth_signal_threshold # External signal (e.g. error, novelty) to trigger growth
        self.max_neurons = max_neurons if max_neurons is not None else hidden_size * 2 # Allow growth up to double the initial size, can be configured

        # Neuron health and activity configuration
        self.neuron_config = {
            'initial_health': 1.0,
            'health_decay_rate': 0.001,
            'activity_target': 0.15
        }

        # CRITICAL FIX: Add missing layer normalization layers
        combined_size = input_size + hidden_size
        self.ln_xh_update = nn.LayerNorm(combined_size)
        self.ln_xh_reset = nn.LayerNorm(combined_size)
        self.ln_x_candidate = nn.LayerNorm(input_size)
        self.ln_h_candidate = nn.LayerNorm(1)  # For single hidden component
        self.ln_h_new = nn.LayerNorm(hidden_size)
        
        # Create biological neurons for each gate
        self.update_gate_neurons = nn.ModuleList([
            BioNeuron(combined_size, **self.neuron_config) for _ in range(self.hidden_size)
        ])
        
        self.reset_gate_neurons = nn.ModuleList([
            BioNeuron(combined_size, **self.neuron_config) for _ in range(self.hidden_size)
        ])
        
        self.candidate_neurons = nn.ModuleList([
            BioNeuron(input_size + 1, **self.neuron_config) for _ in range(self.hidden_size)  # input + single reset component
        ])
        
        # Neuron masks
        self.register_buffer('neuron_mask', torch.ones(self.max_neurons)) # Mask for max_neurons
        self.current_num_neurons = hidden_size # Track active neurons

        # Initialize mask for initial neurons
        self.neuron_mask[:self.current_num_neurons] = 1.0
        if self.current_num_neurons < self.max_neurons:
            self.neuron_mask[self.current_num_neurons:] = 0.0


        # Growth probability
        self.register_buffer('connection_strength', torch.zeros(3, hidden_size, combined_size))
    
    def reset_parameters(self):
        """Reset all neuron states"""
        for neuron in self.update_gate_neurons + self.reset_gate_neurons + self.candidate_neurons:
            neuron.reset_state()
    
    def _manage_neuron_lifecycle(self, growth_signal=0.0): # Added growth_signal
        """Manage neuron death and regeneration."""
        # Pruning: Check for dead neurons (health below threshold)
        num_pruned = 0
        for i in range(self.current_num_neurons): # Iterate only over currently active/potential neurons
            if self.neuron_mask[i] > 0:  # Only check living neurons
                # Calculate average health across gate neurons
                # Ensure neurons exist at index i before accessing health
                update_health = self.update_gate_neurons[i].health.item() if i < len(self.update_gate_neurons) else 0
                reset_health = self.reset_gate_neurons[i].health.item() if i < len(self.reset_gate_neurons) else 0
                candidate_health = self.candidate_neurons[i].health.item() if i < len(self.candidate_neurons) else 0
                
                avg_health = (update_health + reset_health + candidate_health) / 3.0
                
                if avg_health < self.neuron_death_threshold:
                    self.neuron_mask[i] = 0.0 # Mark as dead
                    # No need to shrink ModuleLists, just mask out
                    num_pruned +=1
                    # print(f"Neuron {i} pruned (health: {avg_health:.4f})")
        
        if num_pruned > 0:
            print(f"Pruned {num_pruned} neurons in a cycle.")

        # Neurogenesis: Neuron regeneration/growth
        # Trigger growth if an external signal is strong enough (e.g. high error, novelty)
        # and we have space for new neurons.
        if growth_signal > self.growth_signal_threshold and self.current_num_neurons < self.max_neurons:
            num_grown = 0
            # Try to grow a few neurons, e.g., up to 5% of current num_neurons or at least 1
            potential_growth_count = max(1, int(self.current_num_neurons * 0.05))
            
            for _ in range(potential_growth_count):
                if self.current_num_neurons >= self.max_neurons:
                    break # Stop if max capacity reached

                # Find a slot for a new neuron
                dead_indices = torch.where(self.neuron_mask == 0.0)[0]
                if len(dead_indices) == 0: # Should not happen if current_num_neurons < max_neurons
                    # This case implies neuron_mask might not be sized to max_neurons, or an error in logic.
                    # For safety, if no 'dead' slots are found but we are below max_neurons,
                    # it means we need to find the first available slot *up to max_neurons*.
                    # This typically means appending to the conceptual list of neurons.
                    if self.current_num_neurons < self.max_neurons:
                       new_neuron_idx = self.current_num_neurons # Grow at the end of current live neurons
                    else: # Should not be reached if initial check holds
                       break 
                else:
                    # Prefer to reuse a slot from a previously pruned neuron if available
                    # This logic assumes dead_indices can point to slots beyond current_num_neurons if they were pruned from a larger set
                    # Or, if we are growing into new capacity, it will be an index >= self.current_num_neurons

                    # We want the first available slot.
                    new_neuron_idx = dead_indices[0].item() if dead_indices[0] < self.max_neurons else self.current_num_neurons

                if new_neuron_idx >= self.max_neurons: # Boundary check
                    break

                self.neuron_mask[new_neuron_idx] = 1.0 # Activate the mask for the new/reused slot
                
                # Determine the actual index for ModuleLists. If new_neuron_idx is beyond current list size, we append.
                # Otherwise, we replace the (previously masked) neuron at that index.
                list_idx_to_use = new_neuron_idx

                # Initialize the new neuron's components
                new_update_neuron = BioNeuron(self.update_gate_neurons[0].input_size, **self.neuron_config).to(self.neuron_mask.device)
                new_reset_neuron = BioNeuron(self.reset_gate_neurons[0].input_size, **self.neuron_config).to(self.neuron_mask.device)
                new_candidate_neuron = BioNeuron(self.candidate_neurons[0].input_size, **self.neuron_config).to(self.neuron_mask.device)

                # If list_idx_to_use is for an existing slot (reusing pruned neuron's slot)
                if list_idx_to_use < len(self.update_gate_neurons):
                    self.update_gate_neurons[list_idx_to_use] = new_update_neuron
                    self.reset_gate_neurons[list_idx_to_use] = new_reset_neuron
                    self.candidate_neurons[list_idx_to_use] = new_candidate_neuron
                else: # Append new neuron if growing beyond current ModuleList physical size
                      # This assumes ModuleLists are not pre-allocated to max_neurons
                    self.update_gate_neurons.append(new_update_neuron)
                    self.reset_gate_neurons.append(new_reset_neuron)
                    self.candidate_neurons.append(new_candidate_neuron)
                
                # If we grew into a new slot (not reusing a pruned one within current_num_neurons range)
                if new_neuron_idx == self.current_num_neurons:
                    self.current_num_neurons += 1
                
                num_grown += 1
                # print(f"Neuron at index {new_neuron_idx} (list idx: {list_idx_to_use}) regenerated/grown. Current neurons: {self.current_num_neurons}")

            if num_grown > 0:
                print(f"Grew {num_grown} new neurons. Total active neurons: {self.current_num_neurons}")
                # Ensure hidden state tensors are resized if needed for new neurons
                # This part is tricky as hidden states are passed around.
                # For now, the BioGRU layer will need to handle hidden state compatibility.
                # The mask self.neuron_mask should be used to handle outputs correctly.
    
    def forward(self, x, h=None, error_signal=None, neuromodulators=None, growth_signal=None): # Added growth_signal
        """Forward pass with fixed dimension handling."""
        current_device = x.device
        batch_size = x.size(0)
        
        # Handle 3D input
        if x.dim() > 2:
            x = x[:, -1] if x.size(1) > 0 else x.squeeze(1)
        
        # Handle input dimension mismatch
        if x.size(1) != self.input_size:
            if x.size(1) < self.input_size:
                # Input smaller than expected, pad with zeros
                padding = torch.zeros(batch_size, self.input_size - x.size(1), device=current_device)
                x = torch.cat([x, padding], dim=1)
            else:
                # Input larger than expected, truncate
                x = x[:, :self.input_size]
            
        # Initialize hidden state
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=current_device)
        else:
            h = h.to(current_device)
            
            # Handle hidden state dimension mismatch
            if h.size(1) != self.hidden_size:
                if h.size(1) < self.hidden_size:
                    # Hidden state smaller than expected, pad with zeros
                    h_padding = torch.zeros(batch_size, self.hidden_size - h.size(1), device=current_device)
                    h = torch.cat([h, h_padding], dim=1)
                else:
                    # Hidden state larger than expected, truncate
                    h = h[:, :self.hidden_size]
        
        # Ensure neuron mask is on correct device
        self.neuron_mask = self.neuron_mask.to(current_device)
        active_mask = self.neuron_mask[:self.current_num_neurons].to(current_device)
        masked_h = h * active_mask # Apply mask to hidden state
        
        # Combine input and hidden state
        xh = torch.cat([x, masked_h], dim=1)
        
        # Apply layer normalization
        xh_update_norm = self.ln_xh_update(xh)
        xh_reset_norm = self.ln_xh_reset(xh)
        
        # Initialize outputs
        update_gate_out = torch.zeros(batch_size, self.hidden_size, device=current_device)
        reset_gate_out = torch.zeros(batch_size, self.hidden_size, device=current_device)
        candidate_out = torch.zeros(batch_size, self.hidden_size, device=current_device)
        
        # Apply neuromodulators if provided
        if neuromodulators is not None:
            for i in range(self.hidden_size):
                if self.neuron_mask[i] > 0:
                    for neuron_group in [self.update_gate_neurons, self.reset_gate_neurons, self.candidate_neurons]:
                        if hasattr(neuron_group[i], 'process_neuromodulator_signals'):
                            neuron_group[i].process_neuromodulator_signals(neuromodulators)
        
        # Process each neuron up to current_num_neurons
        for i in range(self.current_num_neurons): # Iterate up to current_num_neurons
            if active_mask[i] > 0: # Check active_mask
                # Update and reset gates
                update_gate_out[:, i:i+1] = self.update_gate_neurons[i](xh_update_norm)
                reset_gate_out[:, i:i+1] = self.reset_gate_neurons[i](xh_reset_norm)
                
                # Candidate state with reset applied
                reset_h_component = reset_gate_out[:, i:i+1] * masked_h[:, i:i+1]
                candidate_input = torch.cat([x, reset_h_component], dim=1)
                candidate_out[:, i:i+1] = self.candidate_neurons[i](candidate_input)
        
        # Compute new hidden state
        h_new = (1 - update_gate_out) * masked_h + update_gate_out * candidate_out
        h_new = self.ln_h_new(h_new) # Apply LayerNorm
        h_new = h_new * active_mask # Apply mask again to ensure output is zero for inactive neurons
        
        # Update neurons during training
        if self.training:
            # Pass growth_signal to lifecycle management
            self._manage_neuron_lifecycle(growth_signal=growth_signal if growth_signal is not None else 0.0)
            if error_signal is not None: # error_signal is used for weight updates
                self._update_neurons(xh_update_norm, error_signal) # Pass xh_update_norm as inputs for Hebbian
            
        
        return h_new
    
    def _update_neurons(self, inputs, error_signal, learning_rate=0.01): # Added inputs for Hebbian
        """Update all neurons with learning and health tracking."""
        with torch.no_grad():
            active_mask = self.neuron_mask[:self.current_num_neurons].to(error_signal.device)
            for i in range(self.current_num_neurons): # Iterate up to current_num_neurons
                if active_mask[i] > 0: # Check active_mask
                    # Update weights - BioNeuron.update_weights now uses its internal input_history
                    self.update_gate_neurons[i].update_weights(error_signal, learning_rate)
                    self.reset_gate_neurons[i].update_weights(error_signal, learning_rate)
                    self.candidate_neurons[i].update_weights(error_signal, learning_rate)
                    
                    # Update health
                    # Reward for health update can be derived from error_signal or a separate reward
                    # Using negative error as reward: lower error = higher reward
                    reward_for_health = -error_signal.mean() if hasattr(error_signal, 'mean') else -error_signal
                    self.update_gate_neurons[i].update_health(reward_for_health)
                    self.reset_gate_neurons[i].update_health(reward_for_health)
                    self.candidate_neurons[i].update_health(reward_for_health)

class BioGRU(nn.Module):
    """
    Fixed multi-layer Biological GRU.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1, neuron_death_threshold=0.1, neuron_growth_threshold=0.7, growth_signal_threshold=0.5, max_neurons_per_layer=None): # Added neurogenesis params
        super().__init__()
        
        self.input_size = input_size
        self.initial_hidden_size = hidden_size # Keep track of initial size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # If max_neurons_per_layer is not specified, default to double initial_hidden_size
        self.max_neurons_per_layer = max_neurons_per_layer if max_neurons_per_layer is not None else hidden_size * 2
        if self.max_neurons_per_layer <= 0:
            raise ValueError(
                f"max_neurons_per_layer must be positive, got {self.max_neurons_per_layer}. "
                f"This may be due to a non-positive 'hidden_size' ({hidden_size}) in config, "
                f"or max_neurons_per_layer being explicitly set to a non-positive value."
            )

        # Create layers with neurogenesis/pruning parameters
        self.layers = nn.ModuleList()
        current_layer_input_size = self.input_size
        for _ in range(num_layers):
            cell = BiologicalGRUCell(
                current_layer_input_size, # Input size from previous layer's max neuron count (or model input)
                hidden_size, # Initial hidden size for the cell
                neuron_death_threshold=neuron_death_threshold,
                neuron_growth_threshold=neuron_growth_threshold,
                growth_signal_threshold=growth_signal_threshold,
                max_neurons=self.max_neurons_per_layer
            )
            self.layers.append(cell)
            # Input to the next layer is the max output dimension of the current layer
            current_layer_input_size = self.max_neurons_per_layer
        
        # Output projection - input size needs to be dynamic or max_neurons from last layer
        # For simplicity, let's assume output neurons take input from max_neurons_per_layer of the last GRU layer
        # Or, more robustly, from the current_num_neurons of the last layer.
        # This requires careful handling in forward pass if current_num_neurons changes.
        # For now, let's make BioNeuron handle variable input size by padding/truncating.
        self.output_neurons = nn.ModuleList([
            BioNeuron(self.max_neurons_per_layer) for _ in range(output_size) # Input size is max possible from GRU layer
        ])
        
        # Neuromodulator system
        self.neuromodulator_levels = {
            'dopamine': 0.5,
            'serotonin': 0.5,
            'norepinephrine': 0.5,
            'acetylcholine': 0.5
        }
        
        # Activity records
        self.register_buffer('pathway_activity', torch.zeros(num_layers, self.max_neurons_per_layer))
        
        # Hidden state storage
        self.hidden_states = None
    
    def reset_state(self):
        """Reset all internal states"""
        self.hidden_states = None
        for layer in self.layers:
            layer.reset_parameters()
        for neuron in self.output_neurons:
            neuron.reset_state()
    
    def forward(self, x, hidden_states_init=None, error_signal_for_update=None, growth_signal_for_layers=None): # Added growth_signal_for_layers
        """Forward pass with proper error handling."""
        original_dim = x.dim()
        current_device = x.device
        
        # Ensure 3D input [batch, seq, features]
        if original_dim == 2:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden states. Size should be based on max_neurons_per_layer for buffer, but masked by current_num_neurons.
        if hidden_states_init is None:
            if (not hasattr(self, 'hidden_states') or self.hidden_states is None):
                # Initialize hidden states to the max possible size for each layer's buffer
                self.hidden_states = [torch.zeros(batch_size, self.max_neurons_per_layer, device=current_device)
                                    for _ in range(self.num_layers)]
            current_h_states = self.hidden_states
        else:
            # Handle both stacked tensor format [num_layers, batch_size, hidden_size] 
            # and list format [list of tensors with shape [batch_size, hidden_size]]
            hidden_states_list = []
            
            if isinstance(hidden_states_init, torch.Tensor) and hidden_states_init.dim() == 3:
                # Convert stacked tensor to list of tensors
                for i in range(hidden_states_init.size(0)):
                    hidden_states_list.append(hidden_states_init[i])
            elif isinstance(hidden_states_init, list):
                hidden_states_list = hidden_states_init
            else:
                raise ValueError(f"Invalid hidden_states_init format: {type(hidden_states_init)}. "
                                f"Expected list of tensors or stacked tensor with shape [num_layers, batch_size, hidden_size]")
            
            # Ensure provided hidden states are padded/truncated to max_neurons_per_layer if necessary
            current_h_states = []
            for i, h_init in enumerate(hidden_states_list):
                if i >= self.num_layers:
                    break
                
                h_init_dev = h_init.to(current_device)
                if h_init_dev.size(1) < self.max_neurons_per_layer:
                    padding = torch.zeros(batch_size, self.max_neurons_per_layer - h_init_dev.size(1), device=current_device)
                    current_h_states.append(torch.cat([h_init_dev, padding], dim=1))
                elif h_init_dev.size(1) > self.max_neurons_per_layer:
                    current_h_states.append(h_init_dev[:, :self.max_neurons_per_layer])
                else:
                    current_h_states.append(h_init_dev)
                    
            # Handle the case where we have fewer hidden states than layers
            while len(current_h_states) < self.num_layers:
                current_h_states.append(torch.zeros(batch_size, self.max_neurons_per_layer, device=current_device))
        
        # Process sequence
        output_sequence = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            layer_input = x_t
            new_h_for_step = []
            
            for i, layer in enumerate(self.layers):
                h_layer_prev = current_h_states[i]
                
                # Pass only the relevant part of hidden state for the layer's current_num_neurons
                # However, the GRUCell itself expects a hidden state buffer of its own hidden_size (which is its current_num_neurons or initial size)
                # The BiologicalGRUCell's forward method will use its internal neuron_mask.
                # The hidden state passed (h_layer_prev) should match the cell's expected full buffer size (max_neurons for that cell).
                # The output of the previous layer (layer_input) is already masked by its active neurons.
                # The current layer's GRUCell input_size should be compatible.
                # This might require adjusting the input_size of BioNeurons within the GRUCell if previous layer shrinks/grows.
                # For now, BioNeuron handles input size mismatches.

                current_growth_signal = 0.0
                if growth_signal_for_layers is not None:
                    if isinstance(growth_signal_for_layers, list) and i < len(growth_signal_for_layers):
                        current_growth_signal = growth_signal_for_layers[i]
                    elif isinstance(growth_signal_for_layers, (float, int)):
                        current_growth_signal = growth_signal_for_layers


                h_layer_new = layer(layer_input, h_layer_prev,
                                  error_signal=error_signal_for_update,
                                  neuromodulators=self.neuromodulator_levels,
                                  growth_signal=current_growth_signal) # Pass growth signal
                
                new_h_for_step.append(h_layer_new) # h_layer_new is [batch_size, max_neurons_for_cell]
                # The output of the layer (layer_input for the next layer) is h_layer_new.
                # It's already masked internally by the cell's neuron_mask.
                layer_input = h_layer_new 
            
            current_h_states = new_h_for_step # Store the full hidden states [batch_size, max_neurons_per_layer]
            
            # Update pathway activity
            if self.training:
                if (not hasattr(self, 'pathway_activity') or 
                    self.pathway_activity.device != current_device or
                    self.pathway_activity.shape[1] != self.max_neurons_per_layer):
                    self.pathway_activity = torch.zeros(self.num_layers, self.max_neurons_per_layer, 
                                                       device=current_device)
                
                for i_layer in range(self.num_layers):
                    layer_activity = current_h_states[i_layer].detach().abs().mean(0)
                    # Ensure shape compatibility
                    activity_size = min(layer_activity.shape[0], self.pathway_activity.shape[1])
                    # Exponential moving average for activity tracking using masked update
                    self.pathway_activity[i_layer, :activity_size] = (
                        0.9 * self.pathway_activity[i_layer, :activity_size] +
                        0.1 * layer_activity[:activity_size]
                    )
            
            output_sequence.append(current_h_states[-1])
        
        self.hidden_states = current_h_states # Save the last step's hidden states
        # The 'outputs' variable here is the output from the last GRU layer.
        # It will be of shape [batch_size, seq_len, max_neurons_per_layer] but masked.
        outputs = torch.stack(output_sequence, dim=1) 
        
        # Apply output projection
        # The input to output_neurons is the hidden state of the last GRU layer.
        # We take the state from the last time step.
        final_outputs_list = [] # Renamed to avoid conflict
        
        # Get the last time step output from the last layer GRU
        last_layer_output_t = outputs[:, -1, :]  # [batch_size, max_neurons_per_layer]
        
        # Handle potential dimension issues with the last layer output
        if last_layer_output_t.size(1) > self.max_neurons_per_layer:
            # Truncate if somehow larger than expected (shouldn't happen)
            last_layer_output_t = last_layer_output_t[:, :self.max_neurons_per_layer]
        elif last_layer_output_t.size(1) < self.max_neurons_per_layer:
            # Pad with zeros if smaller than expected
            padding = torch.zeros(
                last_layer_output_t.size(0), 
                self.max_neurons_per_layer - last_layer_output_t.size(1), 
                device=last_layer_output_t.device
            )
            last_layer_output_t = torch.cat([last_layer_output_t, padding], dim=1)
            
        # Process through output neurons
        for i, neuron in enumerate(self.output_neurons):
            # Now the input is guaranteed to match neuron.input_size
            neuron_output_single_step = neuron(last_layer_output_t) # [batch_size, 1]
            final_outputs_list.append(neuron_output_single_step)
        
        # Concatenate outputs from all output neurons
        final_output_cat = torch.cat(final_outputs_list, dim=1) # [batch_size, output_size]
        
        # If original input was 3D (sequence), we might want sequential output.
        # The current setup gives one output vector per sequence.
        # If sequential output is needed, output_neurons should process `outputs` [batch, seq, features]
        # For now, assuming one output vector per input sequence based on last hidden state.
        if original_dim == 3:
            if seq_len > 1:
                # This implies we want to provide an output for each item in the sequence.
                # The current output_neurons process only the last hidden state.
                # To provide per-sequence-item output, output_neurons would need to process `outputs`
                # For simplicity, if seq_len > 1, we replicate the final_output_cat across seq_len.
                # This is a placeholder; a more proper way would be to apply output_neurons per time step.
                final_output_expanded = final_output_cat.unsqueeze(1).expand(-1, seq_len, -1)
            else:
                # For sequences of length 1, maintain 3D shape but don't expand
                final_output_expanded = final_output_cat.unsqueeze(1)  # [batch_size, 1, output_size]
        else:
            final_output_expanded = final_output_cat # Already [batch_size, output_size]

        # Return to original dimensions if needed (e.g. if original_dim == 2, squeeze seq dim)
        if original_dim == 2:
            final_output_to_return = final_output_expanded.squeeze(1) # [batch_size, output_size]
        else:
            final_output_to_return = final_output_expanded # [batch_size, seq_len, output_size]

        # Stack hidden states for consistent format [num_layers, batch_size, max_neurons_per_layer]
        stacked_hidden_states = torch.stack(self.hidden_states, dim=0)
        
        return final_output_to_return, stacked_hidden_states