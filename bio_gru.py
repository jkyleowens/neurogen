import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import math

class IonChannelSystem(nn.Module):
    """
    Simulates ion channel dynamics for neural activation.
    
    In biological neurons, ion channels control the flow of ions across the cell membrane,
    which determines the neuron's electrical properties and activation. This model
    simulates Na+, K+, and Ca2+ channels with their respective dynamics.
    """
    def __init__(self, input_size):
        super().__init__()
        
        # Channel conductances (learnable)
        self.na_conductance = nn.Parameter(torch.ones(input_size) * 0.5)  # Sodium channels (fast activation)
        self.k_conductance = nn.Parameter(torch.ones(input_size) * 0.5)   # Potassium channels (delayed rectifier)
        self.ca_conductance = nn.Parameter(torch.ones(input_size) * 0.2)  # Calcium channels (learning modulation)
        
        # Channel states
        self.register_buffer('na_state', torch.zeros(input_size))
        self.register_buffer('k_state', torch.zeros(input_size))
        self.register_buffer('ca_state', torch.zeros(input_size))
        
        # Time constants for channel dynamics
        self.na_tau = 0.1  # Fast
        self.k_tau = 0.5   # Medium
        self.ca_tau = 1.0  # Slow
        
        # Reversal potentials
        self.na_reversal = 1.0
        self.k_reversal = -0.8
        self.ca_reversal = 1.0
        
        # Membrane state
        self.register_buffer('membrane_potential', torch.zeros(input_size))
        self.register_buffer('resting_potential', torch.ones(input_size) * -0.7)
        
        # Calcium concentration (affects plasticity)
        self.register_buffer('calcium_concentration', torch.zeros(input_size))
        
    def reset_states(self):
        """Reset all channel and membrane states"""
        self.na_state.zero_()
        self.k_state.zero_()
        self.ca_state.zero_()
        self.membrane_potential = self.resting_potential.clone()
        self.calcium_concentration.zero_()
    
    def forward(self, x, previous_activity=None, dt=0.1):
        """
        Simulate ion channel dynamics and calculate membrane potential.
        
        Args:
            x: Input current
            previous_activity: Previous neuron activity (for adaptation)
            dt: Time step for simulation
            
        Returns:
            Membrane potential after channel dynamics
        """
        batch_size = x.size(0)
        
        # Expand membrane state for batch processing if needed
        if self.membrane_potential.dim() == 1:
            self.membrane_potential = self.membrane_potential.unsqueeze(0).expand(batch_size, -1)
            self.na_state = self.na_state.unsqueeze(0).expand(batch_size, -1)
            self.k_state = self.k_state.unsqueeze(0).expand(batch_size, -1)
            self.ca_state = self.ca_state.unsqueeze(0).expand(batch_size, -1)
            self.calcium_concentration = self.calcium_concentration.unsqueeze(0).expand(batch_size, -1)
        
        # Channel activation variables (gating)
        na_activation = torch.sigmoid((self.membrane_potential - (-0.5)) * 10)  # Na activates when membrane depolarizes
        k_activation = torch.sigmoid((self.membrane_potential - 0.0) * 5)      # K activates more slowly
        ca_activation = torch.sigmoid((self.membrane_potential - 0.2) * 8)     # Ca activates at higher depolarization
        
        # Update channel states with time dynamics
        self.na_state = self.na_state + dt * (na_activation - self.na_state) / self.na_tau
        self.k_state = self.k_state + dt * (k_activation - self.k_state) / self.k_tau
        self.ca_state = self.ca_state + dt * (ca_activation - self.ca_state) / self.ca_tau
        
        # Calculate ion currents
        na_current = self.na_conductance * self.na_state * (self.na_reversal - self.membrane_potential)
        k_current = self.k_conductance * self.k_state * (self.k_reversal - self.membrane_potential)
        ca_current = self.ca_conductance * self.ca_state * (self.ca_reversal - self.membrane_potential)
        
        # Input current
        input_current = x
        
        # Calculate leak current (return to resting potential)
        leak_conductance = 0.05
        leak_current = leak_conductance * (self.resting_potential - self.membrane_potential)
        
        # Adaptation current (if previous activity provided)
        adaptation_current = torch.zeros_like(self.membrane_potential)
        if previous_activity is not None:
            adaptation_conductance = 0.1
            adaptation_reversal = -1.0
            adaptation_current = adaptation_conductance * previous_activity * (adaptation_reversal - self.membrane_potential)
        
        # Update membrane potential
        total_current = na_current + k_current + ca_current + leak_current + input_current + adaptation_current
        self.membrane_potential = self.membrane_potential + dt * total_current
        
        # Update calcium concentration (for learning)
        calcium_influx = F.relu(self.ca_state * ca_current)
        calcium_decay = 0.1 * self.calcium_concentration
        self.calcium_concentration = self.calcium_concentration + dt * (calcium_influx - calcium_decay)
        
        return self.membrane_potential
    
    def get_calcium_level(self):
        """Return current calcium concentration (important for plasticity)"""
        return self.calcium_concentration
    
    def get_channel_states(self):
        """Return current state of all ion channels"""
        return {
            'na': self.na_state,
            'k': self.k_state,
            'ca': self.ca_state
        }


class BioNeuron(nn.Module):
    """
    A biologically-inspired neuron with ion channels, local learning,
    synaptic plasticity, and health monitoring.
    """
    def __init__(self, input_size, activation='adaptive'):
        super().__init__()
        
        # Synaptic weights
        self.weights = nn.Parameter(torch.randn(input_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Ion channel system
        self.ion_channels = IonChannelSystem(1)  # Single output
        
        # Neuron health and utility tracking
        self.register_buffer('health', torch.ones(1))
        self.register_buffer('utility', torch.zeros(1))
        self.register_buffer('activation_history', torch.zeros(100))  # Recent activations
        self.register_buffer('activation_count', torch.zeros(1))
        
        # Local gradient estimation
        self.local_gradient_estimator = nn.Sequential(
            nn.Linear(input_size + 3, 32),  # inputs + activation + error + calcium
            nn.ReLU(),
            nn.Linear(32, input_size)
        )
        
        # Activation function
        self.activation_type = activation
        
        # Synaptic plasticity records
        self.input_history = deque(maxlen=100)
        self.activation_records = deque(maxlen=100)
        self.error_history = deque(maxlen=100)
        
        # STP (Short Term Plasticity) parameters
        self.register_buffer('synaptic_facilitation', torch.ones(input_size))
        self.register_buffer('synaptic_depression', torch.ones(input_size))
        
        # Homeostatic parameters
        self.target_activity = 0.1  # Target average activity
        self.homeostatic_rate = 0.01
        
    def reset_state(self):
        """Reset neuron state"""
        self.ion_channels.reset_states()
        self.activation_history = torch.zeros_like(self.activation_history)
        self.activation_count = torch.zeros_like(self.activation_count)
        self.synaptic_facilitation = torch.ones_like(self.synaptic_facilitation)
        self.synaptic_depression = torch.ones_like(self.synaptic_depression)
    
    def forward(self, x, previous_activity=None):
        """
        Forward pass with biophysical simulation.
        
        Args:
            x: Input signal (can be 2D [batch, features] or 3D [batch, seq, features])
            previous_activity: Previous activation state
            
        Returns:
            Neuron activation
        """
        # Handle 3D inputs by reshaping
        original_shape = x.shape
        if x.dim() > 2:
            # Flatten the first dimensions to create a 2D tensor
            x = x.reshape(-1, x.size(-1))
        
        # Store input for plasticity
        if self.training:
            self.input_history.append(x.detach().mean(0))
        
        # Apply short-term plasticity to inputs
        effective_weights = self.weights * self.synaptic_facilitation * self.synaptic_depression
        
        # Calculate weighted input - force correct dimensions
        if effective_weights.dim() > 1:
            # Already multi-dimensional
            weighted_input = F.linear(x, effective_weights, self.bias)
        else:
            # Make it 2D [1, features]
            weighted_input = F.linear(x, effective_weights.unsqueeze(0), self.bias)
        
        # Pass through ion channels to get membrane potential
        membrane_potential = self.ion_channels(weighted_input, previous_activity)
        
        # Apply activation function
        if self.activation_type == 'adaptive':
            # Adaptive threshold based on calcium
            ca_level = self.ion_channels.get_calcium_level()
            threshold = 0.2 + 0.3 * ca_level
            activation = torch.sigmoid((membrane_potential - threshold) * 5)
        else:
            # Standard activation
            activation = torch.sigmoid(membrane_potential)
        
        # Update activation history
        if self.training:
            # Handle different input dimensions
            if activation.dim() > 1:
                act_mean = activation.detach().mean(0)
                act_val = activation.detach().mean().item()
            else:
                act_mean = activation.detach()
                act_val = activation.detach().mean().item()
                
            self.activation_records.append(act_mean)
            
            # Ensure consistent dimensions for concatenation
            if self.activation_history.dim() == 1:
                # Shift activation history and add new activation as a scalar
                if act_mean.dim() == 0:
                    new_val = act_mean.unsqueeze(0)
                else:
                    new_val = act_mean.mean().unsqueeze(0)
                self.activation_history = torch.cat([self.activation_history[1:], new_val])
            else:
                # Handle multi-dimensional case
                if act_mean.dim() == 0:
                    new_val = act_mean.reshape(1, 1)
                else:
                    new_val = act_mean.reshape(1, -1)
                self.activation_history = torch.cat([self.activation_history[1:], new_val])
                
            self.activation_count += (act_val > 0.1)
        
        # Update short-term plasticity
        if self.training:
            # Synaptic facilitation (increases with activity)
            mean_act = activation.detach().mean(0) if activation.dim() > 0 else activation.detach()
            self.synaptic_facilitation = self.synaptic_facilitation * 0.95 + 0.05 * mean_act.unsqueeze(-1)
            # Synaptic depression (decreases with activity)
            self.synaptic_depression = self.synaptic_depression * 0.98 + 0.02 * (1 - mean_act.unsqueeze(-1))
        
        # Restore original shape if input was 3D
        if 'original_shape' in locals() and len(original_shape) > 2 and activation.dim() < len(original_shape):
            # Calculate the new shape based on the original input dimensions
            new_shape = list(original_shape[:-1]) + [-1]
            activation = activation.reshape(new_shape)
            
        return activation
    
    def update_health(self, reward=None, activity_threshold=0.05):
        """
        Update neuron health based on utility and activity pattern.
        """
        recent_activity = self.activation_history[-20:].mean()
        
        # Penalize neurons with very low or extremely high activity
        balanced_activity = 1.0 - abs(recent_activity - self.target_activity) * 2
        
        # Reward neurons showing stable, moderate activity patterns
        activity_variability = self.activation_history[-20:].std()
        stability_factor = torch.exp(-activity_variability * 5)
        
        # Incorporate reward signal if provided
        if reward is not None:
            health_change = 0.01 * (0.4 * balanced_activity + 
                                    0.3 * stability_factor + 
                                    0.3 * torch.sigmoid(reward * 5) - 0.2)
        else:
            health_change = 0.01 * (0.6 * balanced_activity + 
                                    0.4 * stability_factor - 0.2)
        
        # Update health with bounds
        self.health = torch.clamp(self.health + health_change, 0.0, 1.0)
        
        return self.health
    
    def update_weights(self, error_signal, learning_rate=0.01, regularization=0.0001):
        """
        Update weights using biologically-plausible local learning rules.
        """
        # Skip update if not enough history
        if len(self.input_history) < 10 or len(self.activation_records) < 10:
            return 0.0
        
        # Calculate average recent input and activation
        recent_inputs = torch.stack(list(self.input_history)[-10:]).mean(0)
        recent_activation = torch.stack(list(self.activation_records)[-10:]).mean(0)
        
        # Calculate input-output correlation (Hebbian component)
        hebbian_update = recent_activation * recent_inputs
        
        # Error-driven component with sign-dependent scaling
        error_magnitude = torch.abs(error_signal.detach().mean())
        error_sign = torch.sign(error_signal.detach().mean())
        # Scale learning rate based on error magnitude (larger errors = larger updates)
        adaptive_lr = learning_rate * torch.sigmoid(error_magnitude * 3)
        
        # Directional update based on error sign and inputs
        error_update = -error_sign * recent_inputs * error_magnitude
        
        # Combine Hebbian and error-driven updates with dynamically adjusted balance
        # When error is large, favor error-driven updates
        hebbian_weight = torch.clamp(1.0 - error_magnitude, 0.2, 0.8)
        error_weight = 1.0 - hebbian_weight
        
        total_update = hebbian_update * hebbian_weight + error_update * error_weight
        
        # Apply L2 regularization to prevent overfitting
        regularization_term = regularization * self.weights.data
        
        # Apply weight update
        self.weights.data += adaptive_lr * total_update - regularization_term
        
        # Homeostatic bias adjustment with adaptive rate based on error
        activity_diff = self.target_activity - recent_activation
        bias_update_rate = self.homeostatic_rate * (1.0 + error_magnitude)
        self.bias.data += bias_update_rate * activity_diff
        
        # Apply weight normalization
        self.normalize_synapses()
        
        return torch.abs(total_update).mean().item()
    def update_calcium_metaplasticity(self):
        """
        Update synaptic plasticity based on calcium levels to regulate learning.
        """
        # Get current calcium concentration
        calcium = self.ion_channels.get_calcium_level()
        
        # Calculate calcium thresholds for LTP and LTD
        ca_low = 0.2  # Low threshold (LTD)
        ca_high = 0.7  # High threshold (LTP)
        
        # Calculate learning rate modulators based on calcium level
        ltp_factor = torch.sigmoid((calcium - ca_high) * 10)  # Long-term potentiation
        ltd_factor = torch.sigmoid((ca_low - calcium) * 10)   # Long-term depression
        
        # Store metaplasticity factors for weight updates
        self.ltp_rate = ltp_factor
        self.ltd_rate = ltd_factor
        
        # Calculate overall plasticity state (bell-shaped curve)
        plasticity_state = ltp_factor + ltd_factor - ltp_factor * ltd_factor
        
        # Update neuron's learning properties
        self.plasticity_factor = plasticity_state
        
        return plasticity_state

    def process_neuromodulator_signals(self, neuromodulator_levels):
        """
        Process neuromodulator signals to adapt neuron behavior.
        
        Args:
            neuromodulator_levels: Dictionary of neuromodulator levels
            
        Returns:
            dict: Updated neuron parameters
        """
        # Extract neuromodulator levels
        dopamine = neuromodulator_levels.get('dopamine', 0.5)
        serotonin = neuromodulator_levels.get('serotonin', 0.5)
        norepinephrine = neuromodulator_levels.get('norepinephrine', 0.5)
        acetylcholine = neuromodulator_levels.get('acetylcholine', 0.5)
        
        # Adjust learning rate based on dopamine (reward prediction)
        # Higher dopamine increases learning for positive outcomes
        learning_rate_mod = 0.5 + dopamine
        
        # Adjust threshold based on serotonin (mood/satisfaction)
        # Higher serotonin raises threshold, making neuron more selective
        threshold_mod = 0.3 + serotonin * 0.4
        
        # Adjust signal-to-noise ratio based on norepinephrine (attention)
        # Higher norepinephrine increases contrast between strong and weak inputs
        snr_mod = 0.8 + norepinephrine * 0.4
        
        # Adjust memory persistence based on acetylcholine (memory formation)
        # Higher acetylcholine strengthens memory trace
        memory_mod = 0.7 + acetylcholine * 0.6
        
        # Apply modulations to neuron parameters
        self.learning_rate_multiplier = learning_rate_mod
        self.activation_threshold = threshold_mod
        self.signal_noise_ratio = snr_mod
        self.memory_persistence = memory_mod
        
        # Return updated parameters
        return {
            'learning_rate_mod': learning_rate_mod,
            'threshold_mod': threshold_mod,
            'snr_mod': snr_mod,
            'memory_mod': memory_mod
        }

    def normalize_synapses(self, method="l2"):
        """
        Normalize synapse weights to prevent dominance and improve generalization.
        
        Args:
            method: Normalization method ("l2", "l1", or "max")
        """
        if method == "l2":
            # L2 normalization
            norm = torch.norm(self.weights, p=2)
            if norm > 1.0:
                self.weights.data = self.weights.data / norm
        elif method == "l1":
            # L1 normalization with scaling
            norm = torch.sum(torch.abs(self.weights))
            if norm > self.input_size * 0.5:  # Scale based on input size
                self.weights.data = self.weights.data / (norm / (self.input_size * 0.5))
        elif method == "max":
            # Max normalization
            max_val = torch.max(torch.abs(self.weights))
            if max_val > 1.0:
                self.weights.data = self.weights.data / max_val

    def apply_homeostasis(self):
        """
        Apply homeostatic plasticity to maintain target activity levels.
        """
        # Calculate the average activity level
        avg_activity = self.activation_history.mean()
        
        # Calculate the activity error (difference from target)
        activity_error = self.target_activity - avg_activity
        
        # Scale bias to drive activity toward target
        self.bias.data += self.homeostatic_rate * activity_error
        
        # Apply activity-dependent scaling to weights
        if avg_activity > self.target_activity * 1.5:
            # Globally scale down weights if too active
            self.weights.data *= (1.0 - self.homeostatic_rate * 0.1)
        elif avg_activity < self.target_activity * 0.5:
            # Globally scale up weights if too inactive
            self.weights.data *= (1.0 + self.homeostatic_rate * 0.1)
        
        # Regularize weights to prevent extreme values
        weight_std = torch.std(self.weights)
        weight_mean = torch.mean(self.weights)
        outlier_mask = torch.abs(self.weights - weight_mean) > 3 * weight_std
        # Pull outliers toward the mean
        self.weights.data[outlier_mask] = self.weights.data[outlier_mask] * 0.95 + weight_mean * 0.05

    def update_short_term_dynamics(self, input_activity):
        """
        Update short-term synaptic dynamics (facilitation and depression).
        
        Args:
            input_activity: Presynaptic activity levels
        """
        with torch.no_grad():
            # Calculate input activity strength
            if isinstance(input_activity, torch.Tensor):
                input_strength = torch.mean(torch.abs(input_activity))
            else:
                input_strength = torch.tensor(0.5)  # Default if no input provided
            
            # Facilitation: increases with activity, saturates, decays slowly
            facilitation_growth = 0.1 * torch.sigmoid(input_strength * 5 - 1)
            facilitation_decay = 0.05 * self.synaptic_facilitation
            self.synaptic_facilitation = torch.clamp(
                self.synaptic_facilitation + facilitation_growth - facilitation_decay,
                0.5, 2.0
            )
            
            # Depression: increases with high activity, recovers slowly
            depression_growth = 0.15 * torch.sigmoid(input_strength * 6 - 2)
            depression_recovery = 0.02 * (1.0 - self.synaptic_depression)
            self.synaptic_depression = torch.clamp(
                self.synaptic_depression + depression_growth - depression_recovery,
                0.2, 1.0
            )
            
            # Calculate the effective weight modifier
            self.effective_weight_modifier = self.synaptic_facilitation * self.synaptic_depression
            
            return self.effective_weight_modifier

class BiologicalGRUCell(nn.Module):
    """
    A GRU cell that incorporates biological mechanisms like ion channels,
    neuronal health, and Hebbian learning.
    """
    def __init__(self, input_size, hidden_size, neuron_death_threshold=0.1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.neuron_death_threshold = neuron_death_threshold
        
        # Create biological neurons for each gate
        self.update_gate_neurons = nn.ModuleList([
            BioNeuron(input_size + hidden_size) for _ in range(hidden_size)
        ])
        
        self.reset_gate_neurons = nn.ModuleList([
            BioNeuron(input_size + hidden_size) for _ in range(hidden_size)
        ])
        
        self.candidate_neurons = nn.ModuleList([
            BioNeuron(input_size + hidden_size) for _ in range(hidden_size)
        ])
        
        # Neuron masks (for dead neurons)
        self.register_buffer('neuron_mask', torch.ones(hidden_size))
        
        # Growth probability
        self.growth_probability = 0.01
        
        # Connection strength matrix (for visualization and analysis)
        self.register_buffer('connection_strength', torch.zeros(3, hidden_size, input_size + hidden_size))
    
    def reset_parameters(self):
        """Reset all neuron states"""
        for neuron in self.update_gate_neurons + self.reset_gate_neurons + self.candidate_neurons:
            neuron.reset_state()
        
# Inside class BiologicalGRUCell(nn.Module):
    # ... (other methods) ...

    def forward(self, x, h=None, error_signal=None, neuromodulators=None): # Add error_signal=None
        original_shape = x.shape
        is_3d_input = x.dim() > 2

        current_device = x.device
        if h is not None: h = h.to(current_device)

        if is_3d_input:
            x = x[:, -1] if x.size(1) > 0 else x.squeeze(1)

        batch_size = x.size(0)

        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=current_device)

        self.neuron_mask = self.neuron_mask.to(current_device) # Ensure mask is on correct device
        masked_h = h * self.neuron_mask

        xh = torch.cat([x, masked_h], dim=1)

        xh_update_norm = self.ln_xh_update(xh)
        xh_reset_norm = self.ln_xh_reset(xh)

        update_gate_out = torch.zeros(batch_size, self.hidden_size, device=current_device)
        reset_gate_out = torch.zeros(batch_size, self.hidden_size, device=current_device)
        candidate_out = torch.zeros(batch_size, self.hidden_size, device=current_device)

        if neuromodulators is not None:
            for i in range(self.hidden_size):
                if self.neuron_mask[i] > 0:
                    for neuron_group in [self.update_gate_neurons, self.reset_gate_neurons, self.candidate_neurons]:
                       if hasattr(neuron_group[i], 'process_neuromodulator_signals'):
                            neuron_group[i].process_neuromodulator_signals(neuromodulators)

        for i in range(self.hidden_size):
            if self.neuron_mask[i] > 0:
                update_gate_out[:, i:i+1] = self.update_gate_neurons[i](xh_update_norm)
                reset_gate_out[:, i:i+1] = self.reset_gate_neurons[i](xh_reset_norm)

                reset_h_component = reset_gate_out[:, i:i+1] * masked_h[:, i:i+1]
                # Using normalized components for candidate state's input
                # Ensure ln_x_candidate and ln_h_candidate output shapes that cat to input_size + hidden_size
                # This part of candidate_input formation needs to be consistent with BioNeuron's expected input_size
                # Assuming BioNeuron in candidate_neurons is (input_size + hidden_size)
                # Corrected candidate input to use individual components that are then handled by BioNeuron:
                # x_norm_cand = self.ln_x_candidate(x) # this is [B, input_size]
                # reset_h_norm_cand = self.ln_h_candidate(reset_h_component) # this is [B, 1] if h is sliced, need [B, hidden_size]
                # This requires careful slicing or BioNeuron that takes sliced input.
                # Simpler: candidate BioNeuron input should be cat of x and reset_h_component
                candidate_input_raw = torch.cat([x, reset_h_component], dim=1) # Concatenate raw, BioNeuron will handle
                candidate_out[:, i:i+1] = self.candidate_neurons[i](candidate_input_raw)


        h_new = (1 - update_gate_out) * masked_h + update_gate_out * candidate_out
        h_new = self.ln_h_new(h_new)
        h_new = h_new * self.neuron_mask

        if self.training:
            if error_signal is not None:
                 self._update_neurons(xh_update_norm, error_signal) 
            self._manage_neuron_lifecycle()

        return h_new
    
    def _update_neurons(self, inputs, error_signal, learning_rate=0.01):
        """
        Update all neurons with learning and health tracking.
        
        Args:
            inputs: Combined input and hidden state
            error_signal: Error/reward signal for learning
            learning_rate: Base learning rate
        """
        with torch.no_grad():
            # Update neuron weights and health
            for i in range(self.hidden_size):
                if self.neuron_mask[i] > 0:  # Only update living neurons
                    # Apply any neuromodulator adjustments to learning rate
                    effective_lr = learning_rate
                    if hasattr(self.update_gate_neurons[i], 'learning_rate_multiplier'):
                        effective_lr *= self.update_gate_neurons[i].learning_rate_multiplier
                    
                    # Update calcium-based metaplasticity
                    if hasattr(self.update_gate_neurons[i], 'update_calcium_metaplasticity'):
                        self.update_gate_neurons[i].update_calcium_metaplasticity()
                        self.reset_gate_neurons[i].update_calcium_metaplasticity()
                        self.candidate_neurons[i].update_calcium_metaplasticity()
                    
                    # Update weights with adaptive learning rate
                    update_gate_change = self.update_gate_neurons[i].update_weights(error_signal, effective_lr)
                    reset_gate_change = self.reset_gate_neurons[i].update_weights(error_signal, effective_lr)
                    candidate_change = self.candidate_neurons[i].update_weights(error_signal, effective_lr)
                    
                    # Apply homeostatic plasticity
                    if hasattr(self.update_gate_neurons[i], 'apply_homeostasis'):
                        self.update_gate_neurons[i].apply_homeostasis()
                        self.reset_gate_neurons[i].apply_homeostasis()
                        self.candidate_neurons[i].apply_homeostasis()
                    
                    # Update health
                    update_health = self.update_gate_neurons[i].update_health(error_signal.mean())
                    reset_health = self.reset_gate_neurons[i].update_health(error_signal.mean())
                    candidate_health = self.candidate_neurons[i].update_health(error_signal.mean())
                    
                    # Average health for this neuron
                    neuron_health = (update_health + reset_health + candidate_health) / 3
                    
                    # Death check
                    if neuron_health < self.neuron_death_threshold:
                        self.neuron_mask[i] = 0
                        print(f"Neuron {i} died (health: {neuron_health.item():.4f})")
            
            # Store connection strengths for analysis
            for i in range(self.hidden_size):
                if self.neuron_mask[i] > 0:
                    self.connection_strength[0, i] = self.update_gate_neurons[i].weights.abs()
                    self.connection_strength[1, i] = self.reset_gate_neurons[i].weights.abs()
                    self.connection_strength[2, i] = self.candidate_neurons[i].weights.abs()
            
            # Neuron growth probability (replace dead neurons)
            if torch.rand(1).item() < self.growth_probability:
                # Find a dead neuron to replace
                dead_indices = torch.where(self.neuron_mask == 0)[0]
                if len(dead_indices) > 0:
                    # Randomly choose one to regenerate
                    idx = dead_indices[torch.randint(0, len(dead_indices), (1,))]
                    self.neuron_mask[idx] = 1
                    
                    # Reset the neuron
                    self.update_gate_neurons[idx] = BioNeuron(self.input_size + self.hidden_size)
                    self.reset_gate_neurons[idx] = BioNeuron(self.input_size + self.hidden_size)
                    self.candidate_neurons[idx] = BioNeuron(self.input_size + self.hidden_size)
                    
                    print(f"Neuron {idx.item()} regenerated")

class BioGRU(nn.Module):
    """
    Multi-layer Biological GRU with neuron health tracking,
    synaptic pruning, and Hebbian learning.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # Create layers
        self.layers = nn.ModuleList([
            BiologicalGRUCell(
                input_size if i == 0 else hidden_size,
                hidden_size
            )
            for i in range(num_layers)
        ])
        
        # Output projection using a biological neuron
        self.output_neurons = nn.ModuleList([
            BioNeuron(hidden_size) for _ in range(output_size)
        ])
        
        # Neuromodulator system (influences learning rates and neuron health)
        self.neuromodulator_levels = {
            'dopamine': 0.5,    # Reward signaling
            'serotonin': 0.5,   # Mood/stability
            'norepinephrine': 0.5,  # Arousal/attention
            'acetylcholine': 0.5    # Learning/memory
        }
        
        # Activity records for pathway analysis
        self.register_buffer('pathway_activity', torch.zeros(num_layers, hidden_size))
        
        # Hidden state for recurrence
        self.hidden_states = None
    
    def reset_state(self):
        """Reset all internal states"""
        self.hidden_states = None
        for layer in self.layers:
            layer.reset_parameters()
        for neuron in self.output_neurons:
            neuron.reset_state()
    
    def update_neuromodulators(self, reward=None, error=None):
        """
        Update neuromodulator levels based on reward and error.
        
        Args:
            reward: Reward signal (if available)
            error: Error signal (if available)
        """
        # Decay factors
        decay = {
            'dopamine': 0.9,
            'serotonin': 0.95,
            'norepinephrine': 0.8,
            'acetylcholine': 0.93
        }
        
        # Update based on signals
        if reward is not None:
            # Dopamine increases with reward
            reward_value = reward.mean().item() if hasattr(reward, 'mean') else reward
            dopamine_change = (torch.sigmoid(torch.tensor(reward_value * 5)) - 0.5).item() * 0.1
            self.neuromodulator_levels['dopamine'] = self.neuromodulator_levels['dopamine'] * decay['dopamine'] + dopamine_change
        
        if error is not None:
            # Norepinephrine increases with high error (surprise)
            error_value = error.mean().item() if hasattr(error, 'mean') else error
            norepinephrine_change = (torch.sigmoid(torch.tensor(abs(error_value) * 5)) - 0.5).item() * 0.1
            self.neuromodulator_levels['norepinephrine'] = self.neuromodulator_levels['norepinephrine'] * decay['norepinephrine'] + norepinephrine_change
            
            # Acetylcholine increases with decreasing error (learning progress)
            if hasattr(self, 'last_error'):
                error_reduction = max(0, self.last_error - error_value)
                acetylcholine_change = error_reduction * 0.1
                self.neuromodulator_levels['acetylcholine'] = self.neuromodulator_levels['acetylcholine'] * decay['acetylcholine'] + acetylcholine_change
            self.last_error = error_value
        
        # Keep levels bounded
        for key in self.neuromodulator_levels:
            self.neuromodulator_levels[key] = max(0.1, min(1.0, self.neuromodulator_levels[key]))
    
# Inside class BioGRU(nn.Module):
    # ... (other methods) ...

    def forward(self, x, hidden_states_init=None, error_signal_for_update=None): # Add error_signal_for_update=None
        original_dim = x.dim()
        current_device = x.device

        if original_dim == 2:
            x = x.unsqueeze(1) 

        batch_size, seq_len, _ = x.size()

        if hidden_states_init is None:
            if self.hidden_states is None or self.hidden_states[0].size(0) != batch_size:
                self.hidden_states = [torch.zeros(batch_size, self.hidden_size, device=current_device) for _ in range(self.num_layers)]
            current_h_states = self.hidden_states
        else:
            current_h_states = [h.to(current_device) for h in hidden_states_init]

        output_sequence = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            layer_input = x_t
            new_h_for_step = []

            for i, layer in enumerate(self.layers):
                h_layer_prev = current_h_states[i]
                # Pass the error_signal to the BiologicalGRUCell layer
                h_layer_new = layer(layer_input, h_layer_prev, 
                                    error_signal=error_signal_for_update if self.training else None, 
                                    neuromodulators=self.neuromodulator_levels)
                new_h_for_step.append(h_layer_new)
                layer_input = h_layer_new

            current_h_states = new_h_for_step

            if self.training: # Pathway activity update logic
                for i in range(self.num_layers):
                    if hasattr(self.pathway_activity, 'device') and self.pathway_activity.device == current_device:
                         self.pathway_activity[i] = 0.9 * self.pathway_activity[i] + \
                                                   0.1 * current_h_states[i].detach().abs().mean(0)
                    else: # Initialize or move if device mismatch or first time
                         self.pathway_activity = self.pathway_activity.to(current_device)
                         self.pathway_activity[i] = 0.9 * self.pathway_activity[i] + \
                                                   0.1 * current_h_states[i].detach().abs().mean(0)


            last_layer_h = current_h_states[-1]
            if isinstance(self.output_neurons, BioNeuron):
                 step_output = self.output_neurons(last_layer_h)
            else:
                 step_output = self.output_neurons(last_layer_h)
            output_sequence.append(step_output)

        self.hidden_states = current_h_states

        outputs = torch.stack(output_sequence, dim=1)

        if original_dim == 2:
            outputs = outputs.squeeze(1)

        return outputs, self.hidden_states

    def _adjust_layer_plasticity(self, layer_idx, increase_factor=1.0):
        """
        Adjust plasticity of a layer based on activity patterns.
        
        Args:
            layer_idx: Index of the layer to adjust
            increase_factor: Factor to increase/decrease plasticity
        """
        if layer_idx >= len(self.layers):
            return
            
        layer = self.layers[layer_idx]
        
        # Ensure we're dealing with BiologicalGRUCell
        if not isinstance(layer, BiologicalGRUCell):
            return
            
        for neuron_array in [layer.update_gate_neurons, layer.reset_gate_neurons, layer.candidate_neurons]:
            for neuron in neuron_array:
                if hasattr(neuron, 'homeostatic_rate'):
                    neuron.homeostatic_rate *= increase_factor
                
                # Also adjust learning rate multiplier if it exists
                if hasattr(neuron, 'learning_rate_multiplier'):
                    neuron.learning_rate_multiplier *= increase_factor
    
    def update_from_error(self, error_signal, learning_rate=0.01):
        """
        Update all neurons based on error signal with enhanced adaptation.
        
        Args:
            error_signal: Error/reward signal
            learning_rate: Base learning rate
        """
        # Skip if not in training mode
        if not self.training:
            return
        
        # Update neuromodulators
        self.update_neuromodulators(error=error_signal.mean().item())
        
        # Get adaptive learning rates from neuromodulators
        dopamine_level = self.neuromodulator_levels.get('dopamine', 0.5)
        acetylcholine_level = self.neuromodulator_levels.get('acetylcholine', 0.5)
        
        # Modulate learning rate with dopamine and acetylcholine
        # Dopamine controls reward-based learning, acetylcholine controls memory formation
        effective_lr = learning_rate * dopamine_level * acetylcholine_level
        
        # Use error magnitude to scale learning - larger errors get larger updates
        error_magnitude = torch.abs(error_signal.mean()).item()
        error_scaling = torch.clamp(torch.tensor(error_magnitude * 2), 0.5, 2.0).item()
        effective_lr *= error_scaling
        
        # Update each layer with adapted learning rate
        for i, layer in enumerate(self.layers):
            # Apply layer-specific scaling based on position in network
            # Earlier layers typically need smaller learning rates
            layer_scaling = 0.7 + 0.3 * (i / max(1, self.num_layers - 1))
            layer_lr = effective_lr * layer_scaling
            
            # Apply the update to the layer
            layer_error = error_signal
            layer._update_neurons(None, layer_error, layer_lr)
        
        # Update output neurons with the same adaptive learning rate
        for i, neuron in enumerate(self.output_neurons):
            neuron_error = error_signal[:, i:i+1] if error_signal.dim() > 1 else error_signal
            neuron.update_weights(neuron_error, effective_lr)
            neuron.update_health(error_signal.mean())
            
            # Apply homeostatic plasticity if available
            if hasattr(neuron, 'apply_homeostasis'):
                neuron.apply_homeostasis()
            
            # Update calcium-based metaplasticity if available
            if hasattr(neuron, 'update_calcium_metaplasticity'):
                neuron.update_calcium_metaplasticity()
    
    def get_health_report(self):
        """
        Generate a health report for all neurons in the network.
        
        Returns:
            Dictionary with health statistics
        """
        report = {
            'layers': [],
            'overall_health': 0,
            'active_neuron_percentage': 0,
            'neuromodulator_levels': self.neuromodulator_levels
        }
        
        total_neurons = 0
        living_neurons = 0
        
        # Check each layer
        for i, layer in enumerate(self.layers):
            layer_report = {
                'living_neurons': layer.neuron_mask.sum().item(),
                'total_neurons': layer.hidden_size,
                'health_values': []
            }
            
            # Sample some neurons for detailed health
            sample_indices = torch.randperm(layer.hidden_size)[:min(5, layer.hidden_size)]
            for idx in sample_indices:
                if layer.neuron_mask[idx] > 0:  # Only report living neurons
                    update_health = layer.update_gate_neurons[idx].health.item()
                    reset_health = layer.reset_gate_neurons[idx].health.item()
                    candidate_health = layer.candidate_neurons[idx].health.item()
                    
                    layer_report['health_values'].append({
                        'neuron_idx': idx.item(),
                        'update_gate_health': update_health,
                        'reset_gate_health': reset_health,
                        'candidate_health': candidate_health,
                        'average_health': (update_health + reset_health + candidate_health) / 3
                    })
            
            report['layers'].append(layer_report)
            
            # Update totals
            total_neurons += layer.hidden_size
            living_neurons += layer.neuron_mask.sum().item()
        
        # Overall statistics
        report['overall_health'] = living_neurons / max(1, total_neurons)
        report['active_neuron_percentage'] = int(100 * living_neurons / max(1, total_neurons))
        
        return report
    
    def analyze_pathways(self):
        """
        Analyze active pathways in the network.
        
        Returns:
            Dictionary with pathway information
        """
        pathway_info = {
            'strongest_pathways': [],
            'layer_connectivity': []
        }
        
        # Analyze each layer's connectivity
        for i, layer in enumerate(self.layers):
            # Get connection strengths
            connections = layer.connection_strength
            
            # Find the strongest pathways
            avg_strength = connections.mean(0)  # Average across gates
            top_indices = torch.topk(avg_strength.sum(1), min(5, avg_strength.size(0))).indices
            
            for idx in top_indices:
                if layer.neuron_mask[idx] > 0:  # Only include living neurons
                    # Get incoming connection strengths
                    incoming = avg_strength[idx].detach().cpu().numpy()
                    
                    # Record this strong pathway
                    pathway_info['strongest_pathways'].append({
                        'layer': i,
                        'neuron_idx': idx.item(),
                        'activity_level': self.pathway_activity[i, idx].item(),
                        'connection_strength': float(incoming.mean())
                    })
            
            # Overall layer connectivity
            living_mask = layer.neuron_mask > 0
            if living_mask.sum() > 0:
                layer_connectivity = {
                    'layer': i,
                    'connection_density': float((connections.abs() > 0.1).float().mean()),
                    'avg_connection_strength': float(connections.abs().mean()),
                    'neuron_health': float(layer.neuron_mask.mean())
                }
                pathway_info['layer_connectivity'].append(layer_connectivity)
        
        return pathway_info
# Inside class BiologicalGRUCell(nn.Module):
# ... (other methods) ...

    def forward(self, x, h=None, error_signal=None, neuromodulators=None): # Add error_signal=None
        original_shape = x.shape
        is_3d_input = x.dim() > 2

        current_device = x.device
        if h is not None: h = h.to(current_device)

        if is_3d_input:
            x = x[:, -1] if x.size(1) > 0 else x.squeeze(1)

        batch_size = x.size(0)

        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=current_device)

        self.neuron_mask = self.neuron_mask.to(current_device) # Ensure mask is on correct device
        masked_h = h * self.neuron_mask

        xh = torch.cat([x, masked_h], dim=1)

        xh_update_norm = self.ln_xh_update(xh)
        xh_reset_norm = self.ln_xh_reset(xh)

        update_gate_out = torch.zeros(batch_size, self.hidden_size, device=current_device)
        reset_gate_out = torch.zeros(batch_size, self.hidden_size, device=current_device)
        candidate_out = torch.zeros(batch_size, self.hidden_size, device=current_device)

        if neuromodulators is not None:
            for i in range(self.hidden_size):
                if self.neuron_mask[i] > 0:
                    for neuron_group in [self.update_gate_neurons, self.reset_gate_neurons, self.candidate_neurons]:
                       if hasattr(neuron_group[i], 'process_neuromodulator_signals'):
                            neuron_group[i].process_neuromodulator_signals(neuromodulators)

        for i in range(self.hidden_size):
            if self.neuron_mask[i] > 0:
                update_gate_out[:, i:i+1] = self.update_gate_neurons[i](xh_update_norm)
                reset_gate_out[:, i:i+1] = self.reset_gate_neurons[i](xh_reset_norm)

                reset_h_component = reset_gate_out[:, i:i+1] * masked_h[:, i:i+1]
                # Using normalized components for candidate state's input
                # Ensure ln_x_candidate and ln_h_candidate output shapes that cat to input_size + hidden_size
                # This part of candidate_input formation needs to be consistent with BioNeuron's expected input_size
                # Assuming BioNeuron in candidate_neurons is (input_size + hidden_size)
                # Corrected candidate input to use individual components that are then handled by BioNeuron:
                # x_norm_cand = self.ln_x_candidate(x) # this is [B, input_size]
                # reset_h_norm_cand = self.ln_h_candidate(reset_h_component) # this is [B, 1] if h is sliced, need [B, hidden_size]
                # This requires careful slicing or BioNeuron that takes sliced input.
                # Simpler: candidate BioNeuron input should be cat of x and reset_h_component
                candidate_input_raw = torch.cat([x, reset_h_component], dim=1) # Concatenate raw, BioNeuron will handle
                candidate_out[:, i:i+1] = self.candidate_neurons[i](candidate_input_raw)


        h_new = (1 - update_gate_out) * masked_h + update_gate_out * candidate_out
        h_new = self.ln_h_new(h_new)
        h_new = h_new * self.neuron_mask

        if self.training:
            if error_signal is not None:
                 self._update_neurons(xh_update_norm, error_signal) 
            self._manage_neuron_lifecycle()

        return h_new


    def train_biogru(model, dataloader, epochs=5, learning_rate=0.01):
        """
        Train the BioGRU model with local learning rules.
        
        Args:
            model: BioGRU model
            dataloader: DataLoader with training data
            epochs: Number of training epochs
            learning_rate: Base learning rate
            
        Returns:
            Training history
        """
        model.train()
        history = {
            'loss': [],
            'neuron_health': []
        }
        
        # MSE loss function
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                # Reset model state for each sequence
                model.reset_state()
                
                # Forward pass
                output, _ = model(data)
                
                # Calculate loss
                loss = criterion(output, target)
                epoch_loss += loss.item()
                
                # Calculate error signal for local update
                error_signal = target - output
                
                # Update model with error signal
                model.update_from_error(error_signal, learning_rate)
                
                # Report progress
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.6f}")
            
            # Calculate epoch average loss
            avg_loss = epoch_loss / len(dataloader)
            history['loss'].append(avg_loss)
            
            # Get health report
            health_report = model.get_health_report()
            history['neuron_health'].append(health_report['overall_health'])
            
            # Report epoch results
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}, Active neurons: {health_report['active_neuron_percentage']}%")
            print(f"Neuromodulator levels: {health_report['neuromodulator_levels']}")
            
            # Analyze pathways at end of epoch
            if epoch % 2 == 0:
                pathway_info = model.analyze_pathways()
                if pathway_info['strongest_pathways']:
                    print("\nStrongest pathways:")
                    for path in pathway_info['strongest_pathways'][:3]:
                        print(f"  Layer {path['layer']}, Neuron {path['neuron_idx']}: Strength {path['connection_strength']:.4f}, Activity {path['activity_level']:.4f}")
        
        return history




# Example usage
def create_sample_data(batch_size=32, seq_length=10, input_size=5, output_size=1):
    """Create sample data for testing"""
    # Create a sinusoidal pattern with noise
    x = torch.randn(batch_size, seq_length, input_size)
    y = torch.sin(x.sum(dim=2, keepdim=True) * 0.5) + torch.randn(batch_size, seq_length, 1) * 0.1
    return x, y

# Create sample dataloader
def create_dataloader(batch_size=32, num_batches=100):
    """Create a dataloader with sample data"""
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create dataset
    dataset = []
    for _ in range(num_batches):
        x, y = create_sample_data(batch_size=batch_size)
        dataset.append((x, y))
    
    # Simple iterable dataloader
    class SimpleDataLoader:
        def __init__(self, dataset):
            self.dataset = dataset
        
        def __iter__(self):
            return iter(self.dataset)
        
        def __len__(self):
            return len(self.dataset)
    
    return SimpleDataLoader(dataset)

