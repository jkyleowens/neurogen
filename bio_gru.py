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
            x: Input signal
            previous_activity: Previous activation state
            
        Returns:
            Neuron activation
        """
        # Store input for plasticity
        if self.training:
            self.input_history.append(x.detach().mean(0))
        
        # Apply short-term plasticity to inputs
        effective_weights = self.weights * self.synaptic_facilitation * self.synaptic_depression
        
        # Calculate weighted input
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
            self.activation_records.append(activation.detach().mean(0))
            # Shift activation history and add new activation
            self.activation_history = torch.cat([self.activation_history[1:], activation.detach().mean(0).unsqueeze(0)])
            self.activation_count += activation.detach().mean() > 0.1
        
        # Update short-term plasticity
        if self.training:
            # Synaptic facilitation (increases with activity)
            self.synaptic_facilitation = self.synaptic_facilitation * 0.95 + 0.05 * activation.detach().mean(0).unsqueeze(-1)
            # Synaptic depression (decreases with activity)
            self.synaptic_depression = self.synaptic_depression * 0.98 + 0.02 * (1 - activation.detach().mean(0).unsqueeze(-1))
        
        return activation
    
    def update_health(self, reward=None):
        """
        Update neuron health based on activity and utility.
        
        Args:
            reward: Optional reward signal to influence health
            
        Returns:
            Current health value
        """
        with torch.no_grad():
            # Calculate recent activity level
            recent_activity = self.activation_history[-20:].mean()
            
            # Activity-based health update
            activity_factor = torch.sigmoid((recent_activity - 0.01) * 10)  # Threshold at 0.01
            
            # Utility-based health update (if utility tracking is enabled)
            utility_factor = torch.sigmoid(self.utility)
            
            # Health update rate
            update_rate = 0.01
            
            # Combine factors with reward if available
            if reward is not None:
                reward_factor = torch.sigmoid(reward * 5)  # Scale and bound reward
                health_change = update_rate * (0.4 * activity_factor + 0.3 * utility_factor + 0.3 * reward_factor - 0.2)
            else:
                health_change = update_rate * (0.6 * activity_factor + 0.4 * utility_factor - 0.2)
            
            # Update health
            self.health = torch.clamp(self.health + health_change, 0.0, 1.0)
            
            return self.health
    
    def update_weights(self, error_signal, learning_rate=0.01):
        """
        Update weights using local learning rules influenced by calcium levels.
        
        Args:
            error_signal: Error/reward signal
            learning_rate: Base learning rate
            
        Returns:
            Weight update magnitude
        """
        with torch.no_grad():
            # Skip update if not enough history
            if len(self.input_history) < 10 or len(self.activation_records) < 10:
                return 0.0
            
            # Track error
            self.error_history.append(error_signal.detach().mean())
            
            # Calculate average recent input and activation
            recent_inputs = torch.stack(list(self.input_history)[-10:]).mean(0)
            recent_activation = torch.stack(list(self.activation_records)[-10:]).mean(0)
            
            # Calcium modulation of learning rate
            calcium_level = self.ion_channels.get_calcium_level().mean()
            calcium_factor = torch.sigmoid(calcium_level * 5 - 1)  # Threshold effect
            effective_lr = learning_rate * calcium_factor.item()
            
            # Hebbian component (correlation between input and output)
            hebbian_update = recent_activation * recent_inputs
            
            # Error-driven component
            error_update = error_signal.detach().mean() * recent_inputs
            
            # Combine updates
            total_update = hebbian_update * 0.5 + error_update * 0.5
            
            # Apply weight update scaled by learning rate
            self.weights.data += effective_lr * total_update
            
            # Homeostatic bias adjustment to maintain target activity
            activity_diff = self.target_activity - recent_activation
            self.bias.data += self.homeostatic_rate * activity_diff
            
            # Weight normalization to prevent explosion
            if torch.norm(self.weights.data) > 3.0:
                self.weights.data = self.weights.data * 3.0 / torch.norm(self.weights.data)
            
            # Update utility based on contribution to error reduction
            if len(self.error_history) > 10:
                recent_errors = torch.tensor(list(self.error_history)[-10:])
                error_improvement = recent_errors[0] - recent_errors[-1]
                utility_update = torch.sigmoid(error_improvement * 5) - 0.5
                self.utility = 0.9 * self.utility + 0.1 * utility_update
            
            return torch.abs(total_update).mean().item()


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
        
    def forward(self, x, h=None, error_signal=None):
        """
        Forward pass with biological neuron dynamics.
        
        Args:
            x: Input tensor [batch_size, input_size]
            h: Previous hidden state [batch_size, hidden_size]
            error_signal: Optional error signal for learning
            
        Returns:
            New hidden state
        """
        batch_size = x.size(0)
        
        # Initialize hidden state if needed
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Concatenate input and hidden state
        xh = torch.cat([x, h], dim=1)
        
        # Apply neuron mask to hide dead neurons
        masked_h = h * self.neuron_mask
        
        # Gate computations
        update_gate = torch.zeros(batch_size, self.hidden_size, device=x.device)
        reset_gate = torch.zeros(batch_size, self.hidden_size, device=x.device)
        candidate = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Process each neuron individually (allows for neuron-specific dynamics)
        for i in range(self.hidden_size):
            if self.neuron_mask[i] > 0:  # Only process living neurons
                # Update gate
                update_gate[:, i:i+1] = self.update_gate_neurons[i](xh)
                
                # Reset gate
                reset_gate[:, i:i+1] = self.reset_gate_neurons[i](xh)
                
                # Candidate state (with reset gate applied)
                reset_hidden = reset_gate[:, i:i+1] * h[:, i:i+1]
                candidate_input = torch.cat([x, reset_hidden], dim=1)
                candidate[:, i:i+1] = self.candidate_neurons[i](candidate_input)
        
        # Combine gates to get new hidden state (standard GRU formula)
        h_new = (1 - update_gate) * h + update_gate * candidate
        
        # Apply mask to hide dead neurons
        h_new = h_new * self.neuron_mask
        
        # If in training mode and error signal provided, update neurons
        if self.training and error_signal is not None:
            self._update_neurons(xh, error_signal)
        
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
                    # Update weights
                    update_gate_change = self.update_gate_neurons[i].update_weights(error_signal, learning_rate)
                    reset_gate_change = self.reset_gate_neurons[i].update_weights(error_signal, learning_rate)
                    candidate_change = self.candidate_neurons[i].update_weights(error_signal, learning_rate)
                    
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
    
    def forward(self, x, hidden_states=None):
        """
        Forward pass through all layers with biological dynamics.
        
        Args:
            x: Input sequence [batch_size, seq_len, input_size]
            hidden_states: Optional initial hidden states
            
        Returns:
            Output sequence and final hidden states
        """
        batch_size, seq_len, _ = x.size()
        
        # Initialize or use provided hidden states
        if hidden_states is None:
            if self.hidden_states is None:
                self.hidden_states = [None] * self.num_layers
            hidden_states = self.hidden_states
        
        # Process each time step
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Process through each layer
            layer_input = x_t
            new_hidden_states = []
            
            for i, layer in enumerate(self.layers):
                h_i = hidden_states[i]
                h_i = layer(layer_input, h_i)
                new_hidden_states.append(h_i)
                layer_input = h_i
                
                # Record pathway activity for analysis
                if self.training:
                    self.pathway_activity[i] = 0.9 * self.pathway_activity[i] + 0.1 * h_i.detach().abs().mean(0)
            
            # Update hidden states
            hidden_states = new_hidden_states
            self.hidden_states = hidden_states
            
            # Process through output neurons
            output = torch.zeros(batch_size, self.output_size, device=x.device)
            for i in range(self.output_size):
                output[:, i:i+1] = self.output_neurons[i](hidden_states[-1])
            
            outputs.append(output)
        
        # Stack outputs [batch_size, seq_len, output_size]
        outputs = torch.stack(outputs, dim=1)
        
        return outputs, hidden_states
    
    def update_from_error(self, error_signal, learning_rate=0.01):
        """
        Update all neurons based on error signal.
        
        Args:
            error_signal: Error/reward signal
            learning_rate: Base learning rate
        """
        # Skip if not in training mode
        if not self.training:
            return
        
        # Update neuromodulators
        self.update_neuromodulators(error=error_signal.mean().item())
        
        # Modulate learning rate with acetylcholine
        effective_lr = learning_rate * self.neuromodulator_levels['acetylcholine']
        
        # Update each layer
        for i, layer in enumerate(self.layers):
            layer_error = error_signal
            layer._update_neurons(None, layer_error, effective_lr)
        
        # Update output neurons
        for i, neuron in enumerate(self.output_neurons):
            neuron_error = error_signal[:, i:i+1] if error_signal.dim() > 1 else error_signal
            neuron.update_weights(neuron_error, effective_lr)
            neuron.update_health(error_signal.mean())
    
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


# Usage example
if __name__ == "__main__":
    # Create model
    input_size = 5
    hidden_size = 32
    model = BioGRU(input_size=input_size, hidden_size=hidden_size, num_layers=2, output_size=1)
    
    # Create dataloader
    dataloader = create_dataloader(batch_size=32, num_batches=100)
    
    # Train model
    history = train_biogru(model, dataloader, epochs=5)
    
    # Print final health report
    health_report = model.get_health_report()
    print("\nFinal health report:")
    print(f"Overall health: {health_report['overall_health']:.4f}")
    print(f"Active neurons: {health_report['active_neuron_percentage']}%")
    print(f"Neuromodulator levels: {health_report['neuromodulator_levels']}")
    
    # Analyze pathways
    pathway_info = model.analyze_pathways()
    print("\nNetork pathway analysis:")
    for layer_info in pathway_info['layer_connectivity']:
        print(f"Layer {layer_info['layer']}: Connection density {layer_info['connection_density']:.4f}, "
              f"Avg strength {layer_info['avg_connection_strength']:.4f}, "
              f"Neuron health {layer_info['neuron_health']:.4f}")