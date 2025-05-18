"""
Persistent GRU Neural Network Controller

This module implements a biologically-inspired GRU-based neural network controller with persistent memory
that serves as the central controller for the brain-inspired neural network system.

The implementation draws inspiration from biological neural mechanisms including:
1. Persistent neural activity observed in prefrontal cortex during working memory tasks
2. Neuromodulatory systems that regulate neural activity based on reward signals
3. Synaptic plasticity mechanisms that allow for dynamic connection strength adjustments
4. Attentional gating mechanisms that selectively enhance or suppress information flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PersistentGRUCell(nn.Module):
    """
    A biologically-inspired GRU cell with persistent memory capabilities.
    
    This cell extends the standard GRU cell with:
    1. A persistent memory component that allows for long-term information storage and retrieval,
       inspired by working memory circuits in the prefrontal cortex
    2. Neuromodulatory gating mechanisms that can dynamically adjust the cell's behavior
       based on reward signals, similar to dopaminergic modulation in the brain
    3. Dynamic connection strength adjustment based on attention and neuromodulatory signals,
       inspired by synaptic plasticity mechanisms
    4. Separate pathways for information processing, similar to segregated cortical pathways
    
    Biological inspiration:
    - The reset and update gates are analogous to the gating mechanisms controlled by
      inhibitory interneurons in cortical circuits
    - The persistent memory component mimics the sustained firing patterns observed in
      prefrontal cortex neurons during working memory tasks
    - The neuromodulatory influence on gates resembles how dopamine, serotonin, and other
      neurotransmitters modulate neural activity in response to rewards and other signals
    - The dynamic connection strength adjustment is inspired by activity-dependent
      synaptic plasticity mechanisms like LTP and LTD
    """
    
    def __init__(self, input_size, hidden_size, persistent_memory_size):
        """
        Initialize the Persistent GRU Cell.
        
        Args:
            input_size (int): Size of the input features
            hidden_size (int): Size of the hidden state
            persistent_memory_size (int): Size of the persistent memory
        """
        super(PersistentGRUCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.persistent_memory_size = persistent_memory_size
        
        # Standard GRU gates
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.new_gate = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Persistent memory components
        # Read gate controls how much information is read from persistent memory
        self.memory_read = nn.Linear(hidden_size, persistent_memory_size)
        # Write gate controls how much information is written to persistent memory
        self.memory_write = nn.Linear(hidden_size, persistent_memory_size)
        # Forget gate controls how much information is retained in persistent memory
        self.memory_forget = nn.Linear(hidden_size, persistent_memory_size)
        
        # Memory addressing mechanism - inspired by hippocampal indexing
        self.memory_address = nn.Linear(hidden_size, persistent_memory_size)
        
        # Neuromodulatory influence parameters
        # These parameters control how neuromodulators affect the gates
        self.dopamine_influence = nn.Parameter(torch.ones(3))  # Influences [reset, update, new] gates
        self.serotonin_influence = nn.Parameter(torch.ones(3))  # Influences [reset, update, new] gates
        self.norepinephrine_influence = nn.Parameter(torch.ones(3))  # Influences [reset, update, new] gates
        self.acetylcholine_influence = nn.Parameter(torch.ones(3))  # Influences [reset, update, new] gates
        
        # Dynamic connection strength parameters
        # These control how connections are strengthened or weakened based on activity
        self.connection_strength = nn.Parameter(torch.ones(hidden_size))
        self.connection_decay = nn.Parameter(torch.ones(1) * 0.99)
        
        # Attention mechanism parameters
        # These control how attention modulates information flow
        self.attention_gate = nn.Linear(hidden_size, hidden_size)
        
        # Initialize parameters
        self.init_parameters()
        
        # Internal state tracking
        self.register_buffer('connection_history', torch.zeros(hidden_size))
        self.register_buffer('reward_history', torch.zeros(1))
        
    def init_parameters(self):
        """
        Initialize the parameters with biologically-inspired distributions.
        
        Uses Xavier/Glorot initialization for weight matrices and small positive
        values for biases to mimic biological neural systems where neurons have
        a baseline firing rate.
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gate' in name:
                    # Initialize gate weights with slightly positive bias toward being closed
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                if 'forget' in name:
                    # Initialize forget gate biases to 1.0 to encourage information retention
                    nn.init.constant_(param, 1.0)
                elif 'gate' in name:
                    # Initialize other gate biases to small negative values
                    nn.init.constant_(param, -0.1)
                else:
                    nn.init.zeros_(param)
            elif 'influence' in name:
                # Initialize neuromodulatory influences to small positive values
                nn.init.constant_(param, 0.1)
    
    def apply_neuromodulation(self, gates, neuromodulators=None):
        """
        Apply neuromodulatory influences to the gates.
        
        Args:
            gates (tuple): Tuple of (reset_gate, update_gate, new_gate) tensors
            neuromodulators (dict, optional): Dictionary of neuromodulator levels
                with keys 'dopamine', 'serotonin', 'norepinephrine', 'acetylcholine'
                
        Returns:
            tuple: Modulated (reset_gate, update_gate, new_gate) tensors
        """
        reset_gate, update_gate, new_gate = gates
        
        # If no neuromodulators provided, return gates unchanged
        if neuromodulators is None:
            return reset_gate, update_gate, new_gate
        
        # Extract neuromodulator levels
        dopamine = neuromodulators.get('dopamine', torch.zeros_like(reset_gate[:, :1]))
        serotonin = neuromodulators.get('serotonin', torch.zeros_like(reset_gate[:, :1]))
        norepinephrine = neuromodulators.get('norepinephrine', torch.zeros_like(reset_gate[:, :1]))
        acetylcholine = neuromodulators.get('acetylcholine', torch.zeros_like(reset_gate[:, :1]))
        
        # Apply neuromodulatory influences
        # Dopamine primarily affects reward-based learning and motivation
        dopamine_mod = dopamine * self.dopamine_influence.unsqueeze(0)
        # Serotonin affects mood and behavioral inhibition
        serotonin_mod = serotonin * self.serotonin_influence.unsqueeze(0)
        # Norepinephrine affects attention and arousal
        norepinephrine_mod = norepinephrine * self.norepinephrine_influence.unsqueeze(0)
        # Acetylcholine affects learning and memory
        acetylcholine_mod = acetylcholine * self.acetylcholine_influence.unsqueeze(0)
        
        # Combine modulations
        reset_mod = dopamine_mod[:, 0:1] + serotonin_mod[:, 0:1] + norepinephrine_mod[:, 0:1] + acetylcholine_mod[:, 0:1]
        update_mod = dopamine_mod[:, 1:2] + serotonin_mod[:, 1:2] + norepinephrine_mod[:, 1:2] + acetylcholine_mod[:, 1:2]
        new_mod = dopamine_mod[:, 2:3] + serotonin_mod[:, 2:3] + norepinephrine_mod[:, 2:3] + acetylcholine_mod[:, 2:3]
        
        # Apply modulations to gates
        reset_gate = torch.sigmoid(reset_gate + reset_mod)
        update_gate = torch.sigmoid(update_gate + update_mod)
        new_gate = torch.tanh(new_gate + new_mod)
        
        return reset_gate, update_gate, new_gate
    
    def update_connections(self, hidden, reward=None):
        """
        Update connection strengths based on activity and reward.
        
        This mimics synaptic plasticity mechanisms like LTP and LTD that
        strengthen or weaken connections based on neural activity and reward signals.
        
        Args:
            hidden (torch.Tensor): Hidden state tensor
            reward (torch.Tensor, optional): Reward signal
            
        Returns:
            torch.Tensor: Updated connection strength tensor
        """
        # Decay existing connection strengths
        self.connection_history = self.connection_history * self.connection_decay
        
        # Update connection history based on current activity
        activity = torch.mean(torch.abs(hidden), dim=0)
        self.connection_history = self.connection_history + activity.detach()
        
        # Update connection strengths based on reward if provided
        if reward is not None:
            self.reward_history = self.reward_history * 0.9 + reward.mean().detach() * 0.1
            reward_factor = torch.sigmoid(self.reward_history * 5)  # Scale and sigmoid for stability
            connection_strength = torch.sigmoid(self.connection_history * reward_factor)
        else:
            connection_strength = torch.sigmoid(self.connection_history)
        
        return self.connection_strength * connection_strength
    
    def forward(self, x, hidden, persistent_memory, neuromodulators=None, reward=None):
        """
        Forward pass of the Persistent GRU Cell.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            hidden (torch.Tensor): Hidden state tensor of shape (batch_size, hidden_size)
            persistent_memory (torch.Tensor): Persistent memory tensor of shape (batch_size, persistent_memory_size)
            neuromodulators (dict, optional): Dictionary of neuromodulator levels
            reward (torch.Tensor, optional): Reward signal
            
        Returns:
            tuple: (new_hidden, new_persistent_memory)
        """
        batch_size = x.size(0)
        
        # Update connection strengths based on activity and reward
        connection_strength = self.update_connections(hidden, reward)
        
        # Apply attention mechanism to hidden state
        attention = torch.sigmoid(self.attention_gate(hidden))
        hidden_attended = hidden * attention
        
        # Combine input and attended hidden state
        combined = torch.cat((x, hidden_attended), dim=1)
        
        # Compute GRU gates
        reset = self.reset_gate(combined)
        update = self.update_gate(combined)
        combined_reset = torch.cat((x, reset * hidden_attended), dim=1)
        new = self.new_gate(combined_reset)
        
        # Apply neuromodulation to gates
        reset, update, new = self.apply_neuromodulation((reset, update, new), neuromodulators)
        
        # Update hidden state with dynamic connection strength
        hidden_new = (1 - update) * hidden + update * new * connection_strength.unsqueeze(0)
        
        # Persistent memory operations
        # Generate memory addressing weights - determines which memory locations to access
        address_weights = torch.softmax(self.memory_address(hidden_new), dim=1)
        
        # Generate read, write, and forget weights
        read_weights = torch.sigmoid(self.memory_read(hidden_new))
        write_weights = torch.sigmoid(self.memory_write(hidden_new))
        forget_weights = torch.sigmoid(self.memory_forget(hidden_new))
        
        # Apply neuromodulation to memory operations if available
        if neuromodulators is not None:
            # Acetylcholine particularly affects memory operations
            acetylcholine = neuromodulators.get('acetylcholine', torch.zeros_like(read_weights[:, :1]))
            # Enhance read operations with acetylcholine (attention/focus)
            read_weights = read_weights * (1 + acetylcholine)
            # Dopamine affects memory writing (reward-based learning)
            dopamine = neuromodulators.get('dopamine', torch.zeros_like(write_weights[:, :1]))
            write_weights = write_weights * (1 + dopamine)
        
        # Update persistent memory with addressing
        memory_new = forget_weights * persistent_memory
        
        # Project hidden state to memory size for writing
        hidden_projected = F.linear(
            hidden_new, 
            weight=torch.ones(self.persistent_memory_size, self.hidden_size, device=hidden_new.device) / self.hidden_size
        )
        
        # Apply write weights and addressing
        memory_write_content = write_weights * hidden_projected * address_weights
        memory_new = memory_new + memory_write_content
        
        # Read from memory with addressing
        memory_read_content = read_weights * memory_new
        
        # Project memory content to hidden size for reading
        if self.persistent_memory_size != self.hidden_size:
            memory_read_projected = F.linear(
                memory_read_content,
                weight=torch.ones(self.hidden_size, self.persistent_memory_size, device=memory_read_content.device) / self.persistent_memory_size
            )
            hidden_final = hidden_new + memory_read_projected
        else:
            hidden_final = hidden_new + memory_read_content
        
        return hidden_final, memory_new


class PersistentGRUController(nn.Module):
    """
    A biologically-inspired GRU-based controller with persistent memory capabilities.
    
    This controller serves as the central component of the brain-inspired
    neural network system, coordinating information flow between different modules.
    
    Biological inspiration:
    - The layered architecture mimics the hierarchical organization of cortical regions
    - The persistent memory component resembles working memory circuits in prefrontal cortex
    - The integration of neuromodulatory signals is inspired by how brain regions are
      regulated by neurotransmitters like dopamine, serotonin, and acetylcholine
    - The reward-based updates reflect reinforcement learning mechanisms in the brain
    """
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, 
                 persistent_memory_size=128, dropout=0.0):
        """
        Initialize the Persistent GRU Controller.
        
        Args:
            input_size (int): Size of the input features
            hidden_size (int): Size of the hidden state
            output_size (int): Size of the output
            num_layers (int): Number of GRU layers
            persistent_memory_size (int): Size of the persistent memory
            dropout (float): Dropout probability
        """
        super(PersistentGRUController, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.persistent_memory_size = persistent_memory_size
        self.dropout = dropout
        
        # Create GRU cells with persistent memory
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.cells.append(PersistentGRUCell(layer_input_size, hidden_size, persistent_memory_size))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        
        # Neuromodulator integration weights
        self.neuromodulator_weights = nn.Parameter(torch.ones(4))  # [dopamine, serotonin, norepinephrine, acetylcholine]
        
        # Initialize internal state tracking
        self.state = None
    
    def forward(self, x, hidden=None, persistent_memory=None, neuromodulators=None, reward=None):
        """
        Forward pass of the Persistent GRU Controller.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
            hidden (torch.Tensor, optional): Initial hidden state tensor
            persistent_memory (torch.Tensor, optional): Initial persistent memory tensor
            neuromodulators (dict, optional): Dictionary of neuromodulator levels
            reward (torch.Tensor, optional): Reward signal
            
        Returns:
            tuple: (outputs, hidden_states, persistent_memories)
        """
        batch_size, seq_length, _ = x.size()
        device = x.device
        
        # Initialize hidden states and persistent memories if not provided
        if hidden is None:
            hidden = [torch.zeros(batch_size, self.hidden_size, device=device) 
                     for _ in range(self.num_layers)]
        if persistent_memory is None:
            persistent_memory = [torch.zeros(batch_size, self.persistent_memory_size, device=device) 
                               for _ in range(self.num_layers)]
        
        # Restore from internal state if available
        if self.state is not None:
            hidden, persistent_memory = self.state
        
        # Process each time step
        outputs = []
        hidden_states = []
        persistent_memories = []
        
        for t in range(seq_length):
            x_t = x[:, t, :]
            
            # Process through each layer
            for layer in range(self.num_layers):
                if layer > 0:
                    x_t = self.dropout_layer(x_t)
                
                # Get reward for this time step if provided
                reward_t = None
                if reward is not None:
                    reward_t = reward[:, t].unsqueeze(1) if reward.dim() > 1 else reward.unsqueeze(1)
                
                # Get neuromodulators for this time step if provided
                neuromodulators_t = None
                if neuromodulators is not None:
                    neuromodulators_t = {
                        k: v[:, t].unsqueeze(1) if v.dim() > 2 else v
                        for k, v in neuromodulators.items()
                    }
                
                # Process through GRU cell
                hidden[layer], persistent_memory[layer] = self.cells[layer](
                    x_t, hidden[layer], persistent_memory[layer], neuromodulators_t, reward_t
                )
                
                # Apply layer normalization for stability
                hidden[layer] = self.layer_norms[layer](hidden[layer])
                
                x_t = hidden[layer]
            
            # Apply output layer
            output = self.output_layer(hidden[-1])
            outputs.append(output)
            
            # Store states for return
            hidden_states.append([h.clone() for h in hidden])
            persistent_memories.append([m.clone() for m in persistent_memory])
        
        # Stack outputs along sequence dimension
        outputs = torch.stack(outputs, dim=1)
        
        # Update internal state
        self.state = (
            [h.detach() for h in hidden],
            [m.detach() for m in persistent_memory]
        )
        
        return outputs, hidden_states, persistent_memories
    
    def init_hidden(self, batch_size, device):
        """
        Initialize hidden states and persistent memories.
        
        Args:
            batch_size (int): Batch size
            device (torch.device): Device to create tensors on
            
        Returns:
            tuple: (hidden, persistent_memory)
        """
        hidden = [torch.zeros(batch_size, self.hidden_size, device=device) 
                 for _ in range(self.num_layers)]
        persistent_memory = [torch.zeros(batch_size, self.persistent_memory_size, device=device) 
                           for _ in range(self.num_layers)]
        return hidden, persistent_memory
    
    def reset_state(self):
        """Reset the internal state."""
        self.state = None
    
    def update_from_reward(self, reward):
        """
        Update the controller based on reward signals.
        
        This method allows for explicit reward-based updates outside of the
        forward pass, enabling reinforcement learning-like behavior.
        
        Args:
            reward (torch.Tensor): Reward signal
            
        Returns:
            bool: True if update was successful
        """
        if self.state is None:
            return False
        
        hidden, persistent_memory = self.state
        
        # Apply reward-based updates to each layer
        for layer in range(self.num_layers):
            # Update connection strengths based on reward
            self.cells[layer].update_connections(hidden[layer], reward)
        
        return True
    
    def get_dynamic_connections(self):
        """
        Get the current dynamic connection strengths.
        
        This method returns the current connection strengths for analysis
        or visualization purposes.
        
        Returns:
            list: List of connection strength tensors for each layer
        """
        connection_strengths = []
        for layer in range(self.num_layers):
            connection_strengths.append(
                self.cells[layer].connection_strength * 
                torch.sigmoid(self.cells[layer].connection_history)
            )
        return connection_strengths
