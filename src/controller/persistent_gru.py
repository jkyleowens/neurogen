"""
Persistent GRU Controller

This module implements a GRU-based controller with persistent memory
for the Brain-Inspired Neural Network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PersistentGRUController(nn.Module):
    """
    A GRU-based controller with persistent memory.
    
    This controller maintains a persistent memory state across
    forward passes, allowing for long-term memory retention.
    """
    
    def __init__(self, input_size, hidden_size, persistent_memory_size=64, 
                 num_layers=1, dropout=0.2, persistence_factor=0.9):
        """
        Initialize the Persistent GRU Controller.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden state
            persistent_memory_size (int): Size of persistent memory
            num_layers (int): Number of GRU layers
            dropout (float): Dropout rate
            persistence_factor (float): Factor controlling memory persistence
        """
        super(PersistentGRUController, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.persistent_memory_size = persistent_memory_size
        self.persistence_factor = persistence_factor
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Persistent memory projection
        self.memory_projection = nn.Linear(persistent_memory_size, hidden_size)
        self.hidden_projection = nn.Linear(hidden_size, persistent_memory_size)
        
        # Persistent memory
        self.persistent_memory = None
    def consolidate_memory(self, replay_samples=5):
        """
        Consolidate important patterns in persistent memory through replay.
        
        Args:
            replay_samples: Number of historical samples to replay
        """
        # Skip if no persistent memory exists
        if self.persistent_memory is None:
            return
        
        # Initialize temporary replay storage if not exists
        if not hasattr(self, 'input_history') or len(self.input_history) < replay_samples:
            return
        
        # Select replay samples (evenly distributed across history)
        indices = torch.linspace(0, len(self.input_history)-1, replay_samples).long()
        replay_inputs = [self.input_history[i] for i in indices]
        
        # Initialize consolidated memory if not exists
        if not hasattr(self, 'consolidated_memory'):
            self.consolidated_memory = torch.zeros_like(self.persistent_memory)
        
        # Process each replay sample with the hidden state
        original_memory = self.persistent_memory.clone()
        reduced_plasticity = 0.3  # Lower learning rate for replay
        
        with torch.no_grad():
            for replay_input in replay_inputs:
                # Process input without updating main memory
                projected_input = self.memory_projection(replay_input)
                
                # Compute importance of this pattern
                importance = torch.sigmoid(torch.mean(torch.abs(projected_input)))
                
                # Update consolidated memory with important features
                self.consolidated_memory = (
                    self.consolidated_memory * 0.9 + 
                    projected_input * 0.1 * importance
                )
            
            # Blend consolidated memory into persistent memory
            self.persistent_memory = (
                original_memory * 0.8 + 
                self.consolidated_memory * 0.2
            )
    
    def init_hidden(self, batch_size, device):
        """
        Initialize hidden state and persistent memory.
        
        Args:
            batch_size (int): Batch size
            device (torch.device): Device to create tensors on
            
        Returns:
            dict: Dictionary containing hidden state and persistent memory
        """
        # Initialize hidden state
        hidden = torch.zeros(
            self.num_layers, batch_size, self.hidden_size, 
            device=device
        )
        
        # Initialize persistent memory
        self.persistent_memory = torch.zeros(
            batch_size, self.persistent_memory_size,
            device=device
        )
        
        return {
            'hidden': hidden,
            'persistent_memory': self.persistent_memory
        }
    
    def forward(self, x, hidden_dict=None):
        """
        Forward pass through the controller.
        """
        # Initialize or adapt hidden state and persistent memory for current batch size
        if hidden_dict is None:
            hidden_dict = self.init_hidden(x.size(0), x.device)
        else:
            h = hidden_dict.get('hidden')
            # Reinit if batch size changed
            if h is None or h.size(1) != x.size(0):
                hidden_dict = self.init_hidden(x.size(0), x.device)
        hidden = hidden_dict['hidden']
        self.persistent_memory = hidden_dict['persistent_memory']

        # Apply GRU layer
        output, hidden_state = self.gru(x, hidden)
        
        # Update persistent memory if it exists
        if self.persistent_memory is not None:
            # Project hidden state to persistent memory space
            batch_size = x.size(0)
            last_hidden = hidden_state[-1]  # Get the last layer's hidden state
            
            # Project hidden state to persistent memory space
            memory_update = self.hidden_projection(last_hidden)
            
            # Update persistent memory with decay
            self.persistent_memory = (
                self.persistence_factor * self.persistent_memory +
                (1 - self.persistence_factor) * memory_update
            )
            
            # Project persistent memory to hidden space
            memory_influence = self.memory_projection(self.persistent_memory)
            
            # Blend output with memory influence
            output = output + 0.3 * memory_influence.unsqueeze(1).expand(-1, output.size(1), -1)
        
        # Return output and updated hidden state
        return output, {
            'hidden': hidden_state,
            'persistent_memory': self.persistent_memory
        }
