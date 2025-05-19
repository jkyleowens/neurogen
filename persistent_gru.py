import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PersistentGRUController(nn.Module):
    """
    A multi-layer GRU controller with persistent memory for maintaining state over long periods.
    
    This implementation uses PersistentGRUCell as its building block and handles
    sequence processing with proper error checking and dimension compatibility.
    """
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, 
                 persistent_memory_size=128, dropout=0.1):
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
        
        # Input projection (if input_size != hidden_size)
        self.input_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
        
        # Create PersistentGRUCell layers
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = hidden_size
            self.cells.append(PersistentGRUCell(
                input_size=layer_input_size,
                hidden_size=hidden_size,
                persistent_memory_size=persistent_memory_size
            ))
        
        # Dropout layer (applied between layers and to output)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, output_size)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters using orthogonal initialization for better gradient flow."""
        for name, param in self.named_parameters():
            if 'weight' in name and 'cells' not in name:  # Cell params already initialized
                nn.init.orthogonal_(param)
            elif 'bias' in name and 'cells' not in name:
                nn.init.zeros_(param)
    
    def init_hidden(self, batch_size, device):
        """
        Initialize hidden state and persistent memory.
        
        Args:
            batch_size (int): Batch size
            device (torch.device): Device to create tensors on
            
        Returns:
            tuple: (hidden_state, persistent_memory)
        """
        # Initialize hidden state for each layer
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        
        # Initialize persistent memory for each layer
        persistent_memory = torch.zeros(self.num_layers, batch_size, self.persistent_memory_size, device=device)
        
        return hidden, persistent_memory
    
    def forward(self, x, hidden=None, persistent_memory=None, neuromodulators=None):
        """
        Forward pass of the Persistent GRU Controller.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
            hidden (torch.Tensor, optional): Initial hidden state of shape (num_layers, batch_size, hidden_size)
            persistent_memory (torch.Tensor, optional): Initial persistent memory
                of shape (num_layers, batch_size, persistent_memory_size)
            neuromodulators (dict, optional): Dictionary of neuromodulator levels
            
        Returns:
            tuple: (outputs, hidden_states, persistent_memories)
        """
        # Get dimensions
        try:
            batch_size, seq_length, input_dim = x.size()
        except ValueError:
            # Handle case where input is 2D (batch_size, input_size)
            batch_size, input_dim = x.size()
            seq_length = 1
            x = x.unsqueeze(1)  # Add sequence dimension
        
        device = x.device
        
        # Check input dimensions and adapt if needed
        if input_dim != self.input_size:
            print(f"Input dimension mismatch. Expected {self.input_size}, got {input_dim}. Adapting.")
            # Create a temporary projection layer
            temp_proj = nn.Linear(input_dim, self.input_size).to(device)
            nn.init.xavier_uniform_(temp_proj.weight)
            nn.init.zeros_(temp_proj.bias)
            
            # Apply projection
            x_reshaped = x.reshape(-1, input_dim)
            x_projected = temp_proj(x_reshaped)
            x = x_projected.reshape(batch_size, seq_length, self.input_size)
        
        # Initialize states if not provided
        if hidden is None or persistent_memory is None:
            hidden, persistent_memory = self.init_hidden(batch_size, device)
        
        # Convert flat hidden/memory to layered format if needed
        if hidden.dim() == 2:  # (batch_size, hidden_size)
            hidden = hidden.unsqueeze(0).expand(self.num_layers, batch_size, self.hidden_size)
        
        if persistent_memory.dim() == 2:  # (batch_size, persistent_memory_size)
            persistent_memory = persistent_memory.unsqueeze(0).expand(
                self.num_layers, batch_size, self.persistent_memory_size
            )
        
        # Pre-allocate output tensors
        outputs = torch.zeros(batch_size, seq_length, self.output_size, device=device)
        
        # Store all states for each layer and time step
        hidden_states = []
        persistent_memories = []
        
        # Process each time step
        for t in range(seq_length):
            # Input at current time step
            x_t = x[:, t]
            
            # Apply input projection if needed
            if self.input_proj is not None:
                x_t = self.input_proj(x_t)
            
            # Store states for this time step
            layer_hidden_states = []
            layer_memory_states = []
            
            # Process through layers
            for layer in range(self.num_layers):
                # Get hidden and memory for this layer
                h_t = hidden[layer]
                m_t = persistent_memory[layer]
                
                # Apply dropout to input except for first layer
                if layer > 0 and self.dropout_layer is not None:
                    x_t = self.dropout_layer(x_t)
                
                # Process through GRU cell
                h_t, m_t = self.cells[layer](x_t, h_t, m_t, neuromodulators)
                
                # Set input for next layer
                x_t = h_t
                
                # Store updated states
                layer_hidden_states.append(h_t)
                layer_memory_states.append(m_t)
            
            # Update hidden and persistent memory
            hidden = torch.stack(layer_hidden_states)
            persistent_memory = torch.stack(layer_memory_states)
            
            # Store states for this time step
            hidden_states.append(hidden.clone())
            persistent_memories.append(persistent_memory.clone())
            
            # Generate output for this time step (using the final layer's hidden state)
            if self.dropout_layer is not None:
                output_t = self.output_proj(self.dropout_layer(hidden[-1]))
            else:
                output_t = self.output_proj(hidden[-1])
            
            outputs[:, t] = output_t
        
        # Stack states along sequence dimension
        hidden_states = torch.stack(hidden_states, dim=1)  # (num_layers, seq_length, batch_size, hidden_size)
        persistent_memories = torch.stack(persistent_memories, dim=1)  # (num_layers, seq_length, batch_size, persistent_memory_size)
        
        # Reorder dimensions to (batch_size, seq_length, num_layers, *)
        hidden_states = hidden_states.permute(2, 1, 0, 3)
        persistent_memories = persistent_memories.permute(2, 1, 0, 3)
        
        return outputs, hidden_states, persistent_memories
    
    def get_last_states(self, hidden_states, persistent_memories):
        """
        Get the last hidden and memory states for each layer.
        
        Args:
            hidden_states (torch.Tensor): Hidden states of shape (batch_size, seq_length, num_layers, hidden_size)
            persistent_memories (torch.Tensor): Persistent memories
                of shape (batch_size, seq_length, num_layers, persistent_memory_size)
            
        Returns:
            tuple: (last_hidden, last_memory)
        """
        last_hidden = hidden_states[:, -1].transpose(0, 1)  # (num_layers, batch_size, hidden_size)
        last_memory = persistent_memories[:, -1].transpose(0, 1)  # (num_layers, batch_size, persistent_memory_size)
        
        return last_hidden, last_memory

class PersistentGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, persistent_memory_size, bias=True):
        super(PersistentGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.persistent_memory_size = persistent_memory_size
        self.bias = bias
        
        # GRU gates
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.new_gate = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        
        # Persistent memory components
        self.memory_read_gate = nn.Linear(hidden_size, persistent_memory_size, bias=bias)
        self.memory_write_gate = nn.Linear(hidden_size, persistent_memory_size, bias=bias)
        self.memory_update = nn.Linear(hidden_size, persistent_memory_size, bias=bias)
        self.memory_to_hidden = nn.Linear(persistent_memory_size, hidden_size, bias=bias)
        
        # Neuromodulation gates - for affecting the memory based on reward signals
        self.dopamine_gate = nn.Linear(hidden_size, persistent_memory_size, bias=bias)
        self.serotonin_gate = nn.Linear(hidden_size, persistent_memory_size, bias=bias)
        
        # Initialize parameters for better gradient flow
        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters using techniques for better training of RNNs."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget bias to 1.0 as is common practice for LSTMs/GRUs
                if 'update_gate.bias' in name:
                    param.data.fill_(1.0)

    def forward(self, x, hidden=None, persistent_memory=None, neuromodulators=None):
        """
        Forward pass for a single time step of the Persistent GRU Cell.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            hidden (torch.Tensor, optional): Hidden state tensor of shape (batch_size, hidden_size)
            persistent_memory (torch.Tensor, optional): Persistent memory tensor 
                                                    of shape (batch_size, persistent_memory_size)
            neuromodulators (dict, optional): Dictionary of neuromodulator levels
                                            with keys 'dopamine', 'serotonin', etc.
            
        Returns:
            tuple: (new_hidden, new_persistent_memory)
        """
        # Get batch size from input
        batch_size = x.size(0)
        device = x.device
        
        # Initialize states if not provided
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=device)
        
        if persistent_memory is None:
            persistent_memory = torch.zeros(batch_size, self.persistent_memory_size, device=device)
        
        # Check input dimensions and adapt if necessary
        if x.size(1) != self.input_size:
            # Create an adaptive projection
            adaptive_proj = nn.Linear(x.size(1), self.input_size).to(device)
            nn.init.xavier_uniform_(adaptive_proj.weight)
            nn.init.zeros_(adaptive_proj.bias)
            x = adaptive_proj(x)
        
        # Combine input and hidden state for gate calculations
        combined = torch.cat([x, hidden], dim=1)
        
        # GRU computations
        reset = torch.sigmoid(self.reset_gate(combined))
        update = torch.sigmoid(self.update_gate(combined))
        combined_reset = torch.cat([x, reset * hidden], dim=1)
        new_hidden = torch.tanh(self.new_gate(combined_reset))
        
        # Update hidden state (standard GRU formula)
        hidden_candidate = (1 - update) * hidden + update * new_hidden
        
        # Interact with persistent memory
        # Compute read gate (how much to read from memory)
        read_gate = torch.sigmoid(self.memory_read_gate(hidden_candidate))
        
        # Read from memory
        memory_output = read_gate * persistent_memory
        
        # Project memory output to hidden state dimensions
        memory_impact = self.memory_to_hidden(memory_output)
        
        # Integrate memory impact into hidden state
        hidden_with_memory = hidden_candidate + memory_impact
        
        # Compute write gate (how much to update memory)
        write_gate = torch.sigmoid(self.memory_write_gate(hidden_with_memory))
        
        # Create new memory content
        new_memory_content = torch.tanh(self.memory_update(hidden_with_memory))
        
        # Apply neuromodulatory effects if provided
        if neuromodulators is not None:
            # Dopamine effect: modulates reward-based memory updates
            if 'dopamine' in neuromodulators:
                dopamine = neuromodulators['dopamine']
                dopamine_effect = torch.sigmoid(self.dopamine_gate(hidden_with_memory) * dopamine)
                write_gate = write_gate * dopamine_effect
            
            # Serotonin effect: modulates persistence of existing memories
            if 'serotonin' in neuromodulators:
                serotonin = neuromodulators['serotonin']
                serotonin_effect = torch.sigmoid(self.serotonin_gate(hidden_with_memory) * serotonin)
                # Higher serotonin = higher persistence of existing memories
                write_gate = write_gate * (1 - serotonin_effect)
        
        # Update persistent memory
        new_persistent_memory = (1 - write_gate) * persistent_memory + write_gate * new_memory_content
        
        return hidden_with_memory, new_persistent_memory

    def extra_repr(self):
        """Return a string with extra information."""
        return f'input_size={self.input_size}, hidden_size={self.hidden_size}, persistent_memory_size={self.persistent_memory_size}'
