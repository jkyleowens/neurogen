
"""
Input Preprocessing Module

This module implements a biologically-inspired input preprocessing system that automatically
detects and preprocesses different types of input data before sending it to the controller.

The implementation draws inspiration from biological sensory processing mechanisms including:
1. Feature extraction similar to how the visual cortex processes visual information
2. Adaptive normalization similar to how sensory neurons adjust their sensitivity
3. Multi-modal integration similar to how the brain combines different sensory inputs
4. Attention mechanisms similar to how the brain selectively focuses on relevant information
5. Online learning and adaptation similar to how sensory systems adapt to changing environments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import re
import math


class FeatureExtractor(nn.Module):
    """
    A biologically-inspired feature extraction module that extracts relevant features
    from different types of input data.
    
    Biological inspiration:
    - Hierarchical processing similar to the visual cortex's processing pathway
    - Receptive fields that respond to specific patterns in the input
    - Feature detectors that extract increasingly complex features
    - Parallel processing pathways for different aspects of the input
    """
    
    def __init__(self, input_size, feature_size, num_filters=32, kernel_sizes=[3, 5, 7]):
        """
        Initialize the Feature Extractor.
        
        Args:
            input_size (int): Size of the input features
            feature_size (int): Size of the output features
            num_filters (int): Number of convolutional filters
            kernel_sizes (list): List of kernel sizes for different receptive fields
        """
        super(FeatureExtractor, self).__init__()
        
        self.input_size = input_size
        self.feature_size = feature_size
        
        # Convolutional layers for spatial feature extraction
        # Inspired by simple and complex cells in the visual cortex
        self.conv_layers = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            self.conv_layers.append(
                nn.Conv1d(1, num_filters, kernel_size, padding=padding)
            )
        
        # Recurrent layer for temporal feature extraction
        # Inspired by how the brain processes temporal patterns
        self.rnn = nn.GRU(
            input_size=num_filters * len(kernel_sizes),
            hidden_size=feature_size,
            batch_first=True
        )
        
        # Attention mechanism for focusing on relevant features
        # Inspired by attentional mechanisms in the brain
        self.attention = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.Tanh(),
            nn.Linear(feature_size, 1)
        )
        
        # Feature integration layer
        # Inspired by how the brain integrates information from different sources
        self.feature_integration = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, feature_size)
        )
        
        # Lateral inhibition mechanism
        # Inspired by lateral inhibition in sensory systems
        self.lateral_inhibition = nn.Parameter(torch.ones(feature_size, feature_size) * 0.1)
        
        # Initialize parameters
        self._init_parameters()
        
        # Feature history for adaptation
        self.register_buffer('feature_history', torch.zeros(feature_size))
        self.register_buffer('feature_variance', torch.ones(feature_size))
        
    def _init_parameters(self):
        """Initialize parameters with biologically-inspired distributions."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'conv' in name:
                    # Initialize convolutional weights to detect edges and patterns
                    nn.init.kaiming_normal_(param, nonlinearity='relu')
                elif 'rnn' in name:
                    # Initialize RNN weights for stable temporal processing
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, hidden=None):
        """
        Forward pass of the Feature Extractor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
            hidden (torch.Tensor, optional): Initial hidden state for RNN
            
        Returns:
            tuple: (features, attention_weights)
        """
        batch_size, seq_length, _ = x.size()
        
        # Process each time step
        all_features = []
        attention_weights_seq = []
        
        for t in range(seq_length):
            # Get input for current time step
            x_t = x[:, t, :].unsqueeze(1)  # Add channel dimension for conv
            
            # Apply convolutional layers (spatial feature extraction)
            conv_outputs = []
            for conv_layer in self.conv_layers:
                conv_out = F.relu(conv_layer(x_t))
                # Global pooling to get the most salient features
                pooled = F.adaptive_max_pool1d(conv_out, 1).squeeze(-1)
                conv_outputs.append(pooled)
            
            # Combine convolutional outputs
            combined_features = torch.cat(conv_outputs, dim=1).unsqueeze(1)  # Add sequence dimension
            
            # Apply RNN for temporal feature extraction
            if hidden is None:
                rnn_out, hidden = self.rnn(combined_features)
            else:
                rnn_out, hidden = self.rnn(combined_features, hidden)
            
            # Apply attention mechanism
            attention_scores = self.attention(rnn_out).squeeze(-1)
            attention_weights = F.softmax(attention_scores, dim=1)
            attended_features = torch.sum(rnn_out * attention_weights.unsqueeze(-1), dim=1)
            
            # Apply lateral inhibition (competitive feature selection)
            inhibition = torch.matmul(attended_features, self.lateral_inhibition)
            inhibited_features = attended_features - F.relu(inhibition)
            
            # Apply feature integration
            integrated_features = self.feature_integration(inhibited_features)
            
            # Store features and attention weights
            all_features.append(integrated_features)
            attention_weights_seq.append(attention_weights)
        
        # Stack features and attention weights along sequence dimension
        features = torch.stack(all_features, dim=1)
        attention_weights = torch.cat(attention_weights_seq, dim=1)
        
        # Update feature history for adaptation
        with torch.no_grad():
            current_mean = features.mean(dim=(0, 1))
            current_var = features.var(dim=(0, 1))
            self.feature_history = self.feature_history * 0.9 + current_mean * 0.1
            self.feature_variance = self.feature_variance * 0.9 + current_var * 0.1
        
        return features, attention_weights
    
    def adapt(self, features):
        """
        Adapt feature extraction based on recent inputs.
        
        This mimics how sensory systems adapt to the statistics of their inputs.
        
        Args:
            features (torch.Tensor): Recent features
            
        Returns:
            torch.Tensor: Adapted features
        """
        # Compute feature statistics
        feature_mean = features.mean(dim=(0, 1), keepdim=True)
        feature_std = features.std(dim=(0, 1), keepdim=True) + 1e-5
        
        # Adapt features using z-score normalization
        adapted_features = (features - feature_mean) / feature_std
        
        return adapted_features


class DataNormalizer(nn.Module):
    """
    A biologically-inspired data normalization module that adaptively normalizes
    different types of input data.
    
    Biological inspiration:
    - Adaptive gain control similar to how sensory neurons adjust their sensitivity
    - Contrast normalization similar to how the visual system enhances contrast
    - Dynamic range adaptation similar to how sensory systems adapt to input statistics
    - Homeostatic regulation similar to how neural circuits maintain stability
    """
    
    def __init__(self, input_size, adaptation_rate=0.1, epsilon=1e-5):
        """
        Initialize the Data Normalizer.
        
        Args:
            input_size (int): Size of the input features
            adaptation_rate (float): Rate at which normalization adapts to new data
            epsilon (float): Small constant for numerical stability
        """
        super(DataNormalizer, self).__init__()
        
        self.input_size = input_size
        self.adaptation_rate = adaptation_rate
        self.epsilon = epsilon
        
        # Running statistics for normalization
        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))
        self.register_buffer('running_min', torch.ones(input_size) * float('inf'))
        self.register_buffer('running_max', torch.ones(input_size) * float('-inf'))
        
        # Learnable parameters for adaptive normalization
        self.gain = nn.Parameter(torch.ones(input_size))
        self.bias = nn.Parameter(torch.zeros(input_size))
        
        # Adaptation state
        self.register_buffer('adaptation_level', torch.zeros(input_size))
        
    def forward(self, x, update_stats=True):
        """
        Forward pass of the Data Normalizer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
            update_stats (bool): Whether to update running statistics
            
        Returns:
            torch.Tensor: Normalized input
        """
        batch_size, seq_length, _ = x.size()
        
        # Compute batch statistics
        batch_mean = x.mean(dim=(0, 1))
        batch_var = x.var(dim=(0, 1), unbiased=False)
        batch_min = x.min(dim=0)[0].min(dim=0)[0]
        batch_max = x.max(dim=0)[0].max(dim=0)[0]
        
        # Update running statistics
        if update_stats:
            self.running_mean = self.running_mean * (1 - self.adaptation_rate) + batch_mean * self.adaptation_rate
            self.running_var = self.running_var * (1 - self.adaptation_rate) + batch_var * self.adaptation_rate
            self.running_min = torch.min(self.running_min, batch_min)
            self.running_max = torch.max(self.running_max, batch_max)
        
        # Z-score normalization (adaptive gain control)
        z_normalized = (x - self.running_mean.view(1, 1, -1)) / (torch.sqrt(self.running_var.view(1, 1, -1)) + self.epsilon)
        
        # Min-max normalization (dynamic range adaptation)
        range_size = self.running_max.view(1, 1, -1) - self.running_min.view(1, 1, -1) + self.epsilon
        min_max_normalized = (x - self.running_min.view(1, 1, -1)) / range_size
        
        # Combine normalizations with learnable parameters
        normalized = self.gain.view(1, 1, -1) * z_normalized + self.bias.view(1, 1, -1)
        
        # Apply adaptive normalization based on input statistics
        # This mimics how sensory systems adapt to the statistics of their inputs
        adaptation_factor = torch.sigmoid(self.adaptation_level.view(1, 1, -1))
        final_output = adaptation_factor * normalized + (1 - adaptation_factor) * min_max_normalized
        
        # Update adaptation level based on input variance
        if update_stats:
            variance_ratio = batch_var / (self.running_var + self.epsilon)
            adaptation_update = torch.log(variance_ratio + self.epsilon)
            self.adaptation_level = self.adaptation_level * 0.9 + adaptation_update * 0.1
        
        return final_output
    
    def reset_stats(self):
        """Reset normalization statistics."""
        self.running_mean.fill_(0)
        self.running_var.fill_(1)
        self.running_min.fill_(float('inf'))
        self.running_max.fill_(float('-inf'))
        self.adaptation_level.fill_(0)


class SensoryEncoder(nn.Module):
    """
    A biologically-inspired sensory encoder that encodes different types of input data
    into a common representation.
    
    Biological inspiration:
    - Specialized sensory pathways similar to different sensory modalities in the brain
    - Multi-modal integration similar to how the brain combines different sensory inputs
    - Population coding similar to how the brain represents information in neural populations
    - Sparse coding similar to how the brain efficiently encodes information
    """
    
    def __init__(self, input_size, encoding_size, num_encoders=4, sparsity=0.1):
        """
        Initialize the Sensory Encoder.
        
        Args:
            input_size (int): Size of the input features
            encoding_size (int): Size of the encoded representation
            num_encoders (int): Number of specialized encoders
            sparsity (float): Target sparsity of the encoded representation
        """
        super(SensoryEncoder, self).__init__()
        
        self.input_size = input_size
        self.encoding_size = encoding_size
        self.num_encoders = num_encoders
        self.sparsity = sparsity
        
        # Specialized encoders for different input types
        # Inspired by specialized sensory pathways in the brain
        self.encoders = nn.ModuleList()
        for _ in range(num_encoders):
            encoder = nn.Sequential(
                nn.Linear(input_size, encoding_size * 2),
                nn.ReLU(),
                nn.Linear(encoding_size * 2, encoding_size)
            )
            self.encoders.append(encoder)
        
        # Encoder selection network
        # Inspired by attentional selection mechanisms in the brain
        self.encoder_selection = nn.Sequential(
            nn.Linear(input_size, num_encoders),
            nn.Softmax(dim=-1)
        )
        
        # Population coding layer
        # Inspired by population coding in neural systems
        self.population_coding = nn.Linear(encoding_size, encoding_size)
        
        # Sparsity control
        # Inspired by sparse coding in neural systems
        self.sparsity_control = nn.Parameter(torch.ones(1) * math.log(sparsity / (1 - sparsity)))
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters with biologically-inspired distributions."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        """
        Forward pass of the Sensory Encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            tuple: (encoded, encoder_weights)
        """
        batch_size, seq_length, _ = x.size()
        
        # Process each time step
        encoded_seq = []
        encoder_weights_seq = []
        
        for t in range(seq_length):
            # Get input for current time step
            x_t = x[:, t, :]
            
            # Compute encoder selection weights
            encoder_weights = self.encoder_selection(x_t)
            
            # Apply each specialized encoder
            encoded_all = []
            for i, encoder in enumerate(self.encoders):
                encoded_i = encoder(x_t)
                encoded_all.append(encoded_i)
            
            # Stack encoded representations
            encoded_stack = torch.stack(encoded_all, dim=1)  # (batch_size, num_encoders, encoding_size)
            
            # Weight encoded representations by encoder selection
            weighted_encoding = torch.sum(encoded_stack * encoder_weights.unsqueeze(-1), dim=1)
            
            # Apply population coding
            population_encoded = self.population_coding(weighted_encoding)
            
            # Apply sparsity control using k-winners-take-all
            # This mimics lateral inhibition in neural circuits
            sparsity_threshold = torch.sigmoid(self.sparsity_control)
            k = max(1, int(self.encoding_size * sparsity_threshold))
            topk_values, topk_indices = torch.topk(population_encoded, k, dim=1)
            sparse_encoded = torch.zeros_like(population_encoded)
            sparse_encoded.scatter_(1, topk_indices, topk_values)
            
            # Store encoded representation and encoder weights
            encoded_seq.append(sparse_encoded)
            encoder_weights_seq.append(encoder_weights)
        
        # Stack along sequence dimension
        encoded = torch.stack(encoded_seq, dim=1)
        encoder_weights = torch.stack(encoder_weights_seq, dim=1)
        
        return encoded, encoder_weights
    
    def get_sparsity(self):
        """Get the current sparsity level."""
        return torch.sigmoid(self.sparsity_control).item()


class InputProcessor(nn.Module):
    """
    A biologically-inspired input processor that automatically detects and preprocesses
    different types of input data before sending it to the controller.
    
    This module integrates feature extraction, data normalization, and sensory encoding
    to preprocess input data in a way that mimics how the brain processes sensory information.
    
    Biological inspiration:
    - Hierarchical processing similar to sensory pathways in the brain
    - Adaptive normalization similar to sensory adaptation mechanisms
    - Multi-modal integration similar to how the brain combines different sensory inputs
    - Online learning and adaptation similar to how sensory systems adapt to changing environments
    """
    
    def __init__(self, input_size, output_size, hidden_size=128, feature_size=64, 
                 encoding_size=128, num_encoders=4, adaptation_rate=0.1):
        """
        Initialize the Input Processor.
        
        Args:
            input_size (int): Size of the input features
            output_size (int): Size of the output features (to match controller input)
            hidden_size (int): Size of the hidden state
            feature_size (int): Size of the extracted features
            encoding_size (int): Size of the encoded representation
            num_encoders (int): Number of specialized encoders
            adaptation_rate (float): Rate at which the processor adapts to new data
        """
        super(InputProcessor, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.encoding_size = encoding_size
        self.adaptation_rate = adaptation_rate
        
        # Input type detection
        # Inspired by how the brain automatically routes information to appropriate processing areas
        self.type_detector = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4)  # 4 types: numerical, categorical, text, mixed
        )
        
        # Data normalizer
        self.normalizer = DataNormalizer(input_size, adaptation_rate)
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(input_size, feature_size)
        
        # Sensory encoder
        self.encoder = SensoryEncoder(feature_size, encoding_size, num_encoders)
        
        # Output projection to match controller input size
        self.output_projection = nn.Linear(encoding_size, output_size)
        
        # Online learning components
        # Inspired by how the brain continuously adapts to new information
        self.online_learning_rate = nn.Parameter(torch.ones(1) * 0.01)
        
        # Memory buffer for online learning
        self.memory_buffer = deque(maxlen=100)
        
        # Input statistics tracking for adaptation
        self.register_buffer('input_mean', torch.zeros(input_size))
        self.register_buffer('input_var', torch.ones(input_size))
        self.register_buffer('categorical_mask', torch.zeros(input_size))
        self.register_buffer('numerical_mask', torch.zeros(input_size))
        self.register_buffer('text_mask', torch.zeros(input_size))
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters with biologically-inspired distributions."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def detect_input_type(self, x):
        """
        Detect the type of input data.
        
        This mimics how the brain automatically routes information to appropriate processing areas.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            tuple: (type_probs, type_masks)
        """
        batch_size, seq_length, _ = x.size()
        
        # Compute statistics for type detection
        means = x.mean(dim=(0, 1))
        vars = x.var(dim=(0, 1))
        unique_values = []
        
        for i in range(self.input_size):
            unique = torch.unique(x[:, :, i])
            unique_values.append(len(unique))
        
        # Create feature vector for type detection
        type_features = []
        for i in range(self.input_size):
            # Numerical features typically have high variance and many unique values
            numerical_score = vars[i] * math.log(unique_values[i] + 1)
            
            # Categorical features typically have low variance and few unique values
            categorical_score = 1.0 / (vars[i] + 1e-5) * (1.0 / (unique_values[i] + 1))
            
            # Text features typically have high variance and many unique values in a specific range
            text_score = vars[i] * (unique_values[i] > 10) * (means[i] > 0)
            
            type_features.extend([numerical_score, categorical_score, text_score])
        
        # Convert to tensor
        type_features = torch.tensor(type_features, device=x.device).float()
        
        # Detect input type using neural network
        type_logits = self.type_detector(type_features.unsqueeze(0))
        type_probs = F.softmax(type_logits, dim=1).squeeze(0)
        
        # Create masks for different input types
        numerical_mask = (type_probs[0] > 0.3).float()
        categorical_mask = (type_probs[1] > 0.3).float()
        text_mask = (type_probs[2] > 0.3).float()
        mixed_mask = (type_probs[3] > 0.3).float()
        
        # Update masks in buffer
        self.numerical_mask = self.numerical_mask * 0.9 + numerical_mask * 0.1
        self.categorical_mask = self.categorical_mask * 0.9 + categorical_mask * 0.1
        self.text_mask = self.text_mask * 0.9 + text_mask * 0.1
        
        # Combine masks
        type_masks = {
            'numerical': self.numerical_mask,
            'categorical': self.categorical_mask,
            'text': self.text_mask,
            'mixed': mixed_mask
        }
        
        return type_probs, type_masks
    
    def preprocess_numerical(self, x, mask):
        """
        Preprocess numerical data.
        
        Args:
            x (torch.Tensor): Input tensor
            mask (torch.Tensor): Mask indicating numerical features
            
        Returns:
            torch.Tensor: Preprocessed numerical data
        """
        # Apply mask to select numerical features
        numerical_x = x * mask.view(1, 1, -1)
        
        # Normalize numerical features
        normalized_x = self.normalizer(numerical_x)
        
        return normalized_x
    
    def preprocess_categorical(self, x, mask):
        """
        Preprocess categorical data.
        
        Args:
            x (torch.Tensor): Input tensor
            mask (torch.Tensor): Mask indicating categorical features
            
        Returns:
            torch.Tensor: Preprocessed categorical data
        """
        batch_size, seq_length, _ = x.size()
        
        # Apply mask to select categorical features
        categorical_x = x * mask.view(1, 1, -1)
        
        # One-hot encode categorical features
        # This is a simplified version - in practice, would need to handle variable categories
        encoded_x = categorical_x
        
        return encoded_x
    
    def preprocess_text(self, x, mask):
        """
        Preprocess text data.
        
        Args:
            x (torch.Tensor): Input tensor
            mask (torch.Tensor): Mask indicating text features
            
        Returns:
            torch.Tensor: Preprocessed text data
        """
        # Apply mask to select text features
        text_x = x * mask.view(1, 1, -1)
        
        # Normalize text features
        normalized_x = self.normalizer(text_x, update_stats=False)
        
        return normalized_x
    
    def forward(self, x, adapt=True):
        """
        Forward pass of the Input Processor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
            adapt (bool): Whether to adapt to the input data
            
        Returns:
            tuple: (processed_output, metadata)
        """
        batch_size, seq_length, _ = x.size()
        
        # Detect input type
        type_probs, type_masks = self.detect_input_type(x)
        
        # Preprocess different types of data
        numerical_x = self.preprocess_numerical(x, type_masks['numerical'])
        categorical_x = self.preprocess_categorical(x, type_masks['categorical'])
        text_x = self.preprocess_text(x, type_masks['text'])
        
        # Combine preprocessed data
        # Weight by type probabilities to focus on the most relevant preprocessing
        combined_x = (
            numerical_x * type_probs[0] +
            categorical_x * type_probs[1] +
            text_x * type_probs[2] +
            x * type_probs[3]  # Mixed type uses raw input
        )
        
        # Extract features
        features, attention_weights = self.feature_extractor(combined_x)
        
        # Adapt feature extraction if requested
        if adapt:
            features = self.feature_extractor.adapt(features)
        
        # Encode features
        encoded, encoder_weights = self.encoder(features)
        
        # Project to output size
        output = self.output_projection(encoded)
        
        # Store in memory buffer for online learning
        if adapt:
            self.memory_buffer.append((x.detach(), output.detach()))
            
            # Update input statistics
            self.input_mean = self.input_mean * (1 - self.adaptation_rate) + x.mean(dim=(0, 1)) * self.adaptation_rate
            self.input_var = self.input_var * (1 - self.adaptation_rate) + x.var(dim=(0, 1)) * self.adaptation_rate
        
        # Collect metadata
        metadata = {
            'type_probs': type_probs,
            'attention_weights': attention_weights,
            'encoder_weights': encoder_weights,
            'features': features
        }
        
        return output, metadata
    
    def adapt_to_feedback(self, output, target, learning_rate=None):
        """
        Adapt the processor based on feedback.
        
        This mimics how the brain adapts based on feedback signals.
        
        Args:
            output (torch.Tensor): Output tensor
            target (torch.Tensor): Target tensor
            learning_rate (float, optional): Learning rate for adaptation
            
        Returns:
            float: Adaptation loss
        """
        if learning_rate is None:
            learning_rate = self.online_learning_rate.item()
        
        # Compute adaptation loss
        loss = F.mse_loss(output, target)
        
        # Update parameters using simple SGD
        for name, param in self.named_parameters():
            if param.requires_grad:
                grad = torch.autograd.grad(loss, param, retain_graph=True)[0]
                param.data -= learning_rate * grad
        
        return loss.item()
    
    def online_learning_step(self):
        """
        Perform an online learning step using the memory buffer.
        
        This mimics how the brain continuously learns from experience.
        
        Returns:
            float: Online learning loss
        """
        if len(self.memory_buffer) == 0:
            return 0.0
        
        # Sample a batch from memory buffer
        indices = np.random.choice(len(self.memory_buffer), min(10, len(self.memory_buffer)), replace=False)
        batch = [self.memory_buffer[i] for i in indices]
        inputs, outputs = zip(*batch)
        
        # Convert to tensors
        inputs = torch.cat([x for x in inputs], dim=0)
        outputs = torch.cat([x for x in outputs], dim=0)
        
        # Forward pass
        new_outputs, _ = self.forward(inputs, adapt=False)
        
        # Compute loss
        loss = F.mse_loss(new_outputs, outputs)
        
        # Update parameters
        learning_rate = self.online_learning_rate.item()
        for name, param in self.named_parameters():
            if param.requires_grad:
                grad = torch.autograd.grad(loss, param, retain_graph=True)[0]
                param.data -= learning_rate * grad
        
        return loss.item()
    
    def reset_adaptation(self):
        """Reset adaptation state."""
        self.normalizer.reset_stats()
        self.memory_buffer.clear()
        self.input_mean.fill_(0)
        self.input_var.fill_(1)
        self.categorical_mask.fill_(0)
        self.numerical_mask.fill_(0)
        self.text_mask.fill_(0)
    
    def integrate_with_controller(self, controller, x, hidden=None, persistent_memory=None, neuromodulators=None):
        """
        Integrate with the persistent GRU controller.
        
        Args:
            controller: PersistentGRUController instance
            x (torch.Tensor): Input tensor
            hidden (torch.Tensor, optional): Initial hidden state
            persistent_memory (torch.Tensor, optional): Initial persistent memory
            neuromodulators (dict, optional): Dictionary of neuromodulator levels
            
        Returns:
            tuple: Controller outputs
        """
        # Preprocess input
        processed_x, metadata = self.forward(x)
        
        # Pass to controller
        outputs, hidden_states, persistent_memories = controller(
            processed_x, hidden, persistent_memory, neuromodulators
        )
        
        return outputs, hidden_states, persistent_memories, metadata
    
    def integrate_with_neuromodulator(self, neuromodulator, x, hidden, reward=None):
        """
        Integrate with the neuromodulator system.
        
        Args:
            neuromodulator: NeuromodulatorSystem instance
            x (torch.Tensor): Input tensor
            hidden (torch.Tensor): Hidden state tensor
            reward (torch.Tensor, optional): Reward signal
            
        Returns:
            tuple: Neuromodulator outputs
        """
        # Preprocess input
        processed_x, metadata = self.forward(x)
        
        # Pass to neuromodulator
        modulated_hidden, neurotransmitter_levels = neuromodulator(
            processed_x, hidden, reward
        )
        
        return modulated_hidden, neurotransmitter_levels, metadata
    
    def integrate_with_brain_nn(self, brain_nn, x, hidden=None, persistent_memory=None, external_reward=None):
        """
        Integrate with the complete brain-inspired neural network.
        
        Args:
            brain_nn: BrainInspiredNN instance
            x (torch.Tensor): Input tensor
            hidden (torch.Tensor, optional): Initial hidden state
            persistent_memory (torch.Tensor, optional): Initial persistent memory
            external_reward (torch.Tensor, optional): External reward signal
            
        Returns:
            tuple: Model outputs
        """
        # Preprocess input
        processed_x, metadata = self.forward(x)
        
        # Pass to brain-inspired neural network
        outputs, predicted_rewards = brain_nn(
            processed_x, hidden, persistent_memory, external_reward
        )
        
        return outputs, predicted_rewards, metadata
