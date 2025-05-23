"""
CuPy implementation of basic neural network components

This module contains implementations of neural network components using only CuPy,
without relying on PyTorch. This is primarily for educational purposes to understand
how neural networks can be implemented directly with GPU arrays.

Warning: These implementations lack many optimizations present in PyTorch and
are not recommended for production use.
"""

import warnings
try:
    import cupy as cp
    USING_CUPY = True
except ImportError:
    import numpy as cp
    USING_CUPY = False
    warnings.warn("CuPy not available. Using NumPy instead.")

class CupyLinear:
    """
    Linear layer implementation using only CuPy.
    
    This is a basic implementation of a linear layer (fully connected) using
    only CuPy operations. It supports forward and backward passes for training.
    """
    
    def __init__(self, in_features, out_features):
        """
        Initialize a linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
        """
        # Initialize weights with Xavier/Glorot initialization
        scale = cp.sqrt(2.0 / (in_features + out_features))
        self.weight = cp.random.normal(0, scale, (out_features, in_features))
        self.bias = cp.zeros(out_features)
        
        # For backward pass
        self.x = None
        self.dweight = None
        self.dbias = None
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input data of shape (batch_size, in_features)
            
        Returns:
            Output of shape (batch_size, out_features)
        """
        self.x = x  # Store for backward pass
        return cp.dot(x, self.weight.T) + self.bias
    
    def backward(self, grad_output):
        """
        Backward pass.
        
        Args:
            grad_output: Gradient of loss with respect to output
                       of shape (batch_size, out_features)
            
        Returns:
            Gradient of loss with respect to input
        """
        batch_size = grad_output.shape[0]
        
        # Compute gradients
        self.dweight = cp.dot(grad_output.T, self.x) / batch_size
        self.dbias = cp.mean(grad_output, axis=0)
        
        # Compute gradient for input (to propagate backward)
        dx = cp.dot(grad_output, self.weight)
        
        return dx
    
    def update_parameters(self, learning_rate):
        """
        Update parameters using calculated gradients.
        
        Args:
            learning_rate: Learning rate for update
        """
        self.weight -= learning_rate * self.dweight
        self.bias -= learning_rate * self.dbias


class CupyReLU:
    """
    ReLU activation implementation using only CuPy.
    """
    
    def __init__(self):
        """Initialize the ReLU activation."""
        self.x = None
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input data
            
        Returns:
            Output after applying ReLU
        """
        self.x = x  # Store for backward pass
        return cp.maximum(0, x)
    
    def backward(self, grad_output):
        """
        Backward pass.
        
        Args:
            grad_output: Gradient of loss with respect to output
            
        Returns:
            Gradient of loss with respect to input
        """
        return grad_output * (self.x > 0)


class CupyMSELoss:
    """
    Mean Squared Error loss implementation using only CuPy.
    """
    
    def __init__(self):
        """Initialize the MSE loss."""
        self.y_pred = None
        self.y_true = None
    
    def forward(self, y_pred, y_true):
        """
        Forward pass.
        
        Args:
            y_pred: Predicted values
            y_true: Target values
            
        Returns:
            MSE loss
        """
        self.y_pred = y_pred
        self.y_true = y_true
        return cp.mean((y_pred - y_true) ** 2)
    
    def backward(self):
        """
        Backward pass.
        
        Returns:
            Gradient of loss with respect to predicted values
        """
        batch_size = self.y_pred.shape[0]
        return 2 * (self.y_pred - self.y_true) / batch_size


class CupySimpleNN:
    """
    A simple neural network implemented using only CuPy.
    """
    
    def __init__(self, layer_sizes):
        """
        Initialize a simple neural network.
        
        Args:
            layer_sizes: List of layer sizes [input_size, hidden_size1, ..., output_size]
        """
        self.layers = []
        self.activations = []
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(CupyLinear(layer_sizes[i], layer_sizes[i + 1]))
            
            # Add ReLU activation after all layers except the last one
            if i < len(layer_sizes) - 2:
                self.activations.append(CupyReLU())
            else:
                self.activations.append(None)  # No activation for output layer
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input data
            
        Returns:
            Output predictions
        """
        output = x
        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            output = layer.forward(output)
            if activation is not None:
                output = activation.forward(output)
        return output
    
    def backward(self, grad_output):
        """
        Backward pass through the network.
        
        Args:
            grad_output: Gradient of loss with respect to output
            
        Returns:
            Gradient of loss with respect to input
        """
        for i in range(len(self.layers) - 1, -1, -1):
            # Gradient through activation (if any)
            if self.activations[i] is not None:
                grad_output = self.activations[i].backward(grad_output)
            
            # Gradient through layer
            grad_output = self.layers[i].backward(grad_output)
        
        return grad_output
    
    def update_parameters(self, learning_rate):
        """
        Update all parameters in the network.
        
        Args:
            learning_rate: Learning rate for update
        """
        for layer in self.layers:
            layer.update_parameters(learning_rate)
    
    def train(self, x, y, learning_rate=0.01, epochs=100, batch_size=32):
        """
        Train the network.
        
        Args:
            x: Training data
            y: Target values
            learning_rate: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            List of training losses
        """
        loss_fn = CupyMSELoss()
        n_samples = x.shape[0]
        losses = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = cp.random.permutation(n_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            # Train in batches
            for i in range(0, n_samples, batch_size):
                x_batch = x_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # Forward pass
                y_pred = self.forward(x_batch)
                
                # Compute loss
                loss = loss_fn.forward(y_pred, y_batch)
                epoch_loss += loss
                
                # Backward pass
                grad_output = loss_fn.backward()
                self.backward(grad_output)
                
                # Update parameters
                self.update_parameters(learning_rate)
            
            # Print progress
            epoch_loss /= (n_samples / batch_size)
            losses.append(float(epoch_loss))
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")
        
        return losses


# Example usage
if __name__ == "__main__":
    # Generate some synthetic data
    n_samples = 1000
    x = cp.random.randn(n_samples, 5)
    # Create a synthetic target: y = 2*x1 + 3*x2 - x3 + 0.5*x4 + noise
    y = 2 * x[:, 0] + 3 * x[:, 1] - x[:, 2] + 0.5 * x[:, 3] + cp.random.randn(n_samples) * 0.1
    y = y.reshape(-1, 1)  # Shape for training
    
    # Create and train network
    model = CupySimpleNN([5, 10, 1])
    losses = model.train(x, y, learning_rate=0.01, epochs=100, batch_size=32)
    
    print("Training complete!")
