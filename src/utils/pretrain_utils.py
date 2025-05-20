"""
Pretraining Utilities for Neural Components

This module provides utilities to pretrain components of the brain-inspired neural network,
particularly the neuromodulator and controller components, ensuring they have valid
starting points before being used in feedback loops.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

def pretrain_controller(controller, dataloader, device, epochs=5, learning_rate=0.001):
    """
    Pretrain a controller component to predict the next state in a sequence.
    
    Args:
        controller: The controller component to pretrain
        dataloader: DataLoader containing sequence data
        device: Device to use for training
        epochs: Number of pretraining epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        pretrained controller
    """
    # Make a copy of the controller for pretraining
    controller.train()
    
    # Use MSE loss for sequence prediction
    criterion = nn.MSELoss()
    
    # Use Adam optimizer
    optimizer = optim.Adam(controller.parameters(), lr=learning_rate)
    
    # Track progress
    best_loss = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc=f'Pretraining Controller (Epoch {epoch+1}/{epochs})')):
            # Move data to device
            data = data.to(device)
            
            # Create targets from sequence (predict next value)
            if data.dim() == 3:  # [batch, seq_len, features]
                x = data[:, :-1]  # All but last sequence element
                y_target = data[:, 1:]  # All but first sequence element
            else:
                # Handle non-sequence data
                continue
                
            # Reset hidden state
            hidden = None
            
            # Forward pass
            output, hidden_dict = controller(x, hidden)
            
            # Calculate loss
            loss = criterion(output, y_target)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            batch_count += 1
        
        # Calculate average loss
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f'Epoch {epoch+1}, Avg Loss: {avg_loss:.6f}')
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f'New best controller (loss: {best_loss:.6f})')
    
    # Set to evaluation mode
    controller.eval()
    
    return controller


def pretrain_neuromodulator_components(model, dataloader, device, epochs=5, learning_rate=0.0005):
    """
    Pretrain the neuromodulator components to produce appropriate feedback signals.
    This trains the model to generate reward signals that correlate with actual prediction errors.
    
    Args:
        model: The brain-inspired model with neuromodulator components
        dataloader: DataLoader containing sequence data
        device: Device to use for training
        epochs: Number of pretraining epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        model with pretrained neuromodulator components
    """
    # Put model in training mode
    model.train()
    
    # Use MSE loss
    criterion = nn.MSELoss()
    
    # Create optimizer that only updates neuromodulator parameters
    # Identify neuromodulator parameters
    neuromodulator_params = []
    for name, param in model.named_parameters():
        if 'dopamine' in name or 'serotonin' in name or 'norepinephrine' in name or 'acetylcholine' in name:
            neuromodulator_params.append(param)
    
    # Create optimizer if we found parameters to train
    if neuromodulator_params:
        optimizer = optim.Adam(neuromodulator_params, lr=learning_rate)
    else:
        # If no explicit neuromodulator components, train the update_weights function indirectly
        # by allowing gradients to flow through the reward pathway
        class RewardPathwayOptimizer:
            def __init__(self, model, lr):
                self.model = model
                self.lr = lr
                self._last_rewards = []
                self._last_errors = []
            
            def zero_grad(self):
                # No explicit grads to zero
                pass
                
            def step(self):
                # Use correlation between rewards and errors to adjust scaling factors
                if len(self._last_rewards) > 10 and len(self._last_errors) > 10:
                    rewards = torch.tensor(self._last_rewards[-10:])
                    errors = torch.tensor(self._last_errors[-10:])
                    
                    # Calculate correlation
                    rewards_mean = rewards.mean()
                    errors_mean = errors.mean()
                    correlation = ((rewards - rewards_mean) * (errors - errors_mean)).sum() / \
                                 (torch.sqrt(((rewards - rewards_mean)**2).sum() * ((errors - errors_mean)**2).sum()) + 1e-8)
                    
                    # Adjust neurotransmitter scales based on correlation
                    # We want strong negative correlation (higher reward for lower error)
                    target_correlation = -0.8
                    adjustment = self.lr * (target_correlation - correlation.item())
                    
                    # Update scales
                    if hasattr(model, 'dopamine_scale'):
                        model.dopamine_scale += adjustment
                    if hasattr(model, 'serotonin_scale'):
                        model.serotonin_scale += adjustment * 0.5
                    if hasattr(model, 'norepinephrine_scale'):
                        model.norepinephrine_scale += adjustment * 0.3
                    if hasattr(model, 'acetylcholine_scale'):
                        model.acetylcholine_scale += adjustment * 0.2
                        
                    # Clear history
                    self._last_rewards = []
                    self._last_errors = []
            
            def track_reward_error(self, reward, error):
                self._last_rewards.append(reward)
                self._last_errors.append(error)
                
        optimizer = RewardPathwayOptimizer(model, learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc=f'Pretraining Neuromodulators (Epoch {epoch+1}/{epochs})')):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Reset model state
            model.reset_state()
            
            # Forward pass without reward signal
            output = model(data)
            
            # Calculate error (negative means good prediction)
            if output.shape != target.shape:
                # Handle dimension mismatches
                if output.dim() == 3 and target.dim() == 2:
                    output = output[:, -1, :]
                
                # Get common dimensions
                batch_size = min(output.size(0), target.size(0))
                feature_size = min(output.size(-1), target.size(-1))
                
                # Trim both tensors
                output = output[:batch_size, ..., :feature_size]
                target = target[:batch_size, ..., :feature_size]
            
            loss = criterion(output, target)
            error = loss.item()
            
            # Generate reward signal (negative of loss)
            reward = -error
            
            # Run model again with reward to update neuromodulator
            output = model(data, reward=torch.tensor(reward, device=device))
            
            # For custom optimizer, track reward-error relationship
            if isinstance(optimizer, RewardPathwayOptimizer):
                optimizer.track_reward_error(reward, error)
            else:
                # For standard optimizer, compute gradient through the reward pathway
                # by comparing the output before and after the reward-based update
                loss_after = criterion(output, target)
                
                # We want the second loss to be lower
                improvement_loss = torch.relu(loss_after - 0.9 * loss)
                
                # Backward and optimize
                optimizer.zero_grad()
                improvement_loss.backward()
                optimizer.step()
            
        # Print progress
        print(f'Epoch {epoch+1}/{epochs} complete')
    
    return model


def create_pretrain_dataloader(dataloader, batch_size=32):
    """
    Create a simplified dataloader for pretraining from an existing dataloader.
    This extracts a subset of the data for faster pretraining.
    
    Args:
        dataloader: Original DataLoader
        batch_size: Batch size for the pretraining dataloader
        
    Returns:
        DataLoader for pretraining
    """
    # Extract some batches from the original dataloader
    all_data = []
    all_targets = []
    
    # Grab a few batches
    for i, (data, target) in enumerate(dataloader):
        all_data.append(data.cpu())
        all_targets.append(target.cpu())
        if i >= 5:  # Limit to ~5 batches for pretraining
            break
    
    if not all_data:
        return dataloader  # Return original if empty
    
    # Combine data
    X = torch.cat(all_data, dim=0)
    y = torch.cat(all_targets, dim=0)
    
    # Create TensorDataset
    dataset = TensorDataset(X, y)
    
    # Create and return DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
