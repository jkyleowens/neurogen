"""
BioGRU Pretraining Utilities

This module provides specialized utilities for pretraining BioGRU components.
These utilities are designed to prepare BioGRU components for integration into
the full brain-inspired neural network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import sys
import os

# Ensure bio_gru is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def pretrain_biogru(bio_gru, dataloader, device, epochs=5, learning_rate=0.001):
    """
    Pretrain a BioGRU component using contrastive self-supervised learning.
    This helps the BioGRU develop meaningful representations and stable dynamics
    before being used in feedback-driven learning.
    
    Args:
        bio_gru: The BioGRU component to pretrain
        dataloader: DataLoader containing sequence data
        device: Device to use for training
        epochs: Number of pretraining epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        pretrained BioGRU component
    """
    # Make sure BioGRU is in training mode
    bio_gru.train()
    
    # Create optimizer
    optimizer = optim.Adam(bio_gru.parameters(), lr=learning_rate)
    
    # Temporal coherence loss for self-supervised learning
    def temporal_coherence_loss(seq_output):
        # Penalizes rapid changes in hidden states
        # This encourages temporally stable representations
        if seq_output.dim() < 3:
            return torch.tensor(0.0, device=device)
            
        # Calculate differences between consecutive time steps
        temp_diffs = seq_output[:, 1:] - seq_output[:, :-1]
        coherence_loss = torch.mean(torch.sum(temp_diffs**2, dim=2))
        return coherence_loss
    
    # Main training loop
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc=f'Pretraining BioGRU (Epoch {epoch+1}/{epochs})')):
            # Move data to device
            data = data.to(device)
            
            # Reset neuron states for each sequence
            bio_gru.reset_state()
            
            # Create prediction targets (next item in sequence)
            if data.dim() == 3:  # [batch, seq_len, features]
                x = data[:, :-1]  # All but last sequence element
                y_target = data[:, 1:]  # All but first sequence element
            else:
                # Handle non-sequence data by skipping
                continue
            
            # Forward pass with health tracking disabled
            with torch.set_grad_enabled(True):
                # Get sequence outputs
                outputs, hidden = bio_gru(x)
                
                # Calculate self-supervised losses:
                
                # 1. Next step prediction loss
                prediction_loss = nn.MSELoss()(outputs, y_target)
                
                # 2. Temporal coherence loss (stability in representations)
                coherence_loss = temporal_coherence_loss(outputs)
                
                # 3. Representation diversity loss (avoid neuron death)
                if outputs.dim() == 3:
                    # Calculate variance across batch and time dimensions
                    # Low variance means neurons are not being utilized diversely
                    neuron_means = outputs.mean(dim=(0, 1))
                    neuron_vars = ((outputs - neuron_means)**2).mean(dim=(0, 1))
                    # Penalize low variance
                    diversity_loss = torch.mean(torch.exp(-5 * neuron_vars))
                else:
                    diversity_loss = torch.tensor(0.0, device=device)
                
                # Combine losses
                loss = prediction_loss + 0.2 * coherence_loss + 0.1 * diversity_loss
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                # Clip gradients to prevent instability
                torch.nn.utils.clip_grad_norm_(bio_gru.parameters(), 2.0)
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                batch_count += 1
                
        # Print epoch summary
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f'Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}')
            
            # Check health of BioGRU neurons (if method available)
            if hasattr(bio_gru, 'get_health_report'):
                health_report = bio_gru.get_health_report()
                print(f"Neuron health: {health_report.get('overall_health', 'N/A')}")
                print(f"Active neurons: {health_report.get('active_neuron_percentage', 'N/A')}%")
    
    # Run a final health check and optimization
    if hasattr(bio_gru, 'optimize_neuron_pathways'):
        print("\nOptimizing neuron pathways...")
        bio_gru.optimize_neuron_pathways()
        
    # Reset neuron states for clean state
    bio_gru.reset_state()
    
    return bio_gru


def pretrain_biogru_feedback_mechanism(bio_gru, dataloader, device, epochs=3, learning_rate=0.0005):
    """
    Pretrain the feedback mechanism of BioGRU to respond appropriately to reward signals.
    This ensures the BioGRU adapts constructively to feedback during actual training.
    
    Args:
        bio_gru: The BioGRU component to pretrain
        dataloader: DataLoader containing sequence data
        device: Device to use for training
        epochs: Number of pretraining epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        pretrained BioGRU component with calibrated feedback response
    """
    # Make sure BioGRU is in training mode
    bio_gru.train()
    
    # For this pretraining, we'll focus on parameters related to feedback
    feedback_params = []
    for name, param in bio_gru.named_parameters():
        if any(key in name.lower() for key in ['neuromod', 'feedback', 'modulation', 'reward']):
            feedback_params.append(param)
    
    # If no feedback params found, use subset of general parameters
    if not feedback_params:
        for name, param in bio_gru.named_parameters():
            if any(key in name.lower() for key in ['bias', 'scale', 'gate']):
                feedback_params.append(param)
    
    # Create optimizer for feedback mechanism
    optimizer = optim.Adam(feedback_params, lr=learning_rate)
    
    # Keep track of best outcome
    best_improvement = 0.0
    best_state_dict = None
    
    # Main training loop
    for epoch in range(epochs):
        total_improvement = 0
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc=f'Pretraining BioGRU Feedback (Epoch {epoch+1}/{epochs})')):
            # Move data to device
            data = data.to(device)
            target = target.to(device)
            
            # Reset neuron states
            bio_gru.reset_state()
            
            # First forward pass without feedback
            with torch.no_grad():
                outputs_before, _ = bio_gru(data)
                
                # Format outputs for comparison
                if outputs_before.dim() == 3:
                    outputs_before = outputs_before[:, -1]  # Take last timestep
                
                # Calculate loss before feedback
                criterion = nn.MSELoss()
                if outputs_before.shape != target.shape:
                    # Handle shape mismatch
                    min_batch = min(outputs_before.shape[0], target.shape[0])
                    min_features = min(outputs_before.shape[-1], target.shape[-1] if target.dim() > 1 else 1)
                    
                    outputs_before = outputs_before[:min_batch, :min_features]
                    if target.dim() > 1:
                        target = target[:min_batch, :min_features]
                    else:
                        target = target[:min_batch].unsqueeze(-1)[:, :min_features]
                
                loss_before = criterion(outputs_before, target)
            
            # Create synthetic reward signal (negative loss)
            reward = -loss_before.item()
            
            # Apply reward manually (if method exists)
            if hasattr(bio_gru, 'process_reward_signal'):
                bio_gru.process_reward_signal(torch.tensor(reward, device=device))
            
            # Reset for new pass
            bio_gru.reset_state()
            
            # Second forward pass to measure impact of feedback
            outputs_after, _ = bio_gru(data)
            
            # Format outputs for comparison
            if outputs_after.dim() == 3:
                outputs_after = outputs_after[:, -1]  # Take last timestep
            
            # Handle shape mismatch again
            if outputs_after.shape != target.shape:
                outputs_after = outputs_after[:min_batch, :min_features]
            
            # Calculate loss after feedback
            loss_after = criterion(outputs_after, target)
            
            # Compute improvement
            improvement = loss_before - loss_after
            
            # We want to maximize improvement (feedback should make things better)
            # If improvement is negative, we need to adjust the feedback mechanism
            with torch.set_grad_enabled(True):
                # Loss function encourages positive improvement
                feedback_loss = torch.relu(0.1 - improvement)  # We want at least 0.1 improvement
                
                # If we got better, small reward. If we got worse, big penalty
                if improvement > 0:
                    feedback_loss *= 0.5
                else:
                    feedback_loss *= 2.0
                
                # Backward and optimize
                optimizer.zero_grad()
                feedback_loss.backward()
                torch.nn.utils.clip_grad_norm_(feedback_params, 1.0)
                optimizer.step()
            
            # Track metrics
            total_improvement += improvement.item()
            batch_count += 1
            
            # Clear memory
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate average improvement
        if batch_count > 0:
            avg_improvement = total_improvement / batch_count
            print(f'Epoch {epoch+1}/{epochs}, Avg Improvement: {avg_improvement:.6f}')
            
            # Save best state
            if avg_improvement > best_improvement:
                best_improvement = avg_improvement
                best_state_dict = {k: v.clone() for k, v in bio_gru.state_dict().items()
                                  if any(key in k for key in ['neuromod', 'feedback', 'modulation', 'reward', 
                                                              'bias', 'scale', 'gate'])}
    
    # Restore best state if we found one
    if best_state_dict and best_improvement > 0:
        print(f"Restoring best feedback mechanism state (improvement: {best_improvement:.6f})")
        # Only update the parameters we modified
        current_state = bio_gru.state_dict()
        for k, v in best_state_dict.items():
            if k in current_state:
                current_state[k] = v
        bio_gru.load_state_dict(current_state)
    
    # Reset state for clean state
    bio_gru.reset_state()
    
    return bio_gru
