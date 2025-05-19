#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Brain-Inspired Neural Network: Training Script

This script trains the brain-inspired neural network model on financial data.

Usage:
    python train.py --config config/config.yaml
"""

import sys
import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import the model
from src.model import BrainInspiredNN  # Ensure correct import for the model
from src.utils.memory_utils import optimize_memory_usage, print_gpu_memory_status  # Ensure utility imports
from src.utils.performance_report import generate_performance_report  # Add performance report utility

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Brain-Inspired Neural Network')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_data(config):
    """
    Load and preprocess financial data.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # TODO: Implement data loading and preprocessing
    # For now, create dummy data
    batch_size = config['training']['batch_size']
    input_size = config['model']['input_size']
    output_size = config['model']['output_size']
    seq_length = config['data']['sequence_length']
    
    # Create dummy data
    X = torch.randn(1000, seq_length, input_size)
    y = torch.randn(1000, output_size)
    
    # Split data
    train_size = int(len(X) * config['data']['train_ratio'])
    val_size = int(len(X) * config['data']['val_ratio'])
    test_size = len(X) - train_size - val_size
    
    X_train, X_val, X_test = torch.split(X, [train_size, val_size, test_size])
    y_train, y_val, y_test = torch.split(y, [train_size, val_size, test_size])
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        criterion: Loss function
        device: Device to use for training
        
    Returns:
        float: Average training loss
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Training')):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Calculate loss
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
    
    # Return average loss
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: The neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to use for validation
        
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validation'):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Accumulate loss
            total_loss += loss.item()
    
    # Return average loss
    return total_loss / len(val_loader)

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, test_loader = load_data(config)
    
    # Create model
    model = BrainInspiredNN(config).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Optimize memory usage
        optimize_memory_usage(model, device)
        
        # Save checkpoint
        checkpoint_path = f"checkpoints/model_epoch_{epoch+1}.pt"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config
        }, checkpoint_path)
        
        print(f"Checkpoint saved to {checkpoint_path}")
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('loss_plot.png')
    
    print("Training completed!")

# Ensure the main function is complete and properly structured
if __name__ == "__main__":
    main()
