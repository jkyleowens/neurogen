#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for comparing neuron optimization techniques.
This script creates two models: one with basic settings and one with optimized
neuron configurations, and compares their performance.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from bio_gru import BioGRU
from src.utils.neuron_analysis import visualize_neuron_health

def create_sample_data(batch_size=32, seq_length=10, input_size=5, output_size=1, num_batches=20):
    """Create sample data for testing"""
    dataset = []
    for _ in range(num_batches):
        # Create a sinusoidal pattern with noise
        x = torch.randn(batch_size, seq_length, input_size)
        y = torch.sin(x.sum(dim=2, keepdim=True) * 0.5) + torch.randn(batch_size, seq_length, 1) * 0.1
        dataset.append((x, y))
    return dataset

def train_model(model, dataset, epochs=5, learning_rate=0.01, validation_split=0.2):
    """
    Train a model and return training and validation metrics.
    
    Args:
        model: The BioGRU model to train
        dataset: List of (x, y) tuples for training
        epochs: Number of training epochs
        learning_rate: Base learning rate
        validation_split: Fraction of data to use for validation
        
    Returns:
        tuple: (train_losses, val_losses)
    """
    # Split into train and validation sets
    split_idx = int(len(dataset) * (1 - validation_split))
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]
    
    model.train()
    train_losses = []
    val_losses = []
    
    # MSE loss function
    criterion = torch.nn.MSELoss()
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_data, desc=f'Epoch {epoch+1}/{epochs}')):
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
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / len(train_data)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for data, target in val_data:
                model.reset_state()
                output, _ = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_data)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")
        
        # Reset to training mode
        model.train()
    
    return train_losses, val_losses

def setup_baseline_model(input_size, hidden_size, num_layers, output_size):
    """Create a baseline model with default settings"""
    return BioGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size
    )

def setup_optimized_model(input_size, hidden_size, num_layers, output_size):
    """Create an optimized model with enhanced neuron settings"""
    model = BioGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size
    )
    
    # Adjust neuron parameters
    for layer in model.layers:
        for i in range(layer.hidden_size):
            if layer.neuron_mask[i] > 0:
                # Set target activity for better balance
                for neuron in [layer.update_gate_neurons[i], layer.reset_gate_neurons[i], layer.candidate_neurons[i]]:
                    neuron.target_activity = 0.15  # 15% activation target
                    neuron.homeostatic_rate = 0.01  # Homeostatic adjustment rate
    
    # Set neuromodulator scales
    model.neuromodulator_levels = {
        'dopamine': 1.2,      # Increased reward influence
        'serotonin': 0.8,     # Balanced mood regulation
        'norepinephrine': 1.0, # Normal attention
        'acetylcholine': 1.1  # Enhanced memory formation
    }
    
    return model

def test_optimizations():
    """
    Test and compare baseline vs optimized neuron configurations.
    """
    print("=== Testing Neuron Optimizations ===\n")
    
    # Setup parameters
    input_size = 5
    hidden_size = 32
    num_layers = 2
    output_size = 1
    epochs = 10
    learning_rate = 0.01
    
    # Create dataset
    print("Creating sample dataset...")
    dataset = create_sample_data(
        batch_size=32,
        seq_length=10,
        input_size=input_size,
        output_size=output_size,
        num_batches=20
    )
    
    # Create models
    print("\nInitializing models...")
    model_baseline = setup_baseline_model(input_size, hidden_size, num_layers, output_size)
    model_optimized = setup_optimized_model(input_size, hidden_size, num_layers, output_size)
    
    # Use the same initialization for fair comparison
    print("Synchronizing initial weights...")
    with torch.no_grad():
        for p1, p2 in zip(model_baseline.parameters(), model_optimized.parameters()):
            p2.data = p1.data.clone()
    
    # Train baseline model
    print("\n=== Training Baseline Model ===")
    train_losses_baseline, val_losses_baseline = train_model(
        model_baseline, dataset, epochs=epochs, learning_rate=learning_rate
    )
    
    # Visualize baseline model health
    print("\nAnalyzing baseline model neuron health...")
    visualize_neuron_health(model_baseline, save_dir='results/baseline_viz')
    
    # Train optimized model
    print("\n=== Training Optimized Model ===")
    train_losses_optimized, val_losses_optimized = train_model(
        model_optimized, dataset, epochs=epochs, learning_rate=learning_rate
    )
    
    # Visualize optimized model health
    print("\nAnalyzing optimized model neuron health...")
    visualize_neuron_health(model_optimized, save_dir='results/optimized_viz')
    
    # Compare results
    print("\n=== Comparison Results ===")
    print(f"Final Baseline Validation Loss: {val_losses_baseline[-1]:.6f}")
    print(f"Final Optimized Validation Loss: {val_losses_optimized[-1]:.6f}")
    print(f"Improvement: {(1 - val_losses_optimized[-1] / val_losses_baseline[-1]) * 100:.2f}%")
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    os.makedirs('results', exist_ok=True)
    
    # Training loss comparison
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses_baseline, 'b-', label='Baseline Training')
    plt.plot(train_losses_optimized, 'g-', label='Optimized Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/training_loss_comparison.png')
    
    # Validation loss comparison
    plt.figure(figsize=(10, 6))
    plt.plot(val_losses_baseline, 'r-', label='Baseline Validation')
    plt.plot(val_losses_optimized, 'm-', label='Optimized Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/validation_loss_comparison.png')
    
    print("\n=== Optimization Test Complete ===")
    print(f"Results saved to the 'results' directory")

if __name__ == "__main__":
    test_optimizations()