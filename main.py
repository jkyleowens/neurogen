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

# Import the model
from src.model import BrainInspiredNN
from src.utils.memory_utils import optimize_memory_usage, print_gpu_memory_status
from src.utils.performance_report import generate_performance_report
from src.utils.reset_model_state import reset_model_state

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Brain-Inspired Neural Network')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config file)')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_data(config):
    """Load and preprocess financial data."""
    # This function would be properly implemented with your actual data loading logic
    # Using placeholder data for now
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

def calculate_accuracy(output, target, threshold=0.1):
    """
    Calculate accuracy for regression by checking closeness.
    
    Args:
        output: Model predictions
        target: Ground truth values
        threshold: Maximum allowed error for a prediction to be considered correct
        
    Returns:
        float: Accuracy as a percentage
    """
    with torch.no_grad():
        # Handle dimension mismatches
        if output.dim() == 3 and target.dim() == 2:
            output = output[:, -1, :]
        
        # Ensure shapes match for comparison
        if output.shape != target.shape:
            # Get common dimensions
            batch_size = min(output.size(0), target.size(0))
            feature_size = min(output.size(-1), target.size(-1))
            
            # Trim both tensors to common dimensions
            output = output[:batch_size, ..., :feature_size]
            target = target[:batch_size, ..., :feature_size]
        
        # Calculate absolute error
        error = torch.abs(output - target)
        
        # Count predictions within threshold
        correct = (error < threshold).float()
        
        # Calculate accuracy percentage
        accuracy = 100.0 * correct.mean().item()
        
        return accuracy

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
        tuple: (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    batch_count = 0
    total_accuracy = 0.0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Training')):
        try:
            # Reset model state for each batch
            model.reset_state()
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Handle dimension mismatches
            if output.dim() == 3 and target.dim() == 2:
                # If model outputs a sequence, take the last time step
                output_for_loss = output[:, -1, :]
            elif output.shape != target.shape:
                # Log the shape mismatch
                print(f"Shape mismatch: output {output.shape}, target {target.shape}")
                
                # Find common dimensions
                batch_size = min(output.size(0), target.size(0))
                if output.dim() > 1 and target.dim() > 1:
                    out_features = output.size(-1)
                    target_features = target.size(-1)
                    features = min(out_features, target_features)
                    
                    # Trim both tensors to common dimensions
                    if output.dim() >= 3:
                        output_for_loss = output[:batch_size, -1, :features]
                    else:
                        output_for_loss = output[:batch_size, :features]
                    
                    target = target[:batch_size, :features]
                else:
                    # Handle scalar case or other extreme mismatches
                    print("Severe shape mismatch - using placeholder loss")
                    output_for_loss = output.reshape(-1)[:1]
                    target = target.reshape(-1)[:1]
            else:
                output_for_loss = output
            
            # Calculate loss
            loss = criterion(output_for_loss, target)
            
            # Handle NaN or Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss detected in batch {batch_idx}. Skipping.")
                continue
                
            # Neuromodulator-driven learning (no backprop)
            if optimizer is None:
                with torch.no_grad():
                    # Use negative loss as reward signal
                    reward = -loss.detach()
                    # Update model with reward feedback
                    model(data, reward=reward)
            else:
                # Traditional backprop learning
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping to prevent explosions
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            # Calculate accuracy
            batch_accuracy = calculate_accuracy(output_for_loss, target)
            
            # Accumulate metrics
            total_loss += loss.item()
            total_accuracy += batch_accuracy
            batch_count += 1
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            # Continue with next batch
            continue
    
    # Return average metrics
    if batch_count == 0:
        return 0.0, 0.0
    
    return total_loss / batch_count, total_accuracy / batch_count

def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: The neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to use for validation
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    # Reset model state before validation
    model.reset_state()
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(val_loader, desc='Validation')):
            try:
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = model(data)
                
                # Handle dimension mismatches
                if output.dim() == 3 and target.dim() == 2:
                    # If model outputs a sequence, take the last time step
                    output_for_loss = output[:, -1, :]
                elif output.shape != target.shape:
                    # Find common dimensions
                    batch_size = min(output.size(0), target.size(0))
                    if output.dim() > 1 and target.dim() > 1:
                        out_features = output.size(-1)
                        target_features = target.size(-1)
                        features = min(out_features, target_features)
                        
                        # Trim both tensors to common dimensions
                        if output.dim() >= 3:
                            output_for_loss = output[:batch_size, -1, :features]
                        else:
                            output_for_loss = output[:batch_size, :features]
                        
                        target = target[:batch_size, :features]
                    else:
                        # Handle scalar case or other extreme mismatches
                        output_for_loss = output.reshape(-1)[:1]
                        target = target.reshape(-1)[:1]
                else:
                    output_for_loss = output
                
                # Calculate loss
                loss = criterion(output_for_loss, target)
                
                # Handle NaN or Inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN or Inf loss detected in validation batch {batch_idx}. Skipping.")
                    continue
                
                # Calculate accuracy
                batch_accuracy = calculate_accuracy(output_for_loss, target)
                
                # Accumulate metrics
                total_loss += loss.item()
                total_accuracy += batch_accuracy
                batch_count += 1
                
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {str(e)}")
                # Continue with next batch
                continue
    
    # Return average metrics
    if batch_count == 0:
        return float('inf'), 0.0
    
    return total_loss / batch_count, total_accuracy / batch_count

def test(model, test_loader, criterion, device):
    """
    Test the model and calculate comprehensive metrics.
    
    Args:
        model: The neural network model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to use for testing
        
    Returns:
        dict: Dictionary of test metrics
    """
    # Reset model state before testing
    model.reset_state()
    model.eval()
    
    total_loss = 0.0
    total_accuracy = 0.0
    batch_count = 0
    
    # Containers for predictions and targets
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc='Testing')):
            try:
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = model(data)
                
                # Process outputs for consistent format
                if output.dim() == 3:
                    # If model outputs a sequence, take the last time step
                    output = output[:, -1, :]
                
                # Ensure output and target have compatible shapes for metrics
                if output.shape != target.shape:
                    # Find common dimensions
                    batch_size = min(output.size(0), target.size(0))
                    out_features = output.size(-1) if output.dim() > 1 else 1
                    target_features = target.size(-1) if target.dim() > 1 else 1
                    features = min(out_features, target_features)
                    
                    # Reshape output and target to have compatible dimensions
                    if output.dim() > 1:
                        output = output[:batch_size, :features]
                    else:
                        output = output[:batch_size].reshape(batch_size, 1)
                    
                    if target.dim() > 1:
                        target = target[:batch_size, :features]
                    else:
                        target = target[:batch_size].reshape(batch_size, 1)
                
                # Calculate loss
                loss = criterion(output, target)
                
                # Skip NaN or Inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN or Inf loss detected in test batch {batch_idx}. Skipping.")
                    continue
                
                # Calculate accuracy
                batch_accuracy = calculate_accuracy(output, target)
                
                # Accumulate metrics
                total_loss += loss.item()
                total_accuracy += batch_accuracy
                batch_count += 1
                
                # Store predictions and targets for further analysis
                all_predictions.append(output.cpu().detach())
                all_targets.append(target.cpu().detach())
                
            except Exception as e:
                print(f"Error in test batch {batch_idx}: {str(e)}")
                # Continue with next batch
                continue
    
    # Handle empty results
    if batch_count == 0:
        return {
            'loss': float('inf'),
            'accuracy': 0.0,
            'error': 'No valid batches processed'
        }
    
    # Calculate average metrics
    avg_loss = total_loss / batch_count
    avg_accuracy = total_accuracy / batch_count
    
    # Compile all predictions and targets for advanced metrics
    if all_predictions and all_targets:
        try:
            # Convert lists of tensors to single tensors
            predictions = torch.cat(all_predictions, dim=0).numpy()
            targets = torch.cat(all_targets, dim=0).numpy()
            
            # Calculate MSE
            mse = np.mean((predictions - targets) ** 2)
            rmse = np.sqrt(mse)
            
            # Calculate MAE
            mae = np.mean(np.abs(predictions - targets))
            
            # Calculate R² score
            ss_tot = np.sum((targets - np.mean(targets, axis=0)) ** 2)
            ss_res = np.sum((targets - predictions) ** 2)
            r2 = 1 - (ss_res / (ss_tot if ss_tot > 0 else 1e-10))
            
            # Calculate directional accuracy
            pred_diff = np.diff(predictions, axis=0)
            target_diff = np.diff(targets, axis=0)
            direction_matches = (np.sign(pred_diff) == np.sign(target_diff))
            directional_accuracy = np.mean(direction_matches) * 100
            
            # Return comprehensive metrics
            metrics = {
                'loss': avg_loss,
                'accuracy': avg_accuracy,
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'directional_accuracy': float(directional_accuracy)
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating advanced metrics: {str(e)}")
            # Return basic metrics if advanced calculations fail
            return {
                'loss': avg_loss,
                'accuracy': avg_accuracy,
                'error': str(e)
            }
    
    # Return basic metrics
    return {
        'loss': avg_loss,
        'accuracy': avg_accuracy
    }

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override num_epochs from command line if provided
    if hasattr(args, 'epochs') and args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
        print(f"Using command-line specified epochs: {args.epochs}")
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, test_loader = load_data(config)
    
    # Create model
    model = BrainInspiredNN(config).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    
    # Configure optimizer based on learning mode
    if config.get('learning_mode', '') == 'neuromodulator':
        optimizer = None
        print("Using neuromodulator-driven learning (no backprop)")
    else:
        weight_decay = config['training'].get('weight_decay', 0.0)
        optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=weight_decay)
        print(f"Using standard backprop with Adam (lr={config['training']['learning_rate']})")
    
    # Define learning rate scheduler
    scheduler = None
    if optimizer is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.6f}")
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            print("Starting training from scratch")
    
    # Training parameters
    num_epochs = config['training']['num_epochs']
    early_stopping_patience = config['training'].get('early_stopping_patience', 10)
    checkpoint_dir = os.path.join('models', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Metrics tracking
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    epochs_without_improvement = 0
    
    # Output directories for logs
    log_dir = os.path.join('logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs")
    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f"\nEpoch {epoch+1}/{start_epoch + num_epochs}")
        
        # Train for one epoch
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validate
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Log results
        print(f"Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.6f}, Val Accuracy: {val_accuracy:.2f}%")
        
        # Save to log file
        with open(os.path.join(log_dir, 'training_log.txt'), 'a') as f:
            f.write(f"Epoch {epoch+1}: Train Loss={train_loss:.6f}, Train Acc={train_accuracy:.2f}%, "
                    f"Val Loss={val_loss:.6f}, Val Acc={val_accuracy:.2f}%\n")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
                'val_accuracy': val_accuracy,
                'config': config
            }
            
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
                
            torch.save(checkpoint, checkpoint_path)
            print(f"New best model saved (val_loss: {best_val_loss:.6f}, val_accuracy: {val_accuracy:.2f}%)")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs")
        
        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Periodically clean up memory
        if (epoch + 1) % 5 == 0:
            model = optimize_memory_usage(model, device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, 'final_model.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_checkpoint_path)
    print(f"Final model saved to {final_checkpoint_path}")
    
    # Load best model for testing
    try:
        best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {best_checkpoint['epoch']+1} for testing")
    except Exception as e:
        print(f"Error loading best model: {str(e)}")
        print("Using final model for testing")
    
    # Test the model
    print("\nEvaluating model on test set...")
    test_metrics = test(model, test_loader, criterion, device)
    
    # Print test results
    print("\n===== Test Results =====")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")
    
    # Generate performance report with all metrics
    report_metrics = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies,
        'test_metrics': test_metrics
    }
    
    # Create performance visualizations
    plot_dir = os.path.join('docs', 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'loss_curves.png'))
    
    # Plot accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_accuracies)+1), train_accuracies, 'g-', label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies)+1), val_accuracies, 'm-', label='Validation Accuracy')  # Fixed color 'p' to 'm'
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'accuracy_curves.png'))
    
    # Generate performance report
    generate_performance_report(
        train_losses[-1], 
        val_losses[-1], 
        test_metrics['loss'],
        metrics=test_metrics,
        output_dir='docs/'
    )
    print("\nPerformance report generated in docs/")

if __name__ == "__main__":
    main()