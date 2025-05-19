#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fixed Training Script for Brain-Inspired Neural Network

This script fixes the tensor shape issues and provides a robust training pipeline
that handles errors gracefully without crashing.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import custom modules
from src.model import BrainInspiredNN
from tensor_shape_fixes import (
    add_init_hidden_to_controller,
    fix_controller_forward,
    fix_brain_nn_forward
)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_model(config):
    """Set up the model based on configuration."""
    # Extract model parameters from config
    input_size = config.get('input_size', 128)
    hidden_size = config['controller']['hidden_size']
    output_size = config.get('output_size', 64)
    persistent_memory_size = config['controller']['persistent_memory_size']
    num_layers = config['controller']['num_layers']
    dropout = config['controller']['dropout']
    
    # Neuromodulator parameters
    dopamine_scale = config['neuromodulator']['dopamine_scale']
    serotonin_scale = config['neuromodulator']['serotonin_scale']
    norepinephrine_scale = config['neuromodulator']['norepinephrine_scale']
    acetylcholine_scale = config['neuromodulator']['acetylcholine_scale']
    reward_decay = config['neuromodulator']['reward_decay']
    
    # Create model
    model = BrainInspiredNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        persistent_memory_size=persistent_memory_size,
        num_layers=num_layers,
        dropout=dropout,
        dopamine_scale=dopamine_scale,
        serotonin_scale=serotonin_scale,
        norepinephrine_scale=norepinephrine_scale,
        acetylcholine_scale=acetylcholine_scale,
        reward_decay=reward_decay
    )
    
    # Apply fixes to model components
    controller_class = model.controller.__class__
    add_init_hidden_to_controller(controller_class)
    fix_controller_forward(controller_class)
    fix_brain_nn_forward(model.__class__)
    
    return model

def setup_optimizer(model, config):
    """Set up optimizer and learning rate scheduler."""
    optimizer_name = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']
    
    # Set up optimizer
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Set up scheduler
    scheduler_name = config['training'].get('scheduler', 'none').lower()
    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['training']['num_epochs']
        )
    elif scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    elif scheduler_name == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    return optimizer, scheduler

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """
    Train the model for one epoch with proper handling of sequence outputs.
    
    Args:
        model: The model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Computation device
        epoch: Current epoch number
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    batch_count = 0
    
    for batch_idx, (data, target, reward) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        try:
            # Move data to device
            data = data.to(device)
            target = target.to(device)
            reward = reward.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            output, predicted_reward = model(data, external_reward=reward)
            
            # CRITICAL FIX: Handle sequence outputs
            if output.dim() == 3 and target.dim() == 2:
                # Extract last time step from sequence
                output_for_loss = output[:, -1, :]
                
                # Also handle reward if it's a sequence
                if predicted_reward.dim() == 3 and reward.dim() == 2:
                    predicted_reward_for_loss = predicted_reward[:, -1, :]
                else:
                    predicted_reward_for_loss = predicted_reward
            else:
                output_for_loss = output
                predicted_reward_for_loss = predicted_reward
            
            # Calculate losses with properly shaped tensors
            task_loss = criterion(output_for_loss, target)
            reward_loss = criterion(predicted_reward_for_loss, reward)
            
            # Combined loss
            loss = task_loss + 0.5 * reward_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            batch_count += 1
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            # Skip problematic batch
            continue
    
    # Return average loss
    return total_loss / max(1, batch_count)

def validate(model, dataloader, criterion, device):
    """
    Validate the model with proper handling of sequence outputs.
    
    Args:
        model: The model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Computation device
        
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, (data, target, reward) in enumerate(tqdm(dataloader, desc="Validation")):
            try:
                # Move data to device
                data = data.to(device)
                target = target.to(device)
                reward = reward.to(device)
                
                # Forward pass
                output, predicted_reward = model(data, external_reward=reward)
                
                # CRITICAL FIX: Handle sequence outputs
                if output.dim() == 3 and target.dim() == 2:
                    # Extract last time step from sequence
                    output_for_loss = output[:, -1, :]
                    
                    # Also handle reward if it's a sequence
                    if predicted_reward.dim() == 3 and reward.dim() == 2:
                        predicted_reward_for_loss = predicted_reward[:, -1, :]
                    else:
                        predicted_reward_for_loss = predicted_reward
                else:
                    output_for_loss = output
                    predicted_reward_for_loss = predicted_reward
                
                # Calculate losses with properly shaped tensors
                task_loss = criterion(output_for_loss, target)
                reward_loss = criterion(predicted_reward_for_loss, reward)
                
                # Combined loss
                loss = task_loss + 0.5 * reward_loss
                
                # Accumulate loss
                total_loss += loss.item()
                batch_count += 1
                
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                # Skip problematic batch
                continue
    
    # Return average loss
    return total_loss / max(1, batch_count)

def test(model, dataloader, device):
    """
    Test the model with proper handling of sequence outputs.
    
    Args:
        model: The model to test
        dataloader: Test data loader
        device: Computation device
        
    Returns:
        dict: Test metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (data, target, reward) in enumerate(tqdm(dataloader, desc="Testing")):
            try:
                # Move data to device
                data = data.to(device)
                target = target.to(device)
                
                # Forward pass
                output, _ = model(data)
                
                # CRITICAL FIX: Handle sequence outputs
                if output.dim() == 3:
                    # Extract last time step for metrics
                    output = output[:, -1, :]
                
                # Store predictions and targets
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                
            except Exception as e:
                print(f"Error in test batch {batch_idx}: {e}")
                # Skip problematic batch
                continue
    
    # Calculate metrics if we have any predictions
    if all_predictions and all_targets:
        try:
            # Stack arrays
            predictions = np.vstack(all_predictions)
            targets = np.vstack(all_targets)
            
            # Ensure consistent shapes
            if predictions.shape != targets.shape:
                print(f"Warning: Shape mismatch - predictions {predictions.shape}, targets {targets.shape}")
                min_rows = min(predictions.shape[0], targets.shape[0])
                predictions = predictions[:min_rows]
                targets = targets[:min_rows]
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            mse = mean_squared_error(targets, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(targets, predictions)
            
            # Safe R² calculation
            try:
                r2 = r2_score(targets, predictions)
            except:
                r2 = float('nan')
            
            # Direction accuracy (if we have at least 2 points)
            direction_accuracy = 0.0
            if len(predictions) >= 2:
                pred_direction = np.diff(predictions.flatten())
                true_direction = np.diff(targets.flatten())
                direction_match = (pred_direction > 0) == (true_direction > 0)
                direction_accuracy = np.mean(direction_match)
            
            # Create metrics dictionary
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'direction_accuracy': float(direction_accuracy * 100),
                'predictions': predictions,
                'targets': targets
            }
            
            # Print metrics
            print("Test Results:")
            for key, value in metrics.items():
                if key not in ['predictions', 'targets']:
                    print(f"  {key.upper()}: {value}")
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {'error': str(e)}
    else:
        print("No valid predictions generated during testing")
        return {'error': 'No valid predictions'}

def create_datasets(config, stock_data=None):
    """
    Create datasets for stock prediction.
    
    Args:
        config (dict): Configuration dictionary
        stock_data (pd.DataFrame, optional): Stock data if already loaded
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset) PyTorch datasets
    """
    # If stock data is not provided, load it
    if stock_data is None:
        import yfinance as yf
        
        ticker_symbol = config['data']['ticker_symbol']
        start_date = config['data']['start_date']
        end_date = config['data']['end_date']
        
        print(f"Loading stock data for {ticker_symbol} from {start_date} to {end_date}")
        stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
        
        if stock_data.empty:
            raise ValueError(f"No data found for ticker {ticker_symbol} in the specified date range")
        
        print(f"Downloaded {len(stock_data)} days of stock data")
    
    # Process the data
    data_info = BrainInspiredNN.preprocess_data(stock_data, config)
    
    # Extract dataloaders
    train_dataloader = data_info['dataloaders']['train']
    val_dataloader = data_info['dataloaders']['val']
    test_dataloader = data_info['dataloaders']['test']
    
    return train_dataloader, val_dataloader, test_dataloader, data_info

def save_checkpoint(model, optimizer, epoch, loss, config, checkpoint_dir):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }, checkpoint_path)
    
    print(f"Checkpoint saved to {checkpoint_path}")

def main(args):
    """Main training function."""
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Create directories
        os.makedirs(config['general']['log_dir'], exist_ok=True)
        os.makedirs(config['general']['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['general']['results_dir'], exist_ok=True)
        
        # Set device
        device_name = config['general'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device_name)
        print(f"Using device: {device}")
        
        # Set random seed
        seed = config['general'].get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed)
        
        # Load stock data
        ticker_symbol = config['data']['ticker_symbol']
        start_date = config['data']['start_date']
        end_date = config['data']['end_date']
        
        print(f"Loading stock data for {ticker_symbol} from {start_date} to {end_date}")
        
        import yfinance as yf
        stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
        
        if stock_data.empty:
            raise ValueError(f"No data found for ticker {ticker_symbol} in the specified date range")
        
        print(f"Downloaded {len(stock_data)} days of stock data")
        
        # Create datasets
        train_dataloader, val_dataloader, test_dataloader, data_info = create_datasets(config, stock_data)
        
        # Set up model
        model = setup_model(config)
        model.to(device)
        
        # Set up optimizer and scheduler
        optimizer, scheduler = setup_optimizer(model, config)
        
        # Set up loss function
        criterion = nn.MSELoss()
        
        # Training parameters
        num_epochs = config['training']['num_epochs']
        patience = config['training'].get('patience', 10)
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Print model summary
        print(f"\nModel Architecture:")
        print(f"  Input Size: {model.input_size}")
        print(f"  Hidden Size: {model.hidden_size}")
        print(f"  Output Size: {model.output_size}")
        print(f"  Persistent Memory Size: {model.controller.persistent_memory_size}")
        print(f"  Number of Layers: {model.controller.num_layers}")
        
        # Print training configuration
        print(f"\nTraining Configuration:")
        print(f"  Number of Epochs: {num_epochs}")
        print(f"  Learning Rate: {config['training']['learning_rate']}")
        print(f"  Optimizer: {config['training']['optimizer']}")
        print(f"  Early Stopping Patience: {patience}")
        
        print("\n" + "="*60)
        print(f"TRAINING FOR {num_epochs} EPOCHS")
        print("="*60 + "\n")
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            # Train epoch
            train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device, epoch)
            
            # Validate
            val_loss = validate(model, val_dataloader, criterion, device)
            
            # Print progress
            print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Update learning rate if scheduler exists
            if scheduler is not None:
                scheduler.step()
            
            # Check for best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # Save best model
                best_model_path = os.path.join(config['general']['checkpoint_dir'], "best_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config
                }, best_model_path)
                
                print(f"New best model saved with validation loss: {val_loss:.6f}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs")
            
            # Save checkpoint at intervals
            if epoch % args.checkpoint_interval == 0:
                save_checkpoint(model, optimizer, epoch, val_loss, config, config['general']['checkpoint_dir'])
            
            # Early stopping
            if patience > 0 and patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Save final model
        final_model_path = os.path.join(config['general']['checkpoint_dir'], "final_model.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': config
        }, final_model_path)
        
        print(f"Final model saved to: {final_model_path}")
        
        # Test the model
        print("\n" + "="*60)
        print("TESTING")
        print("="*60 + "\n")
        
        test_metrics = test(model, test_dataloader, device)
        
        # Save test metrics
        results_path = os.path.join(config['general']['results_dir'], "test_results.yaml")
        with open(results_path, 'w') as f:
            yaml.dump({k: v for k, v in test_metrics.items() if k not in ['predictions', 'targets']}, f)
        
        print(f"Test results saved to: {results_path}")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Brain-Inspired Neural Network")
    parser.add_argument("--config", type=str, default="config/config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--checkpoint-interval", type=int, default=10, 
                        help="Interval (in epochs) for saving checkpoints")
    
    args = parser.parse_args()
    sys.exit(main(args))
