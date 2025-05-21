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
from src.utils.financial_data_utils import load_financial_data, visualize_predictions
from src.utils.pretrain_utils import pretrain_controller, pretrain_neuromodulator_components, create_pretrain_dataloader
import pandas as pd

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
    parser.add_argument('--pretrain', action='store_true',
                        help='Enable pretraining of neural components')
    parser.add_argument('--pretrain-epochs', type=int, default=5,
                        help='Number of pretraining epochs')
    parser.add_argument('--skip-controller-pretrain', action='store_true',
                        help='Skip controller pretraining')
    parser.add_argument('--skip-neuromod-pretrain', action='store_true',
                        help='Skip neuromodulator pretraining')
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
        tuple: (train_loader, val_loader, test_loader, data_info)
    """
    # Use the specialized financial data loader
    return load_financial_data(config)

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
            
            # Import shape utils for fixing dimensions only when needed
            if output.shape != target.shape:
                # Log the shape mismatch
                print(f"Shape mismatch: output {output.shape}, target {target.shape}")
                
                # Use the fixed shape utils
                from src.utils.shape_error_fix import reshape_output_for_loss
                output_for_loss = reshape_output_for_loss(output, target)
                
                # Verify the fix worked
                if output_for_loss.shape == target.shape:
                    print(f"Shape fixed: {output_for_loss.shape}")
                else:
                    print(f"WARNING: Shape fix failed. Output: {output_for_loss.shape}, Target: {target.shape}")
                    # Emergency shape fix - force the correct shape
                    if target.shape[1] == 1 and output_for_loss.shape[1] > 1:
                        # If target is a single column but output has multiple columns, keep only the first column
                        output_for_loss = output_for_loss[:, 0:1]
                        print(f"Emergency fix applied: {output_for_loss.shape}")
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
                
                # Handle dimension mismatches using the same shape fix as in train_epoch
                if output.shape != target.shape:
                    # Use the fixed shape utils
                    from src.utils.shape_error_fix import reshape_output_for_loss
                    output_for_loss = reshape_output_for_loss(output, target)
                    
                    # Emergency fix for the common case in the error logs
                    if target.shape[1] == 1 and output_for_loss.shape[1] > 1:
                        # If target is a single column but output has multiple columns, keep only the first column
                        output_for_loss = output_for_loss[:, 0:1]
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
    
def configure_neuron_optimization(model, config):
    """
    Apply neuron optimization configurations to the model.
    
    Args:
        model: The BioGRU model instance
        config: Configuration dictionary
        
    Returns:
        model: Configured model
    """
    # Skip if neuron optimization not enabled
    if not config.get('neuron_optimization', {}).get('enabled', False):
        return model
    
    print("Applying neuron optimization configurations...")
    opt_config = config['neuron_optimization']
    
    # Apply to each layer
    if hasattr(model, 'layers'):
        for layer_idx, layer in enumerate(model.layers):
            # Skip non-BiologicalGRUCell layers
            if not hasattr(layer, 'neuron_mask'):
                continue
                
            for i in range(layer.hidden_size):
                if layer.neuron_mask[i] > 0:
                    for neuron in [layer.update_gate_neurons[i], layer.reset_gate_neurons[i], layer.candidate_neurons[i]]:
                        # Set target activity
                        neuron.target_activity = opt_config.get('target_activity', 0.15)
                        
                        # Set homeostatic rate
                        neuron.homeostatic_rate = opt_config.get('homeostatic_rate', 0.01)
                        
                        # Apply plasticity settings
                        if hasattr(neuron, 'hebbian_weight'):
                            plasticity = opt_config.get('plasticity', {})
                            neuron.hebbian_weight = plasticity.get('hebbian_weight', 0.3)
                            neuron.error_weight = plasticity.get('error_weight', 0.7)
                            
                        # Apply synapse settings
                        synapse = opt_config.get('synapse', {})
                        if hasattr(neuron, 'synaptic_facilitation'):
                            neuron.facilitation_rate = synapse.get('facilitation_rate', 0.1)
                        if hasattr(neuron, 'synaptic_depression'):
                            neuron.depression_rate = synapse.get('depression_rate', 0.2)
    
    # Apply to neuromodulator levels
    if config.get('neuromodulator', {}):
        neuromod_config = config['neuromodulator']
        model.neuromodulator_levels = {
            'dopamine': neuromod_config.get('dopamine_scale', 1.0),
            'serotonin': neuromod_config.get('serotonin_scale', 1.0),
            'norepinephrine': neuromod_config.get('norepinephrine_scale', 1.0),
            'acetylcholine': neuromod_config.get('acetylcholine_scale', 1.0)
        }
    
    print("Neuron optimization applied")
    return model

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
    
    # Load data with error handling
    try:
        train_loader, val_loader, test_loader, data_info = load_data(config)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("Please check your data configuration and ensure the financial data source is accessible.")
        sys.exit(1)
    
    # Adjust input size based on actual data dimensions if needed
    first_batch = next(iter(train_loader))
    input_features = first_batch[0].shape[-1]
    if input_features != config['model']['input_size']:
        print(f"Adjusting model input size from {config['model']['input_size']} to {input_features} based on data")
        config['model']['input_size'] = input_features
    
    # Create model and apply shape fixes 
    model = BrainInspiredNN(config).to(device)

    # Apply neuron optimization if enabled
    model = configure_neuron_optimization(model, config)

    # Verify neuron configuration
    if model.config.get('model', {}).get('use_bio_gru', False) and hasattr(model, 'controller'):
        print("\nVerifying neuron configurations:")
        if hasattr(model.controller, 'layers'):
            sample_layer = model.controller.layers[0]
            if hasattr(sample_layer, 'neuron_mask'):
                sample_neuron = None
                for i in range(sample_layer.hidden_size):
                    if sample_layer.neuron_mask[i] > 0:
                        sample_neuron = sample_layer.update_gate_neurons[i]
                        break
                
                if sample_neuron is not None:
                    print(f"  - Target activity: {sample_neuron.target_activity}")
                    print(f"  - Homeostatic rate: {sample_neuron.homeostatic_rate}")
                    if hasattr(sample_neuron, 'hebbian_weight'):
                        print(f"  - Hebbian weight: {sample_neuron.hebbian_weight}")
                    if hasattr(sample_neuron, 'synaptic_facilitation'):
                        print(f"  - Has synaptic dynamics: Yes")
    
    # Apply shape error fixes to prevent dimension mismatches during training
    from src.utils.shape_error_fix import apply_fixes
    model = apply_fixes(model)
    
    # Configure the model to expect output_size-shaped targets
    if hasattr(model, 'configure_shape_awareness'):
        model.configure_shape_awareness(target_shape=(model.output_size,), auto_adjust=True)
    
    # Pretraining logic
    if args.pretrain or config.get('pretraining', {}).get('enabled', False):
        # Customize pretraining config if args provided
        pretrain_config = config.get('pretraining', {}).copy()
        
        # Override epochs if specified via args
        if args.pretrain_epochs:
            pretrain_config['controller'] = pretrain_config.get('controller', {})
            pretrain_config['controller']['epochs'] = args.pretrain_epochs
            pretrain_config['neuromodulator'] = pretrain_config.get('neuromodulator', {})
            pretrain_config['neuromodulator']['epochs'] = args.pretrain_epochs
        
        # Skip components if specified via args
        if args.skip_controller_pretrain:
            pretrain_config['controller'] = pretrain_config.get('controller', {})
            pretrain_config['controller']['enabled'] = False
            
        if args.skip_neuromod_pretrain:
            pretrain_config['neuromodulator'] = pretrain_config.get('neuromodulator', {})
            pretrain_config['neuromodulator']['enabled'] = False
        
        # Create smaller dataloader for pretraining
        pretrain_dataloader = create_pretrain_dataloader(train_loader)
        
        # Use the unified model method for pretraining
        model.pretrain_components(pretrain_dataloader, device, pretrain_config)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    
    # Configure optimizer based on learning mode
    if config.get('learning_mode', '') == 'neuromodulator':
        optimizer = None
        print("Using neuromodulator-driven learning (no backprop)")
    else:
        weight_decay = config['training'].get('weight_decay', 0.0)
        optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=weight_decay)
        print(f"Using standard backprop with Adam (lr={config['training']['learnin
        +g+_rate']})")
    
    # Define learning rate scheduler
    scheduler = None
    if optimizer is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
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
    early_stopping_patience = config['training'].get('early_stopping_patience', 30)  # Extended patience period
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
        
        # Check for improvement - prioritize accuracy over loss
        improved = False
        
        # Keep track of best validation accuracy separately
        if 'best_val_accuracy' not in locals():
            best_val_accuracy = 0.0
        
        # Check if accuracy improved
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            improved = True
            epochs_without_improvement = 0
            print(f"Validation accuracy improved to {best_val_accuracy:.2f}%")
        # If accuracy is the same, check if loss improved
        elif val_accuracy == best_val_accuracy and val_loss < best_val_loss:
            best_val_loss = val_loss
            improved = True
            epochs_without_improvement = 0
            print(f"Validation loss improved to {best_val_loss:.6f} with same accuracy")
        # If loss improved significantly
        elif val_loss < best_val_loss * 0.95:  # 5% improvement threshold
            best_val_loss = val_loss
            improved = True
            epochs_without_improvement = 0
            print(f"Validation loss improved significantly to {best_val_loss:.6f}")
        
        # Save checkpoint if there was improvement
        if improved:
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_accuracy': best_val_accuracy,
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
    
    # Generate predictions for visualization
    all_predictions = []
    all_targets = []
    
    model.reset_state()
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Process for consistent format
            if output.dim() == 3:
                output = output[:, -1, :]
                
            all_predictions.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    # Convert to numpy arrays
    try:
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Transform back to original scale if needed
        if hasattr(data_info['test_dataset'], 'inverse_transform'):
            predictions = data_info['test_dataset'].inverse_transform(predictions)
            targets = data_info['test_dataset'].inverse_transform(targets)
            
        # Create and save prediction visualization
        fig = visualize_predictions(targets, predictions, 
                                    title=f"Predictions vs Actual ({config['data']['ticker_symbol']})")
        fig.savefig(os.path.join(plot_dir, 'price_predictions.png'))
        plt.close(fig)
        
        # Save predictions to CSV
        prediction_df = pd.DataFrame({
            'Actual': targets.flatten(),
            'Predicted': predictions.flatten()
        })
        prediction_df.to_csv(os.path.join('results', 'predictions.csv'), index=False)
        print(f"Predictions saved to results/predictions.csv")
        
    except Exception as e:
        print(f"Error visualizing predictions: {str(e)}")
    
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