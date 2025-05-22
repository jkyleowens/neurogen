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
from datetime import datetime
import pandas as pd

# Import the model and utilities
from src.model import BrainInspiredNN
from src.utils.memory_utils import optimize_memory_usage, print_gpu_memory_status
from src.utils.performance_report import generate_performance_report
from src.utils.reset_model_state import reset_model_state
from src.utils.shape_error_fix import quick_fix_model

# Try to import enhanced utilities with fallbacks
try:
    from src.utils.pretrain_utils import pretrain_controller, pretrain_neuromodulator_components, create_pretrain_dataloader
    PRETRAIN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Pretraining utilities not available: {e}")
    PRETRAIN_AVAILABLE = False
    # Create dummy functions
    def create_pretrain_dataloader(dataloader, batch_size=32):
        return dataloader

# Import data loading with enhanced fallback
try:
    # Try to import from the existing file first
    from src.utils.financial_data_utils import load_financial_data
    print("Using existing financial data utilities")
    ENHANCED_DATA_AVAILABLE = False
except ImportError:
    try:
        # Try data_loader.py as fallback
        from data_loader import create_datasets
        print("Using data_loader.py as fallback")
        ENHANCED_DATA_AVAILABLE = False
        
        def load_financial_data(config):
            """Fallback data loading function"""
            try:
                train_dataset, val_dataset, test_dataset = create_datasets(config)
                
                batch_size = config['training']['batch_size']
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                
                data_info = {
                    'train_dataset': train_dataset,
                    'val_dataset': val_dataset, 
                    'test_dataset': test_dataset
                }
                
                return train_loader, val_loader, test_loader, data_info
            except Exception as e:
                print(f"Error in fallback data loading: {e}")
                raise
    except ImportError as e:
        print(f"Error: Could not import data loading functions: {e}")
        print("Please ensure you have either:")
        print("1. src/utils/financial_data_utils.py with load_financial_data function")
        print("2. data_loader.py with create_datasets function")
        sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Brain-Inspired Neural Network with Overfitting Mitigation')
    parser.add_argument('--config', type=str, default='config/financial_config.yaml',
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
    parser.add_argument('--cross-validate', action='store_true',
                        help='Run cross-validation to find optimal hyperparameters')
    parser.add_argument('--ensemble', action='store_true',
                        help='Train ensemble of models with different augmentations')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Config file {config_path} not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)

def calculate_accuracy(output, target, threshold=0.02):
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

def train_epoch(model, train_loader, optimizer, criterion, device, config):
    """
    Train the model for one epoch with enhanced anti-overfitting techniques.
    """
    model.train()
    total_loss = 0.0
    batch_count = 0
    total_accuracy = 0.0
    
    for batch_idx, batch_data in enumerate(tqdm(train_loader, desc='Training')):
        try:
            # Handle different batch formats
            if len(batch_data) == 3:
                data, target, reward = batch_data
                reward = reward.to(device)
            else:
                data, target = batch_data
                reward = None
            
            # Reset model state for each batch
            model.reset_state()
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Handle shape mismatches
            if output.shape != target.shape:
                from src.utils.shape_error_fix import reshape_output_for_loss
                output_for_loss = reshape_output_for_loss(output, target)
            else:
                output_for_loss = output
            
            # Calculate main prediction loss
            prediction_loss = criterion(output_for_loss, target)
            loss = prediction_loss
            
            # Handle NaN or Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss detected in batch {batch_idx}. Skipping.")
                continue
            
            # Learning approach based on configuration
            learning_mode = config.get('training', {}).get('learning_mode', 'backprop')
            
            if learning_mode == 'neuromodulator' or optimizer is None:
                # Neuromodulator-driven learning
                with torch.no_grad():
                    # Use negative loss as reward signal, but scale it
                    raw_reward = -loss.detach()
                    reward_signal = torch.tanh(raw_reward * 0.5)  # Scale to prevent excessive feedback
                    
                    # Update model with reward feedback
                    model(data, reward=reward_signal)
            else:
                # Traditional backprop learning
                optimizer.zero_grad()
                loss.backward()
                
                # Enhanced gradient clipping
                max_norm = config.get('training', {}).get('gradient_clip', 1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                
                optimizer.step()
            
            # Calculate accuracy
            batch_accuracy = calculate_accuracy(
                output_for_loss, target, 
                threshold=config.get('training', {}).get('accuracy_threshold', 0.02)
            )
            
            # Accumulate metrics
            total_loss += loss.item()
            total_accuracy += batch_accuracy
            batch_count += 1
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            continue
    
    # Return average metrics
    if batch_count == 0:
        return 0.0, 0.0
    
    return total_loss / batch_count, total_accuracy / batch_count

def validate(model, val_loader, criterion, device, config):
    """
    Validate the model with enhanced metrics and overfitting detection.
    """
    model.reset_state()
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    batch_count = 0
    
    # Track predictions for detailed analysis
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(val_loader, desc='Validation')):
            try:
                # Handle different batch formats
                if len(batch_data) == 3:
                    data, target, _ = batch_data
                else:
                    data, target = batch_data
                
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = model(data)
                
                # Handle dimension mismatches
                if output.shape != target.shape:
                    from src.utils.shape_error_fix import reshape_output_for_loss
                    output_for_loss = reshape_output_for_loss(output, target)
                else:
                    output_for_loss = output
                
                # Calculate loss
                loss = criterion(output_for_loss, target)
                
                # Skip NaN/Inf losses
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                # Calculate accuracy
                batch_accuracy = calculate_accuracy(
                    output_for_loss, target,
                    threshold=config.get('training', {}).get('accuracy_threshold', 0.02)
                )
                
                # Accumulate metrics
                total_loss += loss.item()
                total_accuracy += batch_accuracy
                batch_count += 1
                
                # Store for detailed analysis
                all_predictions.append(output_for_loss.cpu().detach())
                all_targets.append(target.cpu().detach())
                
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {str(e)}")
                continue
    
    # Calculate detailed metrics
    metrics = {}
    if all_predictions and all_targets and batch_count > 0:
        try:
            predictions = torch.cat(all_predictions, dim=0).numpy()
            targets = torch.cat(all_targets, dim=0).numpy()
            
            # MSE and direction accuracy
            from sklearn.metrics import mean_squared_error
            mse = mean_squared_error(targets, predictions)
            metrics['mse'] = mse
            
            # Direction accuracy for financial data
            if len(predictions) > 1:
                pred_diff = np.diff(predictions, axis=0)
                target_diff = np.diff(targets, axis=0)
                correct_directions = np.sign(pred_diff) == np.sign(target_diff)
                direction_accuracy = np.mean(correct_directions) * 100
                metrics['direction_accuracy'] = direction_accuracy
            
        except Exception as e:
            print(f"Error calculating detailed metrics: {e}")
    
    # Return results
    if batch_count == 0:
        return float('inf'), 0.0, metrics
    
    avg_loss = total_loss / batch_count
    avg_accuracy = total_accuracy / batch_count
    
    metrics['avg_loss'] = avg_loss
    metrics['avg_accuracy'] = avg_accuracy
    
    return avg_loss, avg_accuracy, metrics

def early_stopping_check(val_metrics, epoch, best_metrics, patience, delta=0.001):
    """Enhanced early stopping with multi-metric evaluation."""
    improved = False
    epochs_no_improve = best_metrics.get('epochs_no_improve', 0)
    
    # Primary metric based on availability
    if 'direction_accuracy' in val_metrics:
        primary_metric = 'direction_accuracy'
        if val_metrics[primary_metric] > best_metrics.get(primary_metric, 0) + delta:
            improved = True
    else:
        primary_metric = 'avg_loss'
        if val_metrics[primary_metric] < best_metrics.get(primary_metric, float('inf')) - delta:
            improved = True
    
    # Secondary improvement check
    if not improved and 'mse' in val_metrics and 'mse' in best_metrics:
        if val_metrics['mse'] < best_metrics['mse'] * 0.95:
            improved = True
            print(f"No improvement in {primary_metric}, but MSE improved significantly")
    
    # Update metrics
    if improved:
        print(f"Validation metrics improved at epoch {epoch+1}")
        best_metrics = val_metrics.copy()
        best_metrics['best_epoch'] = epoch
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve}/{patience} epochs")
    
    best_metrics['epochs_no_improve'] = epochs_no_improve
    stop_training = epochs_no_improve >= patience
    
    return stop_training, improved, best_metrics, epochs_no_improve

def test_model(model, test_loader, criterion, device, config):
    """Comprehensive model testing with detailed metrics."""
    model.reset_state()
    model.eval()
    
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Testing"):
            try:
                # Handle different batch formats
                if len(batch_data) == 3:
                    data, target, _ = batch_data
                else:
                    data, target = batch_data
                
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = model(data)
                
                # Handle output shapes
                if output.dim() == 3:
                    output = output[:, -1, :]  # Take last timestep
                
                # Ensure compatible shapes
                if output.shape != target.shape:
                    from src.utils.shape_error_fix import reshape_output_for_loss
                    output = reshape_output_for_loss(output, target)
                
                # Calculate loss
                loss = criterion(output, target)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    batch_count += 1
                
                # Store results
                all_predictions.append(output.cpu())
                all_targets.append(target.cpu())
                
            except Exception as e:
                print(f"Error in test batch: {e}")
                continue
    
    # Calculate comprehensive metrics
    if all_predictions and all_targets:
        predictions = torch.cat(all_predictions, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()
        
        # Basic metrics
        try:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics = {
                'loss': total_loss / max(1, batch_count),
                'mse': mean_squared_error(targets, predictions),
                'mae': mean_absolute_error(targets, predictions),
                'rmse': np.sqrt(mean_squared_error(targets, predictions))
            }
            
            # R² score (handle edge cases)
            try:
                metrics['r2'] = r2_score(targets, predictions)
            except:
                metrics['r2'] = 0.0
            
            # Financial-specific metrics
            if len(predictions) > 1:
                # Direction accuracy
                pred_directions = np.sign(np.diff(predictions, axis=0))
                true_directions = np.sign(np.diff(targets, axis=0))
                direction_matches = pred_directions == true_directions
                metrics['direction_accuracy'] = np.mean(direction_matches) * 100
                
                # Prediction accuracy within threshold
                threshold = config.get('training', {}).get('accuracy_threshold', 0.02)
                within_threshold = np.abs(predictions - targets) < threshold
                metrics['accuracy'] = np.mean(within_threshold) * 100
            
            metrics['predictions'] = predictions
            metrics['targets'] = targets
            
            return metrics
        except ImportError:
            print("Warning: sklearn not available for detailed metrics")
            return {
                'loss': total_loss / max(1, batch_count),
                'predictions': predictions,
                'targets': targets
            }
    
    return {'error': 'No valid predictions generated'}

def main():
    """Enhanced main training function with overfitting mitigation."""
    print("Starting Brain-Inspired Neural Network Training")
    
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
        print(f"Using command-line specified epochs: {args.epochs}")
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    try:
        train_loader, val_loader, test_loader, data_info = load_financial_data(config)
        print("Data loaded successfully")
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("Please check your data configuration and ensure required files exist.")
        sys.exit(1)
    
    # Adjust input size based on actual data
    try:
        first_batch = next(iter(train_loader))
        if len(first_batch) >= 2:
            input_features = first_batch[0].shape[-1]
            if input_features != config['model']['input_size']:
                print(f"Adjusting input size from {config['model']['input_size']} to {input_features}")
                config['model']['input_size'] = input_features
    except Exception as e:
        print(f"Warning: Could not determine input size from data: {e}")
    
    # Create and configure model
    try:
        model = BrainInspiredNN(config).to(device)
        model = quick_fix_model(model)
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model created with {total_params:,} parameters")
        
    except Exception as e:
        print(f"Error creating model: {e}")
        sys.exit(1)
    
    # Pretraining
    if (args.pretrain or config.get('pretraining', {}).get('enabled', False)) and PRETRAIN_AVAILABLE:
        print("Starting component pretraining...")
        
        try:
            pretrain_config = config.get('pretraining', {}).copy()
            if args.pretrain_epochs:
                pretrain_config.setdefault('controller', {})['epochs'] = args.pretrain_epochs
                pretrain_config.setdefault('neuromodulator', {})['epochs'] = args.pretrain_epochs
            
            if args.skip_controller_pretrain:
                pretrain_config.setdefault('controller', {})['enabled'] = False
            if args.skip_neuromod_pretrain:
                pretrain_config.setdefault('neuromodulator', {})['enabled'] = False
            
            pretrain_dataloader = create_pretrain_dataloader(train_loader)
            
            if hasattr(model, 'pretrain_components'):
                model.pretrain_components(pretrain_dataloader, device, pretrain_config)
                print("Pretraining completed")
            else:
                print("Warning: Model does not support pretraining")
                
        except Exception as e:
            print(f"Error during pretraining: {e}")
            print("Continuing without pretraining...")
    
    # Setup training
    criterion = nn.MSELoss()
    learning_mode = config.get('training', {}).get('learning_mode', 'backprop')
    
    if learning_mode == 'neuromodulator':
        optimizer = None
        print("Using neuromodulator-driven learning")
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0.0)
        )
        print(f"Using backprop with Adam optimizer (lr={config['training']['learning_rate']})")
    
    # Setup learning rate scheduler
    scheduler = None
    if optimizer and config.get('training', {}).get('lr_scheduler'):
        scheduler_config = config['training']['lr_scheduler']
        if scheduler_config.get('type') == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5), verbose=True
            )
        print("Learning rate scheduler enabled")
    
    # Training setup
    num_epochs = config['training']['num_epochs']
    early_stopping_patience = config['training'].get('early_stopping_patience', 30)
    early_stopping_delta = config['training'].get('early_stopping_delta', 0.001)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join('models', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_metrics = {'avg_loss': float('inf'), 'avg_accuracy': 0}
    
    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_metrics = checkpoint.get('best_metrics', best_metrics)
            print(f"Resumed from epoch {start_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    # Training tracking
    history = {
        'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': [],
        'direction_accuracy': [], 'learning_rates': []
    }
    
    # Main training loop
    print(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Track learning rate
        if optimizer:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")
            history['learning_rates'].append(current_lr)
        
        # Train epoch
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, criterion, device, config)
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        
        # Validate
        val_loss, val_accuracy, val_metrics = validate(model, val_loader, criterion, device, config)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        if 'direction_accuracy' in val_metrics:
            history['direction_accuracy'].append(val_metrics['direction_accuracy'])
            print(f"Direction Accuracy: {val_metrics['direction_accuracy']:.2f}%")
        
        print(f"Train Loss: {train_loss:.6f}, Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.6f}, Val Acc: {val_accuracy:.2f}%")
        
        # Check for overfitting signals
        if train_loss * 0.7 > val_loss:
            print("Note: Training loss higher than validation - possible underfitting")
        elif val_loss > train_loss * 1.5:
            print("Warning: Validation loss much higher than training - possible overfitting")
        
        # Update scheduler
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Early stopping check
        val_metrics.update({'avg_loss': val_loss, 'avg_accuracy': val_accuracy})
        stop_training, improved, best_metrics, epochs_no_improve = early_stopping_check(
            val_metrics, epoch, best_metrics, early_stopping_patience, early_stopping_delta
        )
        
        # Save checkpoint if improved
        if improved:
            checkpoint = {
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'best_metrics': best_metrics, 'config': config
            }
            if optimizer:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pt'))
            print(f"Best model saved (val_loss: {val_loss:.6f})")
        
        # Early stopping
        if stop_training:
            print(f"Early stopping after {epoch+1} epochs")
            break
        
        # Memory cleanup
        if (epoch + 1) % 10 == 0:
            model = optimize_memory_usage(model, device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Load best model for testing
    try:
        best_checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'), map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {best_checkpoint['epoch']+1}")
    except Exception as e:
        print(f"Could not load best model: {e}")
    
    # Test the model
    print("\nEvaluating model on test set...")
    test_metrics = test_model(model, test_loader, criterion, device, config)
    
    # Print test results
    print("\n" + "="*50)
    print("FINAL TEST RESULTS")
    print("="*50)
    for metric, value in test_metrics.items():
        if metric not in ['predictions', 'targets']:
            if isinstance(value, float):
                print(f"{metric.upper()}: {value:.6f}")
            else:
                print(f"{metric.upper()}: {value}")
    
    # Generate performance report
    try:
        report_metrics = {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'], 
            'train_accuracy': history['train_accuracy'],
            'val_accuracy': history['val_accuracy']
        }
        
        if history.get('direction_accuracy'):
            report_metrics['direction_accuracy'] = history['direction_accuracy']
        
        # Add test metrics
        for key, value in test_metrics.items():
            if key not in ['predictions', 'targets'] and isinstance(value, (int, float)):
                report_metrics[f'test_{key}'] = value
        
        generate_performance_report(
            train_loss=history['train_loss'][-1] if history['train_loss'] else 0,
            val_loss=history['val_loss'][-1] if history['val_loss'] else 0,
            test_loss=test_metrics.get('loss', 0),
            metrics=report_metrics,
            output_dir='docs/'
        )
        
        print("Performance report generated in docs/")
        
    except Exception as e:
        print(f"Error generating performance report: {e}")
    
    # Final analysis
    print("\n" + "="*50)
    print("TRAINING ANALYSIS")
    print("="*50)
    
    # Overfitting analysis
    if history['train_loss'] and history['val_loss']:
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history['val_loss'][-1]
        overfitting_ratio = final_val_loss / final_train_loss if final_train_loss > 0 else 1.0
        
        print(f"Overfitting ratio (val_loss/train_loss): {overfitting_ratio:.2f}")
        if overfitting_ratio > 1.5:
            print("⚠️  High overfitting detected")
        elif overfitting_ratio > 1.2:
            print("⚠️  Moderate overfitting detected")
        else:
            print("✅ Good generalization")
    
    print("\nTraining completed successfully! 🎉")

if __name__ == "__main__":
    main()
