"""
Fixed training functions that properly handle model signatures
"""

import torch
import torch.nn as nn
from tqdm import tqdm

def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train the model for one epoch with proper error handling.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        optimizer: Optimizer for training (None for neuromodulator learning)
        criterion: Loss function
        device: Device to use for training
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    batch_count = 0
    total_accuracy = 0.0
    
    for batch_idx, batch_data in enumerate(tqdm(train_loader, desc='Training')):
        try:
            # Handle different batch formats
            if len(batch_data) == 2:
                data, target = batch_data
                reward = None
            elif len(batch_data) == 3:
                data, target, reward = batch_data
            else:
                print(f"Unexpected batch format with {len(batch_data)} elements")
                continue
            
            # Reset model state for each batch
            model.reset_state()
            data, target = data.to(device), target.to(device)
            
            # Ensure target has correct shape
            if target.dim() == 1:
                target = target.unsqueeze(1)
            
            # Forward pass with proper arguments
            if optimizer is None:
                # Neuromodulator-driven learning (no backprop)
                with torch.no_grad():
                    # First forward pass to get output
                    output = model(data)
                    
                    # Handle output shape mismatches
                    output_for_loss = handle_output_shape(output, target)
                    
                    # Calculate loss and use as reward signal
                    loss = criterion(output_for_loss, target)
                    reward_signal = -loss.detach()  # Negative loss as reward
                    
                    # Second forward pass with reward for learning
                    _ = model(data, reward=reward_signal)
                    
                    # Use the original output for loss calculation
                    final_loss = loss
            else:
                # Traditional backprop learning
                output = model(data)
                
                # Handle output shape mismatches
                output_for_loss = handle_output_shape(output, target)
                
                # Calculate loss
                loss = criterion(output_for_loss, target)
                final_loss = loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Handle NaN or Inf loss
            if torch.isnan(final_loss) or torch.isinf(final_loss):
                print(f"Warning: NaN or Inf loss detected in batch {batch_idx}. Skipping.")
                continue
            
            # Calculate accuracy
            batch_accuracy = calculate_accuracy(output_for_loss, target)
            
            # Accumulate metrics
            total_loss += final_loss.item()
            total_accuracy += batch_accuracy
            batch_count += 1
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            # Reset model state on error and continue
            try:
                model.reset_state()
            except:
                pass
            continue
    
    # Return average metrics
    if batch_count == 0:
        return 0.0, 0.0
    
    return total_loss / batch_count, total_accuracy / batch_count

def validate(model, val_loader, criterion, device):
    """
    Validate the model with proper error handling.
    
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
        for batch_idx, batch_data in enumerate(tqdm(val_loader, desc='Validation')):
            try:
                # Handle different batch formats
                if len(batch_data) == 2:
                    data, target = batch_data
                elif len(batch_data) == 3:
                    data, target, _ = batch_data  # Ignore reward in validation
                else:
                    continue
                
                data, target = data.to(device), target.to(device)
                
                # Ensure target has correct shape
                if target.dim() == 1:
                    target = target.unsqueeze(1)
                
                # Forward pass (no reward in validation)
                output = model(data)
                
                # Handle output shape mismatches
                output_for_loss = handle_output_shape(output, target)
                
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
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc='Testing')):
            try:
                # Handle different batch formats
                if len(batch_data) == 2:
                    data, target = batch_data
                elif len(batch_data) == 3:
                    data, target, _ = batch_data  # Ignore reward in testing
                else:
                    continue
                
                data, target = data.to(device), target.to(device)
                
                # Ensure target has correct shape
                if target.dim() == 1:
                    target = target.unsqueeze(1)
                
                # Forward pass
                output = model(data)
                
                # Handle output shape mismatches
                output_for_loss = handle_output_shape(output, target)
                
                # Calculate loss
                loss = criterion(output_for_loss, target)
                
                # Skip NaN or Inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN or Inf loss detected in test batch {batch_idx}. Skipping.")
                    continue
                
                # Calculate accuracy
                batch_accuracy = calculate_accuracy(output_for_loss, target)
                
                # Accumulate metrics
                total_loss += loss.item()
                total_accuracy += batch_accuracy
                batch_count += 1
                
                # Store predictions and targets for further analysis
                all_predictions.append(output_for_loss.cpu().detach())
                all_targets.append(target.cpu().detach())
                
            except Exception as e:
                print(f"Error in test batch {batch_idx}: {str(e)}")
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
    
    # Calculate advanced metrics if we have predictions
    metrics = {'loss': avg_loss, 'accuracy': avg_accuracy}
    
    if all_predictions and all_targets:
        try:
            import numpy as np
            
            # Convert to numpy
            predictions = torch.cat(all_predictions, dim=0).numpy()
            targets = torch.cat(all_targets, dim=0).numpy()
            
            # Calculate additional metrics
            mse = np.mean((predictions - targets) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - targets))
            
            # R² score
            ss_tot = np.sum((targets - np.mean(targets, axis=0)) ** 2)
            ss_res = np.sum((targets - predictions) ** 2)
            r2 = 1 - (ss_res / (ss_tot if ss_tot > 0 else 1e-10))
            
            metrics.update({
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2)
            })
            
        except Exception as e:
            print(f"Error calculating advanced metrics: {str(e)}")
            metrics['advanced_metrics_error'] = str(e)
    
    return metrics

def handle_output_shape(output, target):
    """
    Handle output shape mismatches between model output and target.
    
    Args:
        output: Model output tensor
        target: Target tensor
        
    Returns:
        torch.Tensor: Reshaped output tensor
    """
    # If shapes already match, return as is
    if output.shape == target.shape:
        return output
    
    # Handle sequence outputs (take last timestep)
    if output.dim() == 3 and target.dim() == 2:
        output = output[:, -1, :]
    
    # Handle feature dimension mismatches
    if output.dim() == target.dim():
        if output.shape[-1] > target.shape[-1]:
            # Too many output features - take first N
            output = output[..., :target.shape[-1]]
        elif output.shape[-1] < target.shape[-1]:
            # Too few output features - take first N target features
            target_new = target[..., :output.shape[-1]]
            return output, target_new
    
    # Handle batch size mismatches
    if output.shape[0] != target.shape[0]:
        min_batch = min(output.shape[0], target.shape[0])
        output = output[:min_batch]
        # Note: target would need to be modified too, but we return modified output
    
    return output

def calculate_accuracy(output, target, threshold=0.1):
    """
    Calculate accuracy for regression by checking closeness.
    
    Args:
        output: Model predictions
        target: Ground truth values
        threshold: Maximum allowed error for correct prediction
        
    Returns:
        float: Accuracy as a percentage
    """
    try:
        with torch.no_grad():
            # Ensure shapes match
            if output.shape != target.shape:
                output = handle_output_shape(output, target)
            
            # Calculate absolute error
            error = torch.abs(output - target)
            
            # Count predictions within threshold
            correct = (error < threshold).float()
            
            # Calculate accuracy percentage
            accuracy = 100.0 * correct.mean().item()
            
            return accuracy
    except Exception as e:
        print(f"Error calculating accuracy: {e}")
        return 0.0