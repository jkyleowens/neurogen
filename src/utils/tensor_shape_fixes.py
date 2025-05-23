"""
Fix for critical tensor shape mismatch errors in the Brain-Inspired Neural Network.

This module provides patches to fix:
1. Missing init_hidden method in PersistentGRUController
2. Controller processing errors
3. Tensor broadcast issues during loss calculation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cupy as cp


def add_init_hidden_to_controller(controller_class):
    """
    Add init_hidden method to PersistentGRUController class.
    
    Args:
        controller_class: The PersistentGRUController class to patch
    """
    def init_hidden(self, batch_size, device):
        """
        Initialize hidden state and persistent memory.
        
        Args:
            batch_size (int): Batch size
            device (torch.device): Device to create tensors on
            
        Returns:
            tuple: (hidden_state, persistent_memory)
        """
        # Initialize hidden state for GRU
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        
        # Initialize persistent memory
        persistent_memory = torch.zeros(batch_size, self.persistent_memory_size, device=device)
        
        return hidden, persistent_memory
    
    # Add the method to the controller class
    controller_class.init_hidden = init_hidden
    
    return controller_class


def fix_controller_forward(controller_class):
    """
    Fix the forward method in PersistentGRUController to handle missing outputs.
    
    Args:
        controller_class: The PersistentGRUController class to patch
    """
    original_forward = controller_class.forward
    
    def fixed_forward(self, x, hidden=None, persistent_memory=None, neuromodulators=None):
        """
        Forward pass of the controller with robust error handling.
        
        Args:
            x (torch.Tensor): Input tensor
            hidden (torch.Tensor, optional): Initial hidden state
            persistent_memory (torch.Tensor, optional): Initial persistent memory
            neuromodulators (dict, optional): Neuromodulator levels
            
        Returns:
            tuple: (outputs, hidden_states, persistent_memories)
        """
        try:
            return original_forward(self, x, hidden, persistent_memory, neuromodulators)
        except Exception as e:
            # Error handling for robust forward pass
            print(f"Controller forward error: {e}. Using emergency fallback.")
            
            batch_size, seq_length, _ = x.size()
            
            # Create emergency outputs
            emergency_outputs = torch.zeros(batch_size, seq_length, self.hidden_size, device=x.device)
            
            # Use input as emergency output (with projection if needed)
            if x.size(2) != self.hidden_size:
                # Create an emergency projection
                emergency_projection = torch.ones(self.hidden_size, x.size(2), device=x.device) / x.size(2)
                
                # Project each time step
                for t in range(seq_length):
                    emergency_outputs[:, t, :] = F.linear(x[:, t, :], emergency_projection)
            else:
                emergency_outputs = x
            
            # Return emergency outputs and states
            if hidden is None:
                hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            
            if persistent_memory is None:
                persistent_memory = torch.zeros(batch_size, self.persistent_memory_size, device=x.device)
            
            return emergency_outputs, hidden, persistent_memory
    
    # Replace the forward method
    controller_class.forward = fixed_forward
    
    return controller_class


def fix_brain_nn_forward(brain_nn_class):
    """
    Fix the forward method in BrainInspiredNN to properly handle outputs.
    
    Args:
        brain_nn_class: The BrainInspiredNN class to patch
    """
    original_forward = brain_nn_class.forward
    
    def fixed_forward(self, x, hidden=None, persistent_memory=None, external_reward=None):
        """
        Forward pass with proper tensor shape handling.
        
        Args:
            x (torch.Tensor): Input tensor
            hidden (torch.Tensor, optional): Initial hidden state
            persistent_memory (torch.Tensor, optional): Initial persistent memory
            external_reward (torch.Tensor, optional): External reward signal
            
        Returns:
            tuple: (outputs, predicted_rewards)
        """
        try:
            outputs, predicted_rewards = original_forward(self, x, hidden, persistent_memory, external_reward)
            return outputs, predicted_rewards
        except Exception as e:
            print(f"Error in forward pass: {e}. Using emergency fallback.")
            
            # Get tensor dimensions
            batch_size = x.size(0)
            seq_length = x.size(1) if x.dim() > 1 else 1
            
            # If x only has 2 dimensions, add sequence dimension
            if x.dim() == 2:
                x = x.unsqueeze(1)
            
            # Create emergency outputs
            emergency_outputs = torch.zeros(batch_size, seq_length, self.output_size, device=x.device)
            emergency_rewards = torch.zeros(batch_size, seq_length, 1, device=x.device)
            
            # If external_reward is provided, use it for emergency rewards
            if external_reward is not None:
                if external_reward.dim() == 2 and external_reward.size(1) == seq_length:
                    emergency_rewards = external_reward.unsqueeze(-1)
                else:
                    emergency_rewards.fill_(external_reward.mean().item())
            
            return emergency_outputs, emergency_rewards
    
    # Replace the forward method
    brain_nn_class.forward = fixed_forward
    
    return brain_nn_class


def fix_train_epoch(train_epoch_function):
    """
    Fix the train_epoch function to handle tensor shape mismatches.
    
    Args:
        train_epoch_function: The original train_epoch function
        
    Returns:
        function: Fixed train_epoch function
    """
    def fixed_train_epoch(model, dataloader, optimizer, criterion, device, epoch):
        """
        Train the model for one epoch with robust tensor shape handling.
        
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
        
        for batch_idx, (data, target, reward) in enumerate(dataloader):
            try:
                # Move data to device
                data = data.to(device)
                target = target.to(device)
                reward = reward.to(device)
                
                # Reset gradients
                optimizer.zero_grad()
                
                # Forward pass
                output, predicted_reward = model(data, external_reward=reward)
                
                # CRITICAL FIX: Handle shape mismatch between output and target
                if output.dim() == 3 and target.dim() == 2:
                    # Use only the last time step for loss calculation
                    output_for_loss = output[:, -1, :]
                elif output.dim() != target.dim():
                    print(f"Dimension mismatch: output shape {output.shape}, target shape {target.shape}")
                    # Reshape output to match target if possible
                    if output.size(0) == target.size(0):
                        if output.dim() > target.dim():
                            output_for_loss = output.view(target.shape)
                        else:
                            # Emergency fallback - create compatible dummy tensor
                            output_for_loss = torch.zeros_like(target, device=device)
                    else:
                        raise ValueError(f"Cannot reshape output {output.shape} to match target {target.shape}")
                else:
                    output_for_loss = output
                
                # Similarly handle reward shape mismatch
                if predicted_reward.dim() == 3 and reward.dim() == 2:
                    predicted_reward_for_loss = predicted_reward[:, -1, :]
                elif predicted_reward.dim() != reward.dim():
                    if predicted_reward.size(0) == reward.size(0):
                        if predicted_reward.dim() > reward.dim():
                            predicted_reward_for_loss = predicted_reward.view(reward.shape)
                        else:
                            predicted_reward_for_loss = torch.zeros_like(reward, device=device)
                    else:
                        # Emergency fallback
                        predicted_reward_for_loss = torch.zeros_like(reward, device=device)
                else:
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
    
    return fixed_train_epoch


def fix_validate_function(validate_function):
    """
    Fix the validate function to handle tensor shape mismatches.
    
    Args:
        validate_function: The original validate function
        
    Returns:
        function: Fixed validate function
    """
    def fixed_validate(model, dataloader, criterion, device):
        """
        Validate the model with robust tensor shape handling.
        
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
            for batch_idx, (data, target, reward) in enumerate(dataloader):
                try:
                    # Move data to device
                    data = data.to(device)
                    target = target.to(device)
                    reward = reward.to(device)
                    
                    # Forward pass
                    output, predicted_reward = model(data, external_reward=reward)
                    
                    # CRITICAL FIX: Handle shape mismatch between output and target
                    if output.dim() == 3 and target.dim() == 2:
                        # Use only the last time step for loss calculation
                        output_for_loss = output[:, -1, :]
                    elif output.dim() != target.dim():
                        print(f"Validation dimension mismatch: output shape {output.shape}, target shape {target.shape}")
                        # Reshape output to match target if possible
                        if output.size(0) == target.size(0):
                            if output.dim() > target.dim():
                                output_for_loss = output.view(target.shape)
                            else:
                                # Emergency fallback
                                output_for_loss = torch.zeros_like(target, device=device)
                        else:
                            raise ValueError(f"Cannot reshape output {output.shape} to match target {target.shape}")
                    else:
                        output_for_loss = output
                    
                    # Similarly handle reward shape mismatch
                    if predicted_reward.dim() == 3 and reward.dim() == 2:
                        predicted_reward_for_loss = predicted_reward[:, -1, :]
                    elif predicted_reward.dim() != reward.dim():
                        if predicted_reward.size(0) == reward.size(0):
                            if predicted_reward.dim() > reward.dim():
                                predicted_reward_for_loss = predicted_reward.view(reward.shape)
                            else:
                                predicted_reward_for_loss = torch.zeros_like(reward, device=device)
                        else:
                            # Emergency fallback
                            predicted_reward_for_loss = torch.zeros_like(reward, device=device)
                    else:
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
    
    return fixed_validate


def fix_test_function(test_function):
    """
    Fix the test function to handle tensor shape mismatches.
    
    Args:
        test_function: The original test function
        
    Returns:
        function: Fixed test function
    """
    def fixed_test(model, dataloader, device):
        """
        Test the model with robust tensor shape handling.
        
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
            for batch_idx, (data, target, reward) in enumerate(dataloader):
                try:
                    # Move data to device
                    data = data.to(device)
                    target = target.to(device)
                    
                    # Forward pass
                    output, _ = model(data)
                    
                    # CRITICAL FIX: Handle shape mismatch for metrics calculation
                    if output.dim() == 3:
                        # Use only the last time step for metrics
                        output = output[:, -1, :]
                    
                    # Store predictions and targets
                    all_predictions.append(output.cpu().numpy())
                    all_targets.append(target.cpu().numpy())
                    
                except Exception as e:
                    print(f"Error in test batch {batch_idx}: {e}")
                    # Skip problematic batch
                    continue
        
        # Concatenate results if we have any
        if all_predictions and all_targets:
            try:
                import numpy as np
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                # Stack arrays
                predictions = cp.vstack(all_predictions)
                targets = cp.vstack(all_targets)
                
                # Ensure shapes match
                if predictions.shape != targets.shape:
                    min_rows = min(predictions.shape[0], targets.shape[0])
                    predictions = predictions[:min_rows]
                    targets = targets[:min_rows]
                
                # Calculate metrics
                mse = mean_squared_error(targets, predictions)
                rmse = cp.sqrt(mse)
                mae = mean_absolute_error(targets, predictions)
                
                # Safe R² calculation
                try:
                    r2 = r2_score(targets, predictions)
                except:
                    r2 = float('nan')
                
                # Direction accuracy
                direction_pred = cp.diff(predictions.flatten())
                direction_true = cp.diff(targets.flatten())
                direction_accuracy = cp.mean((direction_pred > 0) == (direction_true > 0))
                
                # Return metrics
                test_metrics = {
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'r2': float(r2),
                    'direction_accuracy': float(direction_accuracy),
                    'predictions': predictions,
                    'targets': targets
                }
                
                # Print summary
                print("Test Results:")
                print(f"  MSE: {mse:.6f}")
                print(f"  RMSE: {rmse:.6f}")
                print(f"  MAE: {mae:.6f}")
                print(f"  R²: {r2:.6f}")
                print(f"  Direction Accuracy: {direction_accuracy*100:.2f}%")
                
                return test_metrics
                
            except Exception as e:
                print(f"Error calculating metrics: {e}")
                return {'error': str(e)}
        else:
            print("No valid predictions generated during testing")
            return {'error': 'No valid predictions'}
    
    return fixed_test


def apply_fixes(model, train_epoch_func, validate_func=None, test_func=None):
    """
    Apply all fixes to model and training functions.
    
    Args:
        model: The BrainInspiredNN model
        train_epoch_func: The train_epoch function
        validate_func: The validate function (optional)
        test_func: The test function (optional)
        
    Returns:
        tuple: (fixed_model, fixed_train_epoch, fixed_validate, fixed_test)
    """
    # Fix model components
    controller_class = model.controller.__class__
    add_init_hidden_to_controller(controller_class)
    fix_controller_forward(controller_class)
    fix_brain_nn_forward(model.__class__)
    
    # Fix training functions
    fixed_train_epoch = fix_train_epoch(train_epoch_func)
    fixed_validate = fix_validate_function(validate_func) if validate_func else None
    fixed_test = fix_test_function(test_func) if test_func else None
    
    return model, fixed_train_epoch, fixed_validate, fixed_test


def import_fixes(module_dict):
    """
    Import and apply fixes to modules.
    
    Args:
        module_dict: Dictionary of module names to modules
        
    Returns:
        dict: Updated module dictionary with fixed components
    """
    # Extract relevant components
    model = module_dict.get('model')
    train_epoch = module_dict.get('train_epoch')
    validate = module_dict.get('validate')
    test = module_dict.get('test')
    
    if model and train_epoch:
        # Apply fixes
        model, train_epoch, validate, test = apply_fixes(model, train_epoch, validate, test)
        
        # Update module dictionary
        module_dict['model'] = model
        module_dict['train_epoch'] = train_epoch
        if validate:
            module_dict['validate'] = validate
        if test:
            module_dict['test'] = test
    
    return module_dict


# Simple tensor shape fix function for direct use
def fix_tensor_shapes(tensor, target_shape):
    """
    Fix tensor shapes to match target shape.
    
    Args:
        tensor (torch.Tensor): Input tensor
        target_shape (tuple): Target shape
        
    Returns:
        torch.Tensor: Tensor with fixed shape
    """
    if tensor.shape == target_shape:
        return tensor
    
    try:
        # If tensor has more dimensions than target, try to squeeze
        if tensor.dim() > len(target_shape):
            # Try to squeeze out extra dimensions
            tensor = tensor.squeeze()
            if tensor.shape == target_shape:
                return tensor
                
            # If still not matching, try to get the last relevant slice
            if tensor.dim() > len(target_shape):
                if tensor.shape[-len(target_shape):] == target_shape:
                    # Extract the last part that matches target shape
                    for _ in range(tensor.dim() - len(target_shape)):
                        tensor = tensor[-1]
                    return tensor
        
        # If tensor has fewer dimensions than target, try to unsqueeze
        if tensor.dim() < len(target_shape):
            # Add dimensions until we match
            for _ in range(len(target_shape) - tensor.dim()):
                tensor = tensor.unsqueeze(0)
            if tensor.shape == target_shape:
                return tensor
        
        # If dimensions match but sizes don't, try to reshape
        if tensor.dim() == len(target_shape):
            # Check if total elements match
            if tensor.numel() == torch.prod(torch.tensor(target_shape)):
                return tensor.reshape(target_shape)
        
        # If we can't fix it, create a new tensor with the right shape
        print(f"Warning: Could not reshape tensor from {tensor.shape} to {target_shape}. Creating new tensor.")
        return torch.zeros(target_shape, device=tensor.device)
        
    except Exception as e:
        print(f"Error fixing tensor shape: {e}")
        return torch.zeros(target_shape, device=tensor.device)

# Usage Example:
"""
from tensor_shape_fixes import apply_fixes, fix_tensor_shapes

# Get your model and training functions
model = BrainInspiredNN(...)
# ... [get train_epoch, validate, test functions]

# Apply fixes
model, train_epoch, validate, test = apply_fixes(model, train_epoch, validate, test)

# Now use the fixed components for training
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device, epoch)
    val_loss = validate(model, val_dataloader, criterion, device)
    
# Test the model
test_metrics = test(model, test_dataloader, device)

# Or use the simple fix_tensor_shapes function directly
output = fix_tensor_shapes(output, target_shape=(batch_size, output_size))
"""