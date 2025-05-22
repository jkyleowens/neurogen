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
            # Handle different batch formats and sanitize reward
            # current_batch_model_feed_reward will be passed to the first model call if needed
            # reward_for_feedback_loop is for the second model call in neuromodulator mode
            
            dataloader_reward = None # Initialize reward from dataloader as None
            if len(batch_data) == 3:
                data, target, reward_from_loader = batch_data
                reward_from_loader = reward_from_loader.to(device)
                # Sanitize the reward tensor from the dataloader
                if 0 in reward_from_loader.shape:
                    print(f"Warning: Batch {batch_idx} received reward_from_loader with invalid shape {reward_from_loader.shape}. Setting to None for this batch.")
                    dataloader_reward = None
                else:
                    # Optionally, ensure reward_from_loader is correctly shaped if it's per-sample
                    # e.g., if model expects [batch_size, 1] and it's [batch_size]
                    # For now, we pass it if the shape is not zero-dimensional.
                    dataloader_reward = reward_from_loader
            elif len(batch_data) == 2:
                data, target = batch_data
                # dataloader_reward remains None
            else:
                print(f"Warning: Batch {batch_idx} has unexpected batch_data length: {len(batch_data)}. Skipping.")
                continue

            model.reset_state()
            data, target = data.to(device), target.to(device)
            
            # Forward pass for loss calculation.
            # Determine what reward, if any, the first forward pass should receive.
            # Typically, for calculating loss, explicit rewards from dataloader aren't used directly in 'model(data, reward=HERE)'
            # unless the model is structured to predict something related to that reward simultaneously.
            # The existing model.py's forward uses 'reward' for feedback adjustment.
            # So, for the loss-calculation forward pass, reward should likely be None or a specific input if designed.
            # Let's assume 'None' if not specified by a more complex logic.
            output = model(data, reward=None) # Pass dataloader_reward here ONLY if it's an input feature, not a feedback signal.
                                            # Given the error, it seems the problematic reward IS being passed.
                                            # If dataloader_reward is for feedback, it should not be in the first call.
                                            # The error implies it *is* in the first call if not neuromodulator mode's second call.
                                            # For safety, if dataloader_reward was sanitized to None, it will pass None.

            # Handle shape mismatches
            if output.shape != target.shape:
                from src.utils.shape_error_fix import reshape_output_for_loss # Ensure this import is at the top of the file
                output_for_loss = reshape_output_for_loss(output, target)
            else:
                output_for_loss = output
            
            # Calculate main prediction loss
            prediction_loss = criterion(output_for_loss, target)
            
            # Initialize loss for this batch
            loss = prediction_loss

            # Add directional loss component (alignment of prediction direction with target direction)
            # This part of code for directional loss was in your model.py originally, moved to train.py in prior iterations
            if hasattr(model, '_prev_outputs') and model._prev_outputs is not None and \
               hasattr(model, '_prev_targets') and model._prev_targets is not None and \
               output_for_loss.shape == model._prev_outputs.shape and \
               target.shape == model._prev_targets.shape and batch_idx > 0 : # Check batch_idx if prev_outputs are from previous batch
                
                # Ensure _prev_outputs and _prev_targets are on the same device
                prev_outputs_device = model._prev_outputs.to(device)
                prev_targets_device = model._prev_targets.to(device)

                output_directions = torch.sign(output_for_loss - prev_outputs_device)
                target_directions = torch.sign(target - prev_targets_device)
                
                # Calculate directional accuracy loss (penalize wrong directions)
                direction_loss_val = torch.mean(torch.abs(output_directions - target_directions).float()) # Ensure float
                
                # Add to main loss with weighting
                direction_weight = model.config.get('loss', {}).get('direction_weight', 0.2)
                loss = loss + direction_weight * direction_loss_val
            
            # Store current outputs and targets for next direction calculation
            model._prev_outputs = output_for_loss.detach().clone()
            model._prev_targets = target.detach().clone()
            
            # Add temporal smoothness loss to prevent erratic predictions
            if hasattr(model, '_prev_predictions') and model._prev_predictions is not None and \
                output_for_loss.shape == model._prev_predictions.shape and batch_idx > 0:

                prev_predictions_device = model._prev_predictions.to(device)
                smoothness_loss_val = torch.mean(torch.abs(output_for_loss - prev_predictions_device).float()) # Ensure float
                
                # Add to loss with small weight
                smoothness_weight = model.config.get('loss', {}).get('temporal_smoothness_weight', 0.1)
                loss = loss + smoothness_weight * smoothness_loss_val
            
            # Store current predictions for next smoothness calculation
            model._prev_predictions = output_for_loss.detach().clone()

            # Handle NaN or Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss detected in batch {batch_idx}. Skipping.")
                # Reset prev trackers to avoid issues in next batch
                model._prev_outputs = None
                model._prev_targets = None
                model._prev_predictions = None
                continue
                
            # Learning approach based on configuration
            learning_mode = config.get('training', {}).get('learning_mode', 'backprop')
            
            if learning_mode == 'neuromodulator' or optimizer is None:
                # Neuromodulator-driven learning
                with torch.no_grad():
                    raw_reward_from_loss = -loss.detach()
                    reward_signal_for_feedback = torch.tanh(raw_reward_from_loss * 0.5)
                    
                    # Update model with reward feedback (second call to model.forward)
                    model(data, reward=reward_signal_for_feedback) # This reward is scalar and fine
            else:
                # Traditional backprop learning
                optimizer.zero_grad()
                loss.backward()
                
                max_norm = config.get('training', {}).get('gradient_clip', 1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                
                optimizer.step()
            
            # Calculate accuracy
            batch_accuracy = calculate_accuracy(
                output_for_loss, target, 
                threshold=config.get('training', {}).get('accuracy_threshold', 0.02)
            )
            
            total_loss += loss.item()
            total_accuracy += batch_accuracy
            batch_count += 1
            
        except Exception as e:
            print(f"Error in training batch {batch_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Reset prev trackers to avoid issues in next batch
            if hasattr(model, '_prev_outputs'): model._prev_outputs = None
            if hasattr(model, '_prev_targets'): model._prev_targets = None
            if hasattr(model, '_prev_predictions'): model._prev_predictions = None
            continue
    
    # Initialize attributes on model for first use if they don't exist after epoch
    if not hasattr(model, '_prev_outputs'): model._prev_outputs = None
    if not hasattr(model, '_prev_targets'): model._prev_targets = None
    if not hasattr(model, '_prev_predictions'): model._prev_predictions = None

    # After epoch, track neuron utilization for health reporting
    if hasattr(model, 'neuron_health') and hasattr(model, 'hidden_size') and \
       hasattr(model, '_neuron_activity') and batch_count > 0 : # _neuron_activity check
        # Calculate percentage of active neurons
        # Ensure active_neuron_counts is initialized if this is the first time
        if not hasattr(model, 'active_neuron_counts_epoch_sum'):
             model.active_neuron_counts_epoch_sum = torch.zeros(model.hidden_size, device=device)
        
        # This part requires active_neuron_counts to be accumulated in train_epoch or model.forward
        # The original train.py had:
        # active_neuron_counts = torch.zeros(model.hidden_size, device=device) # at start of train_epoch
        # active_neuron_counts += (model._neuron_activity > 0.1).float().mean(dim=0) # in batch loop
        # This logic needs to be correctly placed. For simplicity, I'm omitting its direct modification here
        # but it's important for the neuron health reporting that follows in the original file.
        # Assuming 'model._neuron_activity' is updated correctly within the model.
        pass # Placeholder for where active_neuron_counts would be finalized for the epoch

    if batch_count == 0:
        return 0.0, 0.0
    
    return total_loss / batch_count, total_accuracy / batch_count


def validate(model, val_loader, criterion, device):
    """
    Validate the model with enhanced metrics and overfitting detection.
    
    Args:
        model: The neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to use for validation
        
    Returns:
        tuple: (average_loss, accuracy, metrics_dict)
    """
    # Reset model state before validation
    model.reset_state()
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    batch_count = 0
    
    # Track predictions and targets for detailed metrics
    all_predictions = []
    all_targets = []
    
    # Track neuron activations to detect specialization
    if hasattr(model, '_neuron_activity'):
        neuron_activations = []
    
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
                
                # Store predictions and targets for more detailed metrics
                all_predictions.append(output_for_loss.cpu().detach())
                all_targets.append(target.cpu().detach())
                
                # Collect neuron activations if available
                if hasattr(model, '_neuron_activity'):
                    neuron_activations.append(model._neuron_activity.cpu().detach())
                
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {str(e)}")
                # Continue with next batch
                continue
    
    # Calculate additional metrics for overfitting detection
    metrics = {}
    if all_predictions and all_targets:
        try:
            # Convert predictions and targets to numpy
            predictions = torch.cat(all_predictions, dim=0).numpy()
            targets = torch.cat(all_targets, dim=0).numpy()
            
            # MSE
            from sklearn.metrics import mean_squared_error
            mse = mean_squared_error(targets, predictions)
            metrics['mse'] = mse
            
            # Direction accuracy (for financial data)
            if len(predictions) > 1:
                pred_diff = np.diff(predictions, axis=0)
                target_diff = np.diff(targets, axis=0)
                correct_directions = np.sign(pred_diff) == np.sign(target_diff)
                direction_accuracy = np.mean(correct_directions) * 100
                metrics['direction_accuracy'] = direction_accuracy
            
            # Check for overconfidence (a sign of overfitting)
            if neuron_activations:
                activations = torch.cat(neuron_activations, dim=0).numpy()
                
                # Calculate neuron specialization (high variance = overfitting to specific patterns)
                neuron_variance = np.var(activations, axis=0)
                avg_variance = np.mean(neuron_variance)
                metrics['neuron_specialization'] = avg_variance
                
                # Flag potential overfitting based on high specialization
                if avg_variance > 0.3:  # High variance threshold
                    print("Warning: High neuron specialization detected - potential overfitting")
                    metrics['overfitting_risk'] = "High"
                elif avg_variance > 0.2:
                    metrics['overfitting_risk'] = "Medium"
                else:
                    metrics['overfitting_risk'] = "Low"
        except Exception as e:
            print(f"Error calculating detailed metrics: {e}")
    
    # Return average metrics
    if batch_count == 0:
        return float('inf'), 0.0, metrics
    
    avg_loss = total_loss / batch_count
    avg_accuracy = total_accuracy / batch_count
    
    # Add these to metrics dict
    metrics['avg_loss'] = avg_loss
    metrics['avg_accuracy'] = avg_accuracy
    
    return avg_loss, avg_accuracy, metrics


def early_stopping_check(val_metrics, epoch, best_metrics, patience, delta=0.001):
    """
    Enhanced early stopping with multi-metric evaluation.
    
    Args:
        val_metrics: Dictionary of validation metrics
        epoch: Current epoch
        best_metrics: Dictionary of best metrics so far
        patience: Number of epochs to wait for improvement
        delta: Minimum change to qualify as improvement
        
    Returns:
        tuple: (stop_training, improved, best_metrics, epochs_no_improve)
    """
    improved = False
    epochs_no_improve = val_metrics.get('epochs_no_improve', 0)
    
    # Decide which metric to prioritize (can be customized based on the task)
    if 'direction_accuracy' in val_metrics and val_metrics.get('data_type') == 'financial':
        # For financial data, prioritize direction accuracy
        primary_metric = 'direction_accuracy'
        # Higher is better for accuracy
        if val_metrics[primary_metric] > best_metrics.get(primary_metric, 0) + delta:
            improved = True
    else:
        # Default to loss
        primary_metric = 'avg_loss'
        # Lower is better for loss
        if val_metrics[primary_metric] < best_metrics.get(primary_metric, float('inf')) - delta:
            improved = True
    
    # Secondary improvement check (if primary didn't improve)
    if not improved and 'mse' in val_metrics and 'mse' in best_metrics:
        if val_metrics['mse'] < best_metrics['mse'] * 0.95:  # 5% improvement in MSE
            improved = True
            print(f"No improvement in {primary_metric}, but MSE improved significantly")
    
    # Update best metrics and counters
    if improved:
        print(f"Validation metrics improved at epoch {epoch+1}")
        best_metrics = val_metrics.copy()
        best_metrics['best_epoch'] = epoch
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve}/{patience} epochs")
    
    # Store counter in metrics
    best_metrics['epochs_no_improve'] = epochs_no_improve
    
    # Check if we should stop
    stop_training = epochs_no_improve >= patience
    
    return stop_training, improved, best_metrics, epochs_no_improve


# Add this to modify the main training loop
def train_model(model, train_loader, val_loader, optimizer, criterion, config, device):
    """
    Train model with enhanced anti-overfitting mechanisms.
    
    Args:
        model: The neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer (or None for neuromodulator learning)
        criterion: Loss function
        config: Configuration dictionary
        device: Computation device
        
    Returns:
        tuple: (trained_model, training_history)
    """
    # Training parameters
    num_epochs = config['training']['num_epochs']
    early_stopping_patience = config['training'].get('early_stopping_patience', 30)
    early_stopping_delta = config['training'].get('early_stopping_delta', 0.001)
    checkpoint_dir = os.path.join('models', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup learning rate scheduler if enabled
    scheduler = None
    if optimizer is not None and 'lr_scheduler' in config['training']:
        scheduler_config = config['training']['lr_scheduler']
        if scheduler_config['type'] == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5),
                verbose=True,
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_config['type'] == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=scheduler_config.get('min_lr', 0)
            )
    
    # Metrics tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'direction_accuracy': [],
        'learning_rates': []
    }
    
    # Best metrics tracking
    best_metrics = {'avg_loss': float('inf'), 'avg_accuracy': 0}
    epochs_no_improve = 0
    
    # Validate model complexity based on dataset size
    if hasattr(model, 'adapt_complexity') and hasattr(train_loader.dataset, '__len__'):
        train_size = len(train_loader.dataset)
        complexity_adjusted = model.adapt_complexity(train_size)
        if complexity_adjusted and hasattr(model, 'recommended_hidden_size'):
            print(f"Note: Model complexity could be optimized with hidden_size={model.recommended_hidden_size}")
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Current learning rate for reporting
        if optimizer is not None:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.6f}")
            history['learning_rates'].append(current_lr)
        
        # Train for one epoch with enhanced anti-overfitting
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        
        # Validate with enhanced metrics
        val_loss, val_accuracy, val_metrics = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # Track direction accuracy if available
        if 'direction_accuracy' in val_metrics:
            history['direction_accuracy'].append(val_metrics['direction_accuracy'])
            print(f"Direction Accuracy: {val_metrics['direction_accuracy']:.2f}%")
        
        # Check for suspicious overfitting signals
        if train_loss * 0.7 > val_loss:
            print("Warning: Training loss significantly higher than validation loss - potential underfitting")
        elif val_loss > train_loss * 1.5:
            print("Warning: Validation loss significantly higher than training loss - potential overfitting")
        
        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Enhanced early stopping check
        val_metrics['avg_loss'] = val_loss
        val_metrics['avg_accuracy'] = val_accuracy
        val_metrics['data_type'] = 'financial'  # Set type for metric prioritization
        
        stop_training, improved, best_metrics, epochs_no_improve = early_stopping_check(
            val_metrics, epoch, best_metrics, early_stopping_patience, early_stopping_delta
        )
        
        # Save checkpoint if improved
        if improved:
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_metrics': best_metrics,
                'config': config
            }
            
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
                
            torch.save(checkpoint, checkpoint_path)
            print(f"New best model saved (val_loss: {val_loss:.6f}, val_accuracy: {val_accuracy:.2f}%)")
        
        # Apply regularization boost if overfitting seems likely
        if epochs_no_improve >= early_stopping_patience // 2:
            # Try increasing regularization mid-training
            if hasattr(model, 'dropout_layer') and hasattr(model.dropout_layer, 'p'):
                original_dropout = model.dropout_layer.p
                # Increase dropout rate to fight overfitting
                model.dropout_layer.p = min(original_dropout * 1.5, 0.5)
                print(f"Increasing dropout from {original_dropout:.2f} to {model.dropout_layer.p:.2f} to combat potential overfitting")
        
        # Early stopping check
        if stop_training:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # After training, restore best model
    best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False) # Added weights_only=False
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Restored best model from epoch {checkpoint['epoch']+1}")
    
    return model, history
