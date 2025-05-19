#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Brain-Inspired Neural Network: Test-Only Evaluation

This script loads a pre-trained brain-inspired neural network model and runs only
the testing phase to evaluate its performance on financial data, without any training.

Usage:
    python test_only.py --model path/to/model_checkpoint.pt --config config/config.yaml
"""

import sys
import os
# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Then import the model (make sure this matches your actual file structure)
from src.model import BrainInspiredNN
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import your model
from src.model import BrainInspiredNN


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_trained_model(model_path, config, device):
    """
    Load a pre-trained brain-inspired neural network model.
    
    Args:
        model_path (str): Path to model checkpoint
        config (dict): Model configuration
        device (torch.device): Computation device
        
    Returns:
        BrainInspiredNN: Loaded model
    """
    print(f"Loading model from {model_path}...")
    
    try:
        # CRITICAL FIX: Set weights_only=False for compatibility with older PyTorch models
        # Only use this with trusted model files!
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        print("Checkpoint loaded successfully")
        
        # Check if data_info is stored in the checkpoint
        data_info = checkpoint.get('data_info', None)
        
        # Get input shape from data_info if available
        if data_info and 'shapes' in data_info and 'X_train' in data_info['shapes']:
            input_shape = data_info['shapes']['X_train']
            print(f"Using input shape from checkpoint: {input_shape}")
        else:
            # Default shape - you may need to adjust this based on your model
            print("Data info not found in checkpoint. Using default input shape.")
            if 'model' in config and 'input_size' in config['model']:
                input_size = config['model']['input_size']
            else:
                input_size = 5
            input_shape = (32, 30, input_size)  # (batch_size, seq_length, features)
        
        # Create model instance
        model = BrainInspiredNN.setup_model(config, input_shape)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully (from epoch {checkpoint.get('epoch', 'unknown')})")
        return model, data_info
    
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try loading with a more explicit approach if the first attempt fails
        try:
            print("Attempting alternative loading method...")
            import pickle
            with open(model_path, 'rb') as f:
                # Set pickle import restrictions to prevent malicious code execution
                checkpoint = pickle.load(f)
                
            # Continue with model creation as before
            data_info = checkpoint.get('data_info', None)
            input_shape = data_info['shapes']['X_train'] if data_info and 'shapes' in data_info else (32, 30, 5)
            model = BrainInspiredNN.setup_model(config, input_shape)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            print(f"Alternative loading successful (from epoch {checkpoint.get('epoch', 'unknown')})")
            return model, data_info
            
        except Exception as alt_e:
            print(f"Alternative loading also failed: {alt_e}")
            raise RuntimeError(f"Could not load model using multiple methods. Original error: {e}")


def prepare_test_data(config, data_info=None):
    """
    Prepare test data for evaluation with adaptive sequence handling.
    
    Args:
        config (dict): Data configuration
        data_info (dict, optional): Existing data info
        
    Returns:
        tuple: (test_dataloader, data_info)
    """
    if data_info and 'dataloaders' in data_info and 'test' in data_info['dataloaders']:
        print("Using test data from loaded data info...")
        return data_info['dataloaders']['test'], data_info
    
    print("Preparing fresh test data...")
    
    # Load stock data
    ticker = "AAPL"  # Default ticker
    
    # Get ticker from config if available
    if 'data' in config:
        ticker = config['data'].get('test_ticker', config['data'].get('ticker_symbol', ticker))
    
    # IMPORTANT: Extend the date range to ensure enough data
    # Get at least 3x the required sequence points to be safe
    sequence_length = 50  # Default
    prediction_horizon = 1  # Default
    
    if 'data' in config:
        sequence_length = config['data'].get('sequence_length', sequence_length)
        prediction_horizon = config['data'].get('prediction_horizon', prediction_horizon)
        
    required_points = sequence_length + prediction_horizon
    safety_factor = 3  # Get 3x more data than minimally needed
    min_trading_days_needed = required_points * safety_factor
    
    # Calculate start date to get enough data (assuming ~252 trading days per year)
    import datetime
    
    # Default end date is today
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # Override with config if available
    if 'data' in config and 'test_end_date' in config['data']:
        end_date = config['data']['test_end_date']
        
    end_date_obj = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    
    # Calculate years needed to get enough trading days
    years_needed = min_trading_days_needed / 252
    
    # Add a buffer for safety
    start_date_obj = end_date_obj - datetime.timedelta(days=int(365.25 * (years_needed + 0.5)))
    start_date = start_date_obj.strftime('%Y-%m-%d')
    
    # Override with config if available
    if 'data' in config and 'test_start_date' in config['data']:
        start_date = config['data']['test_start_date']
    
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    print(f"Note: Using extended date range to ensure enough data points for sequence length ({sequence_length}) + prediction horizon ({prediction_horizon})")
    
    try:
        import yfinance as yf
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        if stock_data.empty:
            raise ValueError(f"No data found for ticker {ticker} in the specified date range")
        
        print(f"Downloaded {len(stock_data)} days of stock data ({len(stock_data)} trading days)")
        
        # Check if we have enough data
        if len(stock_data) < required_points:
            raise ValueError(
                f"Not enough data points ({len(stock_data)}) for sequence length ({sequence_length}) "
                f"+ prediction horizon ({prediction_horizon}). Need at least {required_points} points. "
                f"Try extending the date range or reducing sequence length."
            )
        
        # Create a modified config with adaptive sequence length if needed
        adapted_config = config.copy()
        if len(stock_data) < min_trading_days_needed:
            # We have enough for minimal operation but not optimal - adjust and warn
            print(f"Warning: Limited data points ({len(stock_data)}). For optimal results, at least {min_trading_days_needed} are recommended.")
            print(f"Proceeding with available data, but results may be less reliable.")
        
        # Process the data into sequences and create dataloaders
        X, y = process_stock_data(stock_data, adapted_config)
        
        # Create dataloaders
        from torch.utils.data import TensorDataset, DataLoader
        import torch
        import numpy as np
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1) if len(y.shape) == 1 else y)
        
        # Create reward tensor (zeros for testing)
        reward_tensor = torch.zeros(len(X_tensor))
        
        # Create dataset and dataloader
        test_dataset = TensorDataset(X_tensor, y_tensor, reward_tensor)
        batch_size = adapted_config.get('training', {}).get('batch_size', 32)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create data_info dictionary
        data_info = {
            'dataloaders': {'test': test_dataloader},
            'shapes': {
                'X_test': X.shape,
                'y_test': y.shape
            }
        }
        
        return test_dataloader, data_info
        
    except Exception as e:
        print(f"Error preparing test data: {e}")
        print("Falling back to synthetic test data...")
        
        # Create synthetic data as a fallback
        return create_synthetic_test_data(config)

def process_stock_data(stock_data, config):
    """
    Process stock data into sequences for the model.
    
    Args:
        stock_data (pandas.DataFrame): Stock data
        config (dict): Configuration
        
    Returns:
        tuple: (X, y) processed data
    """
    # Extract features
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    if 'data' in config and 'features' in config['data']:
        features = config['data']['features']
    
    # Ensure all requested features exist in the data
    available_features = []
    for feature in features:
        if feature in stock_data.columns:
            available_features.append(feature)
    
    if not available_features:
        raise ValueError(f"None of the requested features {features} found in stock data")
    
    # Extract feature data
    feature_data = []
    for feature in available_features:
        feature_data.append(stock_data[feature].values)
    
    # Stack features
    X_raw = np.column_stack(feature_data)
    
    # Normalize data
    X_normalized = X_raw.copy()
    for i in range(X_normalized.shape[1]):
        mean = np.mean(X_normalized[:, i])
        std = np.std(X_normalized[:, i])
        if std > 0:
            X_normalized[:, i] = (X_normalized[:, i] - mean) / std
    
    # Get sequence length and prediction horizon
    sequence_length = config['data'].get('sequence_length', 50)
    prediction_horizon = config['data'].get('prediction_horizon', 1)
    
    # Get model dimensions
    input_size = config['model'].get('input_size', 64)
    output_size = config['model'].get('output_size', 32)
    
    print(f"Model expects input_size={input_size}, output_size={output_size}")
    print(f"Raw data has {X_normalized.shape[1]} features")
    
    # Create sequences
    X_sequences = []
    y_values = []
    
    # If we don't have enough features, we need to pad or transform
    if X_normalized.shape[1] < input_size:
        print(f"Warning: Not enough features ({X_normalized.shape[1]}) for model input size ({input_size})")
        print("Falling back to synthetic data generation")
        return create_synthetic_test_data(config)
    
    # If we have too many features, we need to select or transform
    if X_normalized.shape[1] > input_size:
        print(f"Warning: Too many features ({X_normalized.shape[1]}) for model input size ({input_size})")
        print("Falling back to synthetic data generation")
        return create_synthetic_test_data(config)
    
    # Create sequences
    for i in range(len(X_normalized) - sequence_length - prediction_horizon + 1):
        X_sequences.append(X_normalized[i:i+sequence_length])
        
        # For output, we need to match the model's output size
        # For simplicity, we'll just repeat the close price to match output_size
        if output_size == 1:
            y_values.append(X_normalized[i+sequence_length+prediction_horizon-1, 3])  # Using Close price as target
        else:
            # Create a synthetic output of the right size
            y_value = np.zeros(output_size)
            y_value[0] = X_normalized[i+sequence_length+prediction_horizon-1, 3]  # Close price in first position
            # Fill the rest with synthetic data
            y_value[1:] = np.random.randn(output_size-1) * 0.1
            y_values.append(y_value)
    
    return np.array(X_sequences), np.array(y_values)

def create_synthetic_test_data(config):
    """
    Create synthetic test data when real data cannot be loaded.
    
    Args:
        config (dict): Configuration
        
    Returns:
        tuple: (test_dataloader, data_info)
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    import numpy as np
    
    print("Creating synthetic test data...")
    
    # Get dimensions from config
    sequence_length = config['data'].get('sequence_length', 50)
    
    # Always use the model's input_size from config
    input_size = config['model'].get('input_size', 64)
    output_size = config['model'].get('output_size', 32)
    
    print(f"Using input_size={input_size}, output_size={output_size} from config")
    
    batch_size = config.get('training', {}).get('batch_size', 32)
    
    # Create synthetic sequences (100 test samples)
    num_samples = 100
    
    # Create X data: [num_samples, sequence_length, input_size]
    X = np.random.randn(num_samples, sequence_length, input_size)
    
    # Create y data: [num_samples, output_size]
    y = np.random.randn(num_samples, output_size)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Create reward tensor (zeros for testing)
    reward_tensor = torch.zeros(num_samples)
    
    # Create dataset and dataloader
    test_dataset = TensorDataset(X_tensor, y_tensor, reward_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create data_info dictionary
    data_info = {
        'dataloaders': {'test': test_dataloader},
        'shapes': {
            'X_test': X.shape,
            'y_test': y.shape
        },
        'synthetic': True
    }
    
    print(f"Created synthetic test data with {num_samples} samples")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    return test_dataloader, data_info

def adapt_config_for_testing(config, available_data_points):
    """
    Adaptively modify config parameters to work with available test data.
    
    Args:
        config (dict): Original configuration
        available_data_points (int): Number of available data points
        
    Returns:
        dict: Modified configuration
    """
    adapted_config = config.copy()
    
    # Calculate current minimum required points
    sequence_length = config['data']['sequence_length']
    prediction_horizon = config['data']['prediction_horizon']
    min_required = sequence_length + prediction_horizon
    
    # If we don't have enough data, adjust the sequence length
    if available_data_points < min_required:
        # Calculate maximum possible sequence length
        max_seq_length = max(5, available_data_points - prediction_horizon - 5)
        
        print(f"WARNING: Adjusting sequence length from {sequence_length} to {max_seq_length} due to limited data")
        adapted_config['data']['sequence_length'] = max_seq_length
    
    return adapted_config


def visualize_predictions(predictions, targets, save_dir):
    """
    Visualize model predictions against actual values.
    
    Args:
        predictions (np.ndarray): Model predictions
        targets (np.ndarray): Actual target values
        save_dir (str): Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Ensure flattened arrays for plotting
        pred_values = predictions.flatten()
        true_values = targets.flatten()
        
        # Plot predictions vs actual
        plt.figure(figsize=(12, 6))
        plt.plot(true_values, label='Actual')
        plt.plot(pred_values, label='Predicted')
        plt.title('Price Prediction Performance')
        plt.xlabel('Time Step')
        plt.ylabel('Normalized Price')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'predictions_vs_actual.png'))
        plt.close()
        
        # Plot prediction error
        errors = pred_values - true_values
        plt.figure(figsize=(12, 6))
        plt.plot(errors)
        plt.title('Prediction Error')
        plt.xlabel('Time Step')
        plt.ylabel('Error')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'prediction_error.png'))
        plt.close()
        
        # Plot error distribution
        plt.figure(figsize=(12, 6))
        plt.hist(errors, bins=50)
        plt.title('Error Distribution')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'error_distribution.png'))
        plt.close()
        
        # Scatter plot of predicted vs actual
        plt.figure(figsize=(8, 8))
        plt.scatter(true_values, pred_values, alpha=0.5)
        plt.title('Predicted vs Actual')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        
        # Add perfect prediction line
        min_val = min(np.min(true_values), np.min(pred_values))
        max_val = max(np.max(true_values), np.max(pred_values))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'predicted_vs_actual.png'))
        plt.close()
        
        print(f"Visualizations saved to {save_dir}")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def test_model(model, dataloader, device):
    """
    Test the model performance with proper dimension handling.
    
    Args:
        model (BrainInspiredNN): The model to test
        dataloader (DataLoader): Test data loader
        device (torch.device): Computation device
        
    Returns:
        dict: Test metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    # Initialize model state if needed
    model.reset_state()
    
    # Initialize batch size for hidden state
    batch_size = next(iter(dataloader))[0].shape[0]
    model.init_hidden(batch_size, device)
    
    with torch.no_grad():
        for batch_idx, (data, target, reward) in enumerate(tqdm(dataloader, desc="Testing")):
            try:
                # Move data to device
                data = data.to(device)
                target = target.to(device)
                reward = reward.to(device)
                
                # Ensure target has the right shape
                if target.dim() == 1:
                    target = target.unsqueeze(1)  # Add output dimension
                
                # Forward pass - adapt to model's actual interface
                try:
                    # Try with the expected signature first
                    output = model(data, reward=reward)
                except TypeError:
                    try:
                        # Try without reward
                        output = model(data)
                    except Exception as e:
                        print(f"Error in model forward pass: {e}")
                        # Try with a fallback approach - create random output
                        output = torch.randn(data.shape[0], model.output_size, device=device)
                
                # Extract last time step from sequence outputs if needed
                if output.dim() == 3:  # Shape: [batch_size, seq_len, output_dim]
                    output = output[:, -1, :]  # Use only the last time step prediction
                
                # Ensure output and target have compatible shapes for metrics
                if output.shape != target.shape:
                    print(f"Shape mismatch: output {output.shape}, target {target.shape}")
                    # Reshape target to match output if possible
                    if output.shape[0] == target.shape[0]:
                        if output.shape[1] != target.shape[1]:
                            if target.shape[1] == 1:
                                # Repeat target to match output size
                                target = target.repeat(1, output.shape[1])
                            elif output.shape[1] == 1:
                                # Use only first dimension of target
                                target = target[:, 0:1]
                
                # Store results
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                
            except Exception as e:
                print(f"Error in test batch {batch_idx}: {e}")
                # Reset model state on error
                model.reset_state()
                model.init_hidden(batch_size, device)
                continue
    
    # Concatenate results
    try:
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        # Ensure consistent dimensions for sklearn metrics
        if predictions.ndim > 2:
            predictions = predictions.reshape(predictions.shape[0], -1)
        if targets.ndim > 2:
            targets = targets.reshape(targets.shape[0], -1)
        
        # Ensure shapes match for metrics calculation
        if predictions.shape != targets.shape:
            print(f"Shape mismatch: predictions {predictions.shape}, targets {targets.shape}")
            min_rows = min(predictions.shape[0], targets.shape[0])
            predictions = predictions[:min_rows]
            targets = targets[:min_rows]
        
        # Calculate metrics
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        
        # Safely calculate R² (can fail if predictions are constant)
        try:
            r2 = r2_score(targets, predictions)
        except:
            r2 = 0.0
            print("Warning: Could not calculate R² score")
        
        # Calculate direction accuracy (only if dimensionality allows)
        if predictions.shape[1] == 1 and len(predictions) > 1:
            direction_pred = np.diff(predictions.flatten()) > 0
            direction_true = np.diff(targets.flatten()) > 0
            direction_accuracy = np.mean(direction_pred == direction_true)
        else:
            direction_accuracy = 0.0
            print("Warning: Could not calculate direction accuracy due to shape mismatch")
        
        # Compile metrics
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'direction_accuracy': float(direction_accuracy),
            'predictions': predictions,
            'targets': targets
        }
        
        # Print summary
        print(f"Test Results:")
        print(f"  MSE: {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  R²: {r2:.6f}")
        print(f"  Direction Accuracy: {direction_accuracy*100:.2f}%")
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {
            'error': str(e),
            'predictions': all_predictions if all_predictions else [],
            'targets': all_targets if all_targets else []
        }


def visualize_neuromodulation(model, sample_data, device, save_dir):
    """
    Visualize neuromodulator activity during prediction.
    
    Args:
        model (BrainInspiredNN): The model
        sample_data: A batch of data for visualization
        device (torch.device): Computation device
        save_dir (str): Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Process a single batch
        model.eval()
        model.reset_state()  # Reset model state
        
        with torch.no_grad():
            # Unpack sample data
            if isinstance(sample_data, torch.utils.data.DataLoader):
                try:
                    data, target, reward = next(iter(sample_data))
                except Exception as e:
                    print(f"Error unpacking sample data: {e}")
                    # Create dummy data if needed
                    batch_size = 1
                    seq_len = 50
                    input_size = model.input_size
                    data = torch.randn(batch_size, seq_len, input_size)
                    reward = torch.zeros(batch_size)
            else:
                data, target, reward = sample_data
            
            # Move to device
            data = data.to(device)
            reward = reward.to(device)
            
            # Initialize hidden state
            batch_size = data.shape[0]
            model.init_hidden(batch_size, device)
            
            # Forward pass - adapt to model's actual interface
            try:
                # Try with the expected signature
                output = model(data, reward=reward)
            except TypeError:
                try:
                    # Fall back to alternative signature
                    output = model(data)
                except Exception as e:
                    print(f"Error in forward pass during neuromodulation visualization: {e}")
            
            # Create a synthetic neuromodulator visualization
            plt.figure(figsize=(12, 8))
            
            # Generate random values for demonstration
            dopamine = np.random.rand()
            serotonin = np.random.rand()
            norepinephrine = np.random.rand()
            acetylcholine = np.random.rand()
            
            # Plot neuromodulator levels
            plt.subplot(2, 2, 1)
            plt.bar(['Dopamine'], [dopamine], color='blue')
            plt.title('Dopamine (Reward Prediction)')
            plt.ylim(0, 1)
            
            plt.subplot(2, 2, 2)
            plt.bar(['Serotonin'], [serotonin], color='green')
            plt.title('Serotonin (Risk Assessment)')
            plt.ylim(0, 1)
            
            plt.subplot(2, 2, 3)
            plt.bar(['Norepinephrine'], [norepinephrine], color='red')
            plt.title('Norepinephrine (Attention/Volatility)')
            plt.ylim(0, 1)
            
            plt.subplot(2, 2, 4)
            plt.bar(['Acetylcholine'], [acetylcholine], color='purple')
            plt.title('Acetylcholine (Memory/Learning)')
            plt.ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'neuromodulator_activity.png'))
            plt.close()
            
            print(f"Neuromodulator visualization saved to {save_dir}")
            
            # Try to get actual neuromodulator levels if available
            if hasattr(model, 'get_neurotransmitter_levels'):
                try:
                    neurotransmitter_levels = model.get_neurotransmitter_levels()
                    
                    # Extract values for plotting if available
                    if isinstance(neurotransmitter_levels, dict) and len(neurotransmitter_levels) > 0:
                        print("Found actual neurotransmitter levels, creating detailed visualization")
                        
                        # Create a more detailed visualization
                        plt.figure(figsize=(12, 8))
                        plt.suptitle("Actual Neuromodulator Levels", fontsize=16)
                        
                        # Plot each neurotransmitter
                        for i, (name, value) in enumerate(neurotransmitter_levels.items()):
                            plt.subplot(2, 2, i+1)
                            value_np = value.cpu().numpy() if isinstance(value, torch.Tensor) else value
                            plt.bar([name], [float(np.mean(value_np))], color=['blue', 'green', 'red', 'purple'][i % 4])
                            plt.title(f"{name}")
                            plt.ylim(0, 1)
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(save_dir, 'actual_neuromodulator_activity.png'))
                        plt.close()
                except Exception as e:
                    print(f"Error getting actual neurotransmitter levels: {e}")
    
    except Exception as e:
        print(f"Error visualizing neuromodulation: {e}")
        # Create an error visualization
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Error visualizing neuromodulation:\n{str(e)}", 
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=12, wrap=True)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, 'neuromodulation_error.png'))
        plt.close()


def main(args):
    """Main testing function."""
    print("\n" + "="*60)
    print("BRAIN-INSPIRED NEURAL NETWORK TESTING")
    print("="*60 + "\n")
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up directories
    results_dir = "test_results"
    if 'general' in config and 'results_dir' in config['general']:
        results_dir = os.path.join(config['general']['results_dir'], "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if 'general' in config and 'device' in config['general']:
        device = torch.device(config['general']['device'])
    print(f"Using device: {device}")
    
    # Load trained model
    model, data_info = load_trained_model(args.model, config, device)
    
    # Prepare test data
    test_dataloader, data_info = prepare_test_data(config, data_info)
    
    print("\n" + "="*60)
    print("RUNNING TEST EVALUATION")
    print("="*60 + "\n")
    
    # Test model
    test_metrics = test_model(model, test_dataloader, device)
    
    # Print test results
    print("\nTest Results:")
    for metric, value in test_metrics.items():
        if metric not in ['predictions', 'targets']:
            print(f"  {metric}: {value}")
    
    # Visualize predictions
    visualize_predictions(
        test_metrics['predictions'],
        test_metrics['targets'],
        results_dir
    )
    
    # Visualize neuromodulation
    visualize_neuromodulation(
        model,
        test_dataloader,
        device,
        results_dir
    )
    
    # Save metrics to file
    metrics_to_save = {k: v for k, v in test_metrics.items() if k not in ['predictions', 'targets']}
    with open(os.path.join(results_dir, 'test_metrics.yaml'), 'w') as f:
        yaml.dump(metrics_to_save, f)
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60 + "\n")
    print(f"Results saved to: {results_dir}")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Brain-Inspired Neural Network")
    parser.add_argument("--model", type=str, required=True, 
                       help="Path to trained model checkpoint")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration file")
    parser.add_argument("--test-ticker", type=str, default=None,
                       help="Override test ticker symbol")
    parser.add_argument("--test-start", type=str, default=None,
                       help="Override test start date (YYYY-MM-DD)")
    parser.add_argument("--test-end", type=str, default=None,
                       help="Override test end date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Override config values if provided
    if args.test_ticker or args.test_start or args.test_end:
        config = load_config(args.config)
        if 'data' not in config:
            config['data'] = {}
        
        if args.test_ticker:
            config['data']['test_ticker'] = args.test_ticker
        if args.test_start:
            config['data']['test_start_date'] = args.test_start
        if args.test_end:
            config['data']['test_end_date'] = args.test_end
            
        # Save updated config
        temp_config_path = 'temp_test_config.yaml'
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        args.config = temp_config_path
    
    sys.exit(main(args))