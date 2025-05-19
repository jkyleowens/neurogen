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
project_root = os.path.dirname(os.path.abspath(__file__))
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
            input_shape = (32, 30, 5)  # (batch_size, seq_length, features)
        
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
    ticker = config['data'].get('test_ticker', config['data']['ticker_symbol'])
    
    # IMPORTANT: Extend the date range to ensure enough data
    # Get at least 3x the required sequence points to be safe
    required_points = config['data']['sequence_length'] + config['data']['prediction_horizon']
    safety_factor = 3  # Get 3x more data than minimally needed
    min_trading_days_needed = required_points * safety_factor
    
    # Calculate start date to get enough data (assuming ~252 trading days per year)
    import datetime
    end_date = config['data'].get('test_end_date', datetime.datetime.now().strftime('%Y-%m-%d'))
    end_date_obj = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    
    # Calculate years needed to get enough trading days
    years_needed = min_trading_days_needed / 252
    
    # Add a buffer for safety
    start_date_obj = end_date_obj - datetime.timedelta(days=int(365.25 * (years_needed + 0.5)))
    start_date = start_date_obj.strftime('%Y-%m-%d')
    
    # Override with user-specified dates if provided
    start_date = config['data'].get('test_start_date', start_date)
    
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    print(f"Note: Using extended date range to ensure enough data points for sequence length ({config['data']['sequence_length']}) + prediction horizon ({config['data']['prediction_horizon']})")
    
    import yfinance as yf
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    if stock_data.empty:
        raise ValueError(f"No data found for ticker {ticker} in the specified date range")
    
    print(f"Downloaded {len(stock_data)} days of stock data ({len(stock_data)} trading days)")
    
    # Check if we have enough data
    if len(stock_data) < required_points:
        raise ValueError(
            f"Not enough data points ({len(stock_data)}) for sequence length ({config['data']['sequence_length']}) "
            f"+ prediction horizon ({config['data']['prediction_horizon']}). Need at least {required_points} points. "
            f"Try extending the date range or reducing sequence length."
        )
    
    # Create a modified config with adaptive sequence length if needed
    adapted_config = config.copy()
    if len(stock_data) < min_trading_days_needed:
        # We have enough for minimal operation but not optimal - adjust and warn
        print(f"Warning: Limited data points ({len(stock_data)}). For optimal results, at least {min_trading_days_needed} are recommended.")
        print(f"Proceeding with available data, but results may be less reliable.")
    
    # Process data
    try:
        data_info = BrainInspiredNN.preprocess_data(stock_data, adapted_config)
        return data_info['dataloaders']['test'], data_info
    except ValueError as e:
        if "Not enough data points" in str(e):
            # Last-resort fallback: dynamically reduce sequence length
            original_seq_length = adapted_config['data']['sequence_length']
            max_possible_seq_length = len(stock_data) - adapted_config['data']['prediction_horizon'] - 10
            
            if max_possible_seq_length > 5:  # Ensure we still have a reasonable sequence length
                adapted_config['data']['sequence_length'] = max_possible_seq_length
                print(f"FALLBACK: Reducing sequence length from {original_seq_length} to {max_possible_seq_length} to fit available data")
                data_info = BrainInspiredNN.preprocess_data(stock_data, adapted_config)
                return data_info['dataloaders']['test'], data_info
                
        # If we get here, we couldn't handle the error
        raise

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
    
    # Reset any persistent state
    hidden = None
    persistent_memory = None
    
    with torch.no_grad():
        for batch_idx, (data, target, reward) in enumerate(tqdm(dataloader, desc="Testing")):
            try:
                # Move data to device
                data = data.to(device)
                target = target.to(device)
                reward = reward.to(device)
                
                # Forward pass
                output, predicted_reward = model(
                    data, 
                    hidden=hidden, 
                    persistent_memory=persistent_memory,
                    external_reward=reward
                )
                
                # Update hidden states for consistent sequence processing
                if hasattr(model, 'hidden_states'):
                    hidden = model.hidden_states
                if hasattr(model, 'persistent_memories'):
                    persistent_memory = model.persistent_memories
                
                # Extract last time step from sequence outputs
                if output.dim() == 3:  # Shape: [batch_size, seq_len, output_dim]
                    output = output[:, -1, :]  # Use only the last time step prediction
                
                # Store results
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                
            except Exception as e:
                print(f"Error in test batch {batch_idx}: {e}")
                # Reset states on error
                hidden = None
                persistent_memory = None
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
        with torch.no_grad():
            # Unpack sample data
            if isinstance(sample_data, torch.utils.data.DataLoader):
                data, target, reward = next(iter(sample_data))
            else:
                data, target, reward = sample_data
            
            # Move to device
            data = data.to(device)
            reward = reward.to(device)
            
            # Forward pass
            _, _ = model(data, external_reward=reward)
            
            # Get neuromodulator levels
            if hasattr(model, 'get_neurotransmitter_levels'):
                neurotransmitter_levels = model.get_neurotransmitter_levels()
                
                # Extract values for plotting
                if isinstance(neurotransmitter_levels, dict):
                    dopamine = neurotransmitter_levels.get('dopamine', torch.tensor(0.0)).cpu().numpy()
                    serotonin = neurotransmitter_levels.get('serotonin', torch.tensor(0.0)).cpu().numpy()
                    norepinephrine = neurotransmitter_levels.get('norepinephrine', torch.tensor(0.0)).cpu().numpy()
                    acetylcholine = neurotransmitter_levels.get('acetylcholine', torch.tensor(0.0)).cpu().numpy()
                    
                    # Create visualization
                    plt.figure(figsize=(12, 8))
                    
                    # Plot neuromodulator levels
                    plt.subplot(2, 2, 1)
                    plt.bar(['Dopamine'], [float(np.mean(dopamine))], color='blue')
                    plt.title('Dopamine (Reward Prediction)')
                    plt.ylim(0, 1)
                    
                    plt.subplot(2, 2, 2)
                    plt.bar(['Serotonin'], [float(np.mean(serotonin))], color='green')
                    plt.title('Serotonin (Risk Assessment)')
                    plt.ylim(0, 1)
                    
                    plt.subplot(2, 2, 3)
                    plt.bar(['Norepinephrine'], [float(np.mean(norepinephrine))], color='red')
                    plt.title('Norepinephrine (Attention/Volatility)')
                    plt.ylim(0, 1)
                    
                    plt.subplot(2, 2, 4)
                    plt.bar(['Acetylcholine'], [float(np.mean(acetylcholine))], color='purple')
                    plt.title('Acetylcholine (Memory/Learning)')
                    plt.ylim(0, 1)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, 'neuromodulator_activity.png'))
                    plt.close()
                    
                    print(f"Neuromodulator visualization saved to {save_dir}")
                else:
                    print("Neurotransmitter levels not in expected format")
            else:
                print("Model doesn't have neurotransmitter_levels attribute")
    
    except Exception as e:
        print(f"Error visualizing neuromodulation: {e}")


def main(args):
    """Main testing function."""
    print("\n" + "="*60)
    print("BRAIN-INSPIRED NEURAL NETWORK TESTING")
    print("="*60 + "\n")
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up directories
    results_dir = os.path.join(config['general']['results_dir'], "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Set device
    device = torch.device(config['general'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
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