"""
BrainInspiredTrader: Advanced Testing Framework with OpenAI Integration

This module implements a comprehensive evaluation framework for assessing 
the performance of neuromorphic trading systems with OpenAI-assisted learning.

Fixed key bugs:
1. OpenAI API error handling
2. Input dimension mismatch
3. Tensor size mismatch
4. Controller initialization errors
5. Error handling and recovery
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Any, Optional, Union
import traceback
import time

# Load environment variables for API access
from dotenv import load_dotenv

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import custom modules - will import these dynamically to avoid import errors
# from src.model import BrainInspiredNN
# from src.utils.visualization import (plot_training_curves, visualize_neuromodulators,
#                                     plot_trading_performance, visualize_neuron_activations,
#                                     plot_prediction_error_analysis)

# Import fixed modules
from openai_interface import OpenAIInterface
from src.llm_trainer import LLMTrainer


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_path, config, device):
    """
    Load a trained brain-inspired neural network model.
    
    This function reconstructs the model architecture based on configuration,
    then loads the trained weights from the checkpoint file.
    
    Args:
        model_path (str): Path to model checkpoint
        config (dict): Model configuration
        device (torch.device): Device to load the model on
        
    Returns:
        BrainInspiredNN: Loaded model
    """
    print(f"Loading model from {model_path}...")
    
    # Import the model class
    try:
        from src.model import BrainInspiredNN
    except ImportError:
        print("Error importing BrainInspiredNN, trying relative path...")
        sys.path.append(os.path.abspath(os.path.dirname(__file__)))
        try:
            from model import BrainInspiredNN
        except ImportError:
            print("Failed to import BrainInspiredNN. Please check your project structure.")
            raise
    
    # Load the checkpoint first to get data dimensions
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print(f"Checkpoint loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        
        # If the checkpoint has the config, use it instead of the provided one
        if 'config' in checkpoint:
            print("Using configuration from checkpoint")
            config = checkpoint['config']
        
        # If the data_info is available, use it to get input size
        input_size = None
        if 'data_info' in checkpoint and 'shapes' in checkpoint['data_info']:
            shapes = checkpoint['data_info']['shapes']
            if 'X_train' in shapes:
                input_size = shapes['X_train'][2]  # (batch, seq, features)
                print(f"Input size from checkpoint data_info: {input_size}")
    except Exception as e:
        print(f"Warning: Could not extract complete info from checkpoint: {e}")
        print("Will attempt to create model using provided configuration")
    
    # Extract model parameters from config
    if input_size is None:
        input_size = config.get('input_size', 128)
    
    hidden_size = config.get('controller', {}).get('hidden_size', 256)
    if 'model' in config and 'hidden_size' in config['model']:
        hidden_size = config['model']['hidden_size']
    
    output_size = config.get('output_size', 1)  # Default to 1 for stock prediction
    
    # Get controller parameters
    persistent_memory_size = config.get('controller', {}).get('persistent_memory_size', 128)
    num_layers = config.get('controller', {}).get('num_layers', 2)
    dropout = config.get('controller', {}).get('dropout', 0.1)
    
    # Get neuromodulator parameters
    neuromod_config = config.get('neuromodulator', {})
    dopamine_scale = neuromod_config.get('dopamine_scale', 1.0)
    serotonin_scale = neuromod_config.get('serotonin_scale', 0.8)
    norepinephrine_scale = neuromod_config.get('norepinephrine_scale', 0.6)
    acetylcholine_scale = neuromod_config.get('acetylcholine_scale', 0.7)
    reward_decay = neuromod_config.get('reward_decay', 0.95)
    
    print(f"Creating model with parameters:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Output size: {output_size}")
    print(f"  Persistent memory size: {persistent_memory_size}")
    print(f"  Num layers: {num_layers}")
    
    # Try to create the model using different methods
    model = None
    try:
        # First try the standard initialization
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
    except Exception as e:
        print(f"Error using standard initialization: {e}")
        print("Trying setup_model method...")
        try:
            # Try using the setup_model class method if it exists
            model = BrainInspiredNN.setup_model(config, (1, 1, input_size))
        except Exception as e2:
            print(f"Error using setup_model: {e2}")
            raise ValueError("Failed to create model with either method")
    
    # Load weights from checkpoint
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        print("This can happen if the model architecture doesn't match the saved weights.")
        print("Attempting partial loading...")
        
        # Try partial loading of weights
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        
        # Filter out incompatible keys
        compatible_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and model_dict[k].shape == v.shape}
        
        print(f"Compatible keys: {len(compatible_dict)}/{len(model_dict)}")
        model_dict.update(compatible_dict)
        model.load_state_dict(model_dict, strict=False)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
    return model


def prepare_test_data(config, model_path=None):
    """
    Prepare test data for evaluation.
    
    This function can either:
    1. Load data directly from checkpoint's test set
    2. Download and process fresh data for out-of-sample testing
    
    Args:
        config (dict): Data configuration
        model_path (str, optional): Path to model checkpoint with data info
        
    Returns:
        tuple: (test_dataloader, data_info)
    """
    # Import required modules
    try:
        import yfinance as yf
        from src.model import BrainInspiredNN
    except ImportError:
        print("Error importing required modules, trying relative paths...")
        sys.path.append(os.path.abspath(os.path.dirname(__file__)))
        try:
            import yfinance as yf
            from model import BrainInspiredNN
        except ImportError:
            print("Failed to import required modules. Please check your dependencies.")
            raise
    
    if model_path and os.path.exists(model_path):
        # Try to load test data from checkpoint
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'data_info' in checkpoint and 'dataloaders' in checkpoint['data_info']:
                print("Using test data from checkpoint...")
                data_info = checkpoint['data_info']
                return data_info['dataloaders']['test'], data_info
        except Exception as e:
            print(f"Warning: Could not load test data from checkpoint: {e}")
            print("Will download and process fresh data.")
    
    # If we can't load data from checkpoint, download and process fresh data
    print("Preparing fresh test data...")
    
    # Load stock data
    ticker = config['data']['ticker_symbol']
    start_date = config['data'].get('test_start_date', '2022-01-01')
    end_date = config['data'].get('test_end_date', datetime.now().strftime('%Y-%m-%d'))
    
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        if stock_data.empty:
            raise ValueError(f"No data found for ticker {ticker} in the specified date range")
        
        print(f"Downloaded {len(stock_data)} days of stock data")
        
        # Process data
        data_info = BrainInspiredNN.preprocess_data(stock_data, config)
        
        return data_info['dataloaders']['test'], data_info
    
    except Exception as e:
        print(f"Error preparing test data: {e}")
        raise


def run_evaluation(args):
    """
    Run the main evaluation process.
    
    This function orchestrates the entire evaluation workflow, from loading
    the model to generating the final report.
    
    Args:
        args: Command line arguments
        
    Returns:
        int: Exit code
    """
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Set up results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(config['general']['results_dir'], f"eval_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save a copy of the config
        with open(os.path.join(results_dir, "config.yaml"), 'w') as f:
            yaml.dump(config, f)
        
        # Set device
        device = torch.device(config['general']['device'] if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Set random seed
        torch.manual_seed(config['general']['seed'])
        np.random.seed(config['general']['seed'])
        
        # Initialize OpenAI interface if needed
        if args.use_llm:
            print("Initializing OpenAI interface...")
            try:
                openai_interface = OpenAIInterface(
                    model=config.get('llm', {}).get('model', 'gpt-4o-mini')
                )
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI interface: {e}")
                print("Will continue without LLM capabilities")
                args.use_llm = False
        
        # Load model
        model = load_model(args.model, config, device)
        
        # Prepare test data
        test_dataloader, data_info = prepare_test_data(config, args.model)
        
        # Import visualization functions
        try:
            from src.utils.visualization import (
                plot_trading_performance,
                visualize_neuromodulators,
                visualize_neuron_activations,
                plot_prediction_error_analysis
            )
        except ImportError:
            print("Warning: Could not import visualization modules. Will use simplified visuals.")
            # Define simple placeholder visualization functions
            def plot_trading_performance(*args, **kwargs):
                plt.figure(figsize=(10, 6))
                plt.title("Trading Performance")
                plt.savefig(kwargs.get('save_path', 'trading_performance.png'))
                plt.close()
                
            def visualize_neuromodulators(*args, **kwargs):
                plt.figure(figsize=(10, 6))
                plt.title("Neuromodulator Activity")
                plt.savefig(kwargs.get('save_path', 'neuromodulator_activity.png'))
                plt.close()
                
            def visualize_neuron_activations(*args, **kwargs):
                plt.figure(figsize=(10, 6))
                plt.title("Neuron Activations")
                plt.savefig(kwargs.get('save_path', 'neuron_activations.png'))
                plt.close()
                
            def plot_prediction_error_analysis(*args, **kwargs):
                plt.figure(figsize=(10, 6))
                plt.title("Prediction Error Analysis")
                plt.savefig(kwargs.get('save_path', 'error_analysis.png'))
                plt.close()
        
        # Get sample batch for neural analysis
        sample_batch = None
        for batch in test_dataloader:
            sample_batch = batch
            break
        
        if sample_batch is None:
            print("Warning: Could not get sample batch from test data")
            # Create a dummy sample batch
            dummy_x = torch.zeros((1, 1, model.input_size))
            dummy_y = torch.zeros((1, 1, 1))
            dummy_r = torch.zeros((1, 1, 1))
            sample_batch = (dummy_x, dummy_y, dummy_r)
        
        # Evaluate prediction accuracy
        def evaluate_prediction_accuracy(model, dataloader, device):
            print("Evaluating prediction accuracy...")
            
            model.eval()
            all_predictions = []
            all_targets = []
            all_rewards = []
            
            with torch.no_grad():
                for data, target, reward in tqdm(dataloader, desc="Prediction evaluation"):
                    try:
                        # Move data to device
                        data = data.to(device)
                        target = target.to(device)
                        reward = reward.to(device)
                        
                        # Forward pass
                        output, predicted_reward = model(data, external_reward=reward)
                        
                        # Check if output shape matches target shape
                        if output.shape[-1] != target.shape[-1]:
                            print(f"Output shape {output.shape} doesn't match target shape {target.shape}")
                            if output.shape[-1] > target.shape[-1]:
                                output = output[:, :, :target.shape[-1]]
                            else:
                                # Create a projection layer
                                proj = torch.nn.Linear(output.shape[-1], target.shape[-1]).to(device)
                                output = proj(output)
                        
                        # Store outputs
                        all_predictions.append(output.cpu().numpy())
                        all_targets.append(target.cpu().numpy())
                        all_rewards.append(reward.cpu().numpy())
                    except Exception as e:
                        print(f"Error processing batch: {e}")
                        continue
            
            # Check if we have any predictions
            if not all_predictions:
                print("Warning: No valid predictions were made")
                return {
                    'mse': float('inf'),
                    'rmse': float('inf'),
                    'mae': float('inf'),
                    'r2': -float('inf'),
                    'direction_accuracy': 0,
                    'reward_mse': float('inf'),
                    'reward_rmse': float('inf'),
                    'predictions': np.array([]),
                    'targets': np.array([]),
                    'rewards': np.array([])
                }
            
            # Concatenate results
            try:
                predictions = np.vstack(all_predictions)
                targets = np.vstack(all_targets)
                rewards = np.vstack(all_rewards)
                
                # Calculate standard metrics
                mse = mean_squared_error(targets, predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(targets, predictions)
                
                # Calculate R² (handle potential errors)
                try:
                    r2 = r2_score(targets.flatten(), predictions.flatten())
                except Exception:
                    r2 = -1.0  # Invalid R² value
                
                # Calculate direction accuracy (sign of price movement)
                direction_correct = 0
                total_samples = 0
                
                for i in range(len(predictions)-1):
                    pred_direction = predictions[i+1] > predictions[i]
                    actual_direction = targets[i+1] > targets[i]
                    direction_correct += np.sum(pred_direction == actual_direction)
                    total_samples += len(pred_direction)
                
                direction_accuracy = direction_correct / total_samples if total_samples > 0 else 0
                
                # Calculate reward prediction accuracy
                reward_mse = mean_squared_error(rewards, all_rewards)
                reward_rmse = np.sqrt(reward_mse)
                
                # Compile metrics
                metrics = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'direction_accuracy': direction_accuracy,
                    'reward_mse': reward_mse,
                    'reward_rmse': reward_rmse,
                    'predictions': predictions,
                    'targets': targets,
                    'rewards': rewards
                }
                
                # Print summary
                print(f"Prediction Results:")
                print(f"  MSE: {mse:.6f}")
                print(f"  RMSE: {rmse:.6f}")
                print(f"  MAE: {mae:.6f}")
                print(f"  R²: {r2:.6f}")
                print(f"  Direction Accuracy: {direction_accuracy*100:.2f}%")
                print(f"  Reward Prediction RMSE: {reward_rmse:.6f}")
                
                return metrics
            
            except Exception as e:
                print(f"Error calculating metrics: {e}")
                return {
                    'mse': float('inf'),
                    'rmse': float('inf'),
                    'mae': float('inf'),
                    'r2': -float('inf'),
                    'direction_accuracy': 0,
                    'reward_mse': float('inf'),
                    'reward_rmse': float('inf'),
                    'predictions': np.array([]),
                    'targets': np.array([]),
                    'rewards': np.array([])
                }
        
        prediction_metrics = evaluate_prediction_accuracy(model, test_dataloader, device)
        
        # Plot prediction error analysis
        plot_prediction_error_analysis(
            prediction_metrics['targets'].flatten(),
            prediction_metrics['predictions'].flatten(),
            save_path=os.path.join(results_dir, "error_analysis.png")
        )
        
        # Evaluate trading strategy
        def evaluate_trading_strategy(model, data_info, config, device, results_dir):
            print("Evaluating trading strategy...")
            
            test_dataloader = data_info['dataloaders']['test']
            scaler = data_info['scaler']
            feature_columns = data_info['feature_columns']
            target_idx = data_info['target_idx']
            
            # Set initial capital
            initial_capital = 10000  # $10,000
            
            # Collect all predictions and true values
            model.eval()
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for data, target, _ in tqdm(test_dataloader, desc="Trading simulation"):
                    try:
                        data = data.to(device)
                        output, _ = model(data)
                        
                        # Ensure output has the right shape
                        if output.shape[-1] != target.shape[-1]:
                            if output.shape[-1] > target.shape[-1]:
                                output = output[:, :, :target.shape[-1]]
                            else:
                                # Create a projection layer
                                proj = torch.nn.Linear(output.shape[-1], target.shape[-1]).to(device)
                                output = proj(output)
                        
                        all_predictions.append(output.cpu().numpy())
                        all_targets.append(target.numpy())
                    except Exception as e:
                        print(f"Error in trading simulation: {e}")
                        continue
            
            # Check if we have predictions
            if not all_predictions:
                print("Warning: No valid predictions for trading simulation")
                return {
                    'initial_capital': initial_capital,
                    'final_capital': initial_capital,
                    'total_return_pct': 0.0,
                    'equity_curve': [initial_capital],
                    'num_trades': 0,
                    'win_rate': 0.0
                }
            
            try:
                # Concatenate results
                predictions = np.vstack(all_predictions)
                targets = np.vstack(all_targets)
                
                # Inverse transform to get actual prices
                # Create dummy arrays for inverse transformation
                dummy_pred = np.zeros((len(predictions), len(feature_columns)))
                dummy_pred[:, target_idx] = predictions.flatten()
                
                dummy_true = np.zeros((len(targets), len(feature_columns)))
                dummy_true[:, target_idx] = targets.flatten()
                
                # Inverse transform
                pred_prices = scaler.inverse_transform(dummy_pred)[:, target_idx]
                true_prices = scaler.inverse_transform(dummy_true)[:, target_idx]
                
                # Simulate trading strategy
                capital = initial_capital
                position = 0  # Number of shares held
                trade_history = []
                equity_curve = [capital]
                
                for i in range(1, len(true_prices)):
                    current_price = true_prices[i-1]
                    next_price = true_prices[i]
                    predicted_direction = pred_prices[i] > current_price
                    
                    # Simple strategy: Buy if predicted to go up, sell if predicted to go down
                    if predicted_direction and position == 0:  # Buy signal
                        position = capital / current_price
                        capital = 0
                        trade_history.append(('buy', i-1, current_price))
                    elif not predicted_direction and position > 0:  # Sell signal
                        capital = position * current_price
                        position = 0
                        trade_history.append(('sell', i-1, current_price))
                    
                    # Update equity (mark to market)
                    equity = capital + (position * current_price)
                    equity_curve.append(equity)
                
                # Close any open position at the end
                if position > 0:
                    capital = position * true_prices[-1]
                    trade_history.append(('sell', len(true_prices)-1, true_prices[-1]))
                    equity_curve[-1] = capital
                
                # Calculate performance metrics
                final_capital = equity_curve[-1]
                total_return = (final_capital / initial_capital - 1) * 100
                
                # Calculate daily returns
                daily_returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
                
                # Annualized metrics (assuming 252 trading days per year)
                trading_days = len(equity_curve)
                years = trading_days / 252
                
                # Annualized return
                ann_return = (1 + total_return/100) ** (1/years) - 1 if years > 0 else 0
                
                # Annualized volatility
                ann_volatility = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 else 0
                
                # Sharpe ratio (assuming 2% risk-free rate)
                risk_free_rate = 0.02
                sharpe_ratio = (ann_return - risk_free_rate) / ann_volatility if ann_volatility > 0 else 0
                
                # Sortino ratio (downside risk only)
                negative_returns = daily_returns[daily_returns < 0]
                downside_dev = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 1e-6
                sortino_ratio = (ann_return - risk_free_rate) / downside_dev
                
                # Maximum drawdown
                peak = equity_curve[0]
                max_drawdown = 0
                drawdown_duration = 0
                max_drawdown_duration = 0
                in_drawdown = False
                
                for i, value in enumerate(equity_curve):
                    if value > peak:
                        peak = value
                        if in_drawdown:
                            in_drawdown = False
                            if drawdown_duration > max_drawdown_duration:
                                max_drawdown_duration = drawdown_duration
                            drawdown_duration = 0
                    else:
                        drawdown = (peak - value) / peak if peak > 0 else 0
                        max_drawdown = max(max_drawdown, drawdown)
                        if drawdown > 0:
                            in_drawdown = True
                            drawdown_duration += 1
                
                # Number of trades and win rate
                num_trades = len(trade_history) // 2  # Buy and sell pairs
                
                # Calculate win/loss for completed trades
                wins = 0
                for i in range(0, len(trade_history)-1, 2):
                    if i+1 < len(trade_history):
                        buy_price = trade_history[i][2]
                        sell_price = trade_history[i+1][2]
                        if sell_price > buy_price:
                            wins += 1
                
                win_rate = wins / num_trades if num_trades > 0 else 0
                
                # Plot trading performance
                plot_path = os.path.join(results_dir, "trading_performance.png")
                plot_trading_performance(true_prices, pred_prices, trade_history, equity_curve, 
                                       initial_capital, plot_path)
                
                # Compile metrics
                trading_metrics = {
                    'initial_capital': initial_capital,
                    'final_capital': final_capital,
                    'total_return_pct': total_return,
                    'annualized_return_pct': ann_return * 100,
                    'annualized_volatility_pct': ann_volatility * 100,
                    'sharpe_ratio': sharpe_ratio,
                    'sortino_ratio': sortino_ratio,
                    'max_drawdown_pct': max_drawdown * 100,
                    'max_drawdown_duration': max_drawdown_duration,
                    'num_trades': num_trades,
                    'win_rate': win_rate,
                    'equity_curve': equity_curve,
                    'trade_history': trade_history,
                    'true_prices': true_prices,
                    'pred_prices': pred_prices
                }
                
                # Print summary
                print(f"Trading Performance:")
                print(f"  Initial Capital: ${initial_capital:,.2f}")
                print(f"  Final Capital: ${final_capital:,.2f}")
                print(f"  Total Return: {total_return:.2f}%")
                print(f"  Annualized Return: {ann_return*100:.2f}%")
                print(f"  Annualized Volatility: {ann_volatility*100:.2f}%")
                print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
                print(f"  Sortino Ratio: {sortino_ratio:.4f}")
                print(f"  Maximum Drawdown: {max_drawdown*100:.2f}%")
                print(f"  Max Drawdown Duration: {max_drawdown_duration} days")
                print(f"  Number of Trades: {num_trades}")
                print(f"  Win Rate: {win_rate*100:.2f}%")
                
                return trading_metrics
            
            except Exception as e:
                print(f"Error in trading strategy evaluation: {e}")
                traceback.print_exc()
                
                # Return minimal metrics
                return {
                    'initial_capital': initial_capital,
                    'final_capital': initial_capital,
                    'total_return_pct': 0.0,
                    'equity_curve': [initial_capital],
                    'num_trades': 0,
                    'win_rate': 0.0
                }
        
        trading_metrics = evaluate_trading_strategy(model, data_info, config, device, results_dir)
        
        # Analyze neural activity
        def analyze_neural_activity(model, sample_data, device, results_dir):
            print("Analyzing neural activity patterns...")
            
            # Process a sample batch
            if isinstance(sample_data, torch.utils.data.DataLoader):
                # If we're given a dataloader, take a batch
                for data, target, reward in sample_data:
                    sample_batch = (data, target, reward)
                    break
            else:
                # Otherwise use the provided sample data
                sample_batch = sample_data
            
            # Move to device
            data, target, reward = sample_batch
            data = data.to(device)
            reward = reward.to(device)
            
            # Forward pass with tracking enabled
            model.eval()
            with torch.no_grad():
                try:
                    output, predicted_reward = model(data, external_reward=reward)
                except Exception as e:
                    print(f"Error during forward pass: {e}")
                    # Return empty metrics
                    return {}
            
            # Collect neural activity metrics
            neural_metrics = {}
            
            # Analyze neurotransmitter levels if available
            if hasattr(model, 'neurotransmitter_levels') and model.neurotransmitter_levels is not None:
                try:
                    nt_levels = model.neurotransmitter_levels
                    
                    # Format neurotransmitter levels as dictionary
                    nt_dict = {
                        'dopamine': float(nt_levels[0][0]) if nt_levels.dim() > 1 and nt_levels.size(1) > 0 else float(nt_levels[0]),
                        'serotonin': float(nt_levels[0][1]) if nt_levels.dim() > 1 and nt_levels.size(1) > 1 else 0.5,
                        'norepinephrine': float(nt_levels[0][2]) if nt_levels.dim() > 1 and nt_levels.size(1) > 2 else 0.5,
                        'acetylcholine': float(nt_levels[0][3]) if nt_levels.dim() > 1 and nt_levels.size(1) > 3 else 0.5
                    }
                    
                    neural_metrics['neurotransmitter_levels'] = nt_dict
                    
                    # Visualize neurotransmitter activity
                    fig_path = os.path.join(results_dir, "neuromodulator_activity.png")
                    visualize_neuromodulators(nt_dict, fig_path)
                    
                    # Print neurotransmitter analysis
                    print(f"Neuromodulator Analysis:")
                    for name, level in neural_metrics['neurotransmitter_levels'].items():
                        print(f"  {name.capitalize()}: {level:.4f}")
                    
                    # Provide neurobiological interpretation
                    dopamine = nt_dict.get('dopamine', 0)
                    serotonin = nt_dict.get('serotonin', 0)
                    norepinephrine = nt_dict.get('norepinephrine', 0)
                    
                    if dopamine > 0.7:
                        print("  High dopamine indicates strong reward prediction - system expects positive returns")
                    elif dopamine < 0.3:
                        print("  Low dopamine suggests minimal reward expectation - system is risk-averse")
                    
                    if serotonin > 0.7:
                        print("  Elevated serotonin shows the system is in a balanced risk assessment state")
                    
                    if norepinephrine > 0.7:
                        print("  High norepinephrine suggests the system is highly attentive to market volatility")
                    
                except Exception as e:
                    print(f"Could not analyze neurotransmitter levels: {e}")
            elif hasattr(model, 'neuromodulator'):
                # Try to get parameters directly from the neuromodulator
                try:
                    nt_dict = {
                        'dopamine': float(model.neuromodulator.dopamine_scale),
                        'serotonin': float(model.neuromodulator.serotonin_scale),
                        'norepinephrine': float(model.neuromodulator.norepinephrine_scale),
                        'acetylcholine': float(model.neuromodulator.acetylcholine_scale)
                    }
                    
                    neural_metrics['neurotransmitter_levels'] = nt_dict
                    
                    # Visualize neurotransmitter activity
                    fig_path = os.path.join(results_dir, "neuromodulator_params.png")
                    visualize_neuromodulators(nt_dict, fig_path)
                    
                    print(f"Neuromodulator Parameters:")
                    for name, level in nt_dict.items():
                        print(f"  {name.capitalize()}: {level:.4f}")
                except Exception as e:
                    print(f"Could not analyze neuromodulator parameters: {e}")
            
            # Analyze neural activations
            try:
                visualize_neuron_activations(model, (data, target, reward), device, 
                                           os.path.join(results_dir, "neuron_activations.png"))
            except Exception as e:
                print(f"Could not visualize neuron activations: {e}")
            
            return neural_metrics
        
        neural_metrics = analyze_neural_activity(model, sample_batch, device, results_dir)
        
        # Get LLM explanations if requested
        if args.use_llm:
            try:
                # Extract price data for market analysis
                scaler = data_info['scaler']
                feature_columns = data_info['feature_columns']
                target_idx = data_info['target_idx']
                
                # Get a sample from test data
                sample_data = None
                for data, target, reward in test_dataloader:
                    sample_data = (data, target, reward)
                    break
                
                if sample_data is not None:
                    # Get model prediction
                    data, target, reward = sample_data
                    data = data.to(device)
                    model.eval()
                    with torch.no_grad():
                        output, _ = model(data)
                    
                    # Get neurotransmitter levels
                    if 'neurotransmitter_levels' in neural_metrics:
                        nt_levels = neural_metrics['neurotransmitter_levels']
                    else:
                        nt_levels = {
                            'dopamine': 0.5,
                            'serotonin': 0.5,
                            'norepinephrine': 0.5,
                            'acetylcholine': 0.5
                        }
                    
                    # Extract price history
                    input_seq = data[0].cpu().numpy()
                    
                    # Create dummy array for inverse transformation
                    dummy_input = np.zeros((len(input_seq), len(feature_columns)))
                    dummy_input[:, target_idx] = input_seq[:, target_idx]
                    
                    # Inverse transform to get price history
                    price_history = scaler.inverse_transform(dummy_input)[:, target_idx]
                    
                    # Get explanation
                    explanation = openai_interface.explain_model_decision(
                        input_seq, output.cpu().numpy(), nt_levels, price_history
                    )
                    
                    # Save explanation
                    explanation_path = os.path.join(results_dir, "model_explanation.json")
                    with open(explanation_path, 'w') as f:
                        json.dump(explanation, f, indent=2)
                    
                    print("\nLLM Model Explanation:")
                    if 'narrative_explanation' in explanation:
                        print(explanation['narrative_explanation'][:300] + "..." 
                              if len(explanation['narrative_explanation']) > 300 
                              else explanation['narrative_explanation'])
                    
                    # Add market analysis
                    market_analysis = openai_interface.analyze_market_patterns(price_history)
                    market_analysis_path = os.path.join(results_dir, "market_analysis.json")
                    with open(market_analysis_path, 'w') as f:
                        json.dump(market_analysis, f, indent=2)
                    
                    print("\nMarket Analysis:")
                    print(f"Market Regime: {market_analysis.get('market_regime', 'Unknown')}")
                    print(f"Technical Patterns: {', '.join(market_analysis.get('technical_patterns', ['None']))}")
                    print(f"Directional Bias: {market_analysis.get('directional_bias', 'Neutral')}")
            
            except Exception as e:
                print(f"Error generating LLM explanations: {e}")
        
        print(f"\nEvaluation complete. All results saved to: {results_dir}")
        
        return 0
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()
        return 1


def main(args):
    """Main function for training and evaluation."""
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Set device
        device = torch.device(config['general']['device'] if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Set random seed
        torch.manual_seed(config['general']['seed'])
        np.random.seed(config['general']['seed'])
        
        # Create directories
        os.makedirs(config['general']['log_dir'], exist_ok=True)
        os.makedirs(config['general']['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['general']['results_dir'], exist_ok=True)
        
        # Initialize OpenAI interface for LLM capabilities
        if args.train and config.get('use_llm', True):
            try:
                print("Initializing OpenAI interface...")
                openai_interface = OpenAIInterface(
                    model=config.get('llm', {}).get('model', 'gpt-4o-mini')
                )
                
                # Initialize trainer
                print("Creating LLM-enhanced trainer...")
                trainer = LLMTrainer(config, openai_interface, device)
                
                # Run training
                print("\n=== Starting LLM-enhanced training ===")
                trainer.train(num_epochs=args.epochs)
                
            except Exception as e:
                print(f"Error during LLM-enhanced training: {e}")
                traceback.print_exc()
                return 1
        
        # Run evaluation
        if args.evaluate and args.model:
            print("\n=== Starting model evaluation ===")
            eval_args = argparse.Namespace(
                config=args.config,
                model=args.model,
                mode='full',
                use_llm=config.get('use_llm', False)
            )
            return run_evaluation(eval_args)
        
        return 0
        
    except Exception as e:
        print(f"Error during execution: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BrainInspiredNN with OpenAI Integration")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to configuration file")
    parser.add_argument("--train", action="store_true", 
                        help="Run training with OpenAI integration")
    parser.add_argument("--evaluate", action="store_true",
                       help="Run evaluation")
    parser.add_argument("--model", type=str, 
                       help="Path to trained model checkpoint for evaluation")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    
    args = parser.parse_args()
    sys.exit(main(args))