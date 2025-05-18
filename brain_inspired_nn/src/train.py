"""
Training Script for Brain-Inspired Neural Network

This script handles the training process for the brain-inspired neural network,
including data loading, training loop, and LLM integration for validation.
"""

import os
import sys
import yaml
import argparse
import numpy as np
from tqdm import tqdm

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Critical fix: Use proper module paths
from src.model import BrainInspiredNN

class LLMInterface:
    def __init__(self, config):
        self.config = config
        # Initialize LLM connection
        
    def generate_prompt(self, mode):
        """Generate appropriate prompt based on mode (train/val)."""
        # Return prompt string
        
    def get_response(self, prompt):
        """Get response from LLM."""
        # Return LLM response
        
    def process_input(self, llm_response, input_size):
        """Convert LLM response to input tensor."""
        # Return torch.Tensor of size (input_size,)
        
    def process_target(self, llm_response, output_size):
        """Convert LLM response to target tensor."""
        # Return torch.Tensor of size (output_size,)
        
    def calculate_reward(self, llm_response):
        """Calculate reward based on LLM response."""
        # Return float or torch.Tensor of size (1,)


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
    scheduler_name = config['training']['scheduler'].lower()
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

def evaluate_trading_strategy(model, test_dataloader, data_info, device):
    """Evaluate model with financial metrics including trading strategy returns."""
    model.eval()
    predictions = []
    actuals = []
    rewards = []
    
    with torch.no_grad():
        for data, target, reward in test_dataloader:
            # Forward pass
            data = data.to(device)
            output, _ = model(data)
            
            # Store results
            predictions.append(output.cpu().numpy())
            actuals.append(target.numpy())
            rewards.append(reward.numpy())
    
    # Concatenate results
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    # Calculate standard metrics
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    
    # Direction accuracy (crucial for trading)
    pred_direction = np.diff(predictions.flatten(), prepend=predictions[0,0])
    actual_direction = np.diff(actuals.flatten(), prepend=actuals[0,0])
    direction_accuracy = np.mean((pred_direction > 0) == (actual_direction > 0))
    
    # Simulate trading strategy (buy when prediction is up, sell when down)
    initial_capital = 10000
    position = 0
    capital = initial_capital
    trades = []
    
    # Implementation of trading strategy simulation
    # [Trading strategy simulation code]
    
    return {
        'mse': mse,
        'mae': mae,
        'direction_accuracy': direction_accuracy,
        'final_capital': capital,
        'return': (capital - initial_capital) / initial_capital * 100,
        'trades': trades
    }

def visualize_neuromodulation(model, sample_data, device):
    """Visualize neuromodulator activity during financial prediction."""
    # Process a single batch of data
    sample_data = sample_data.to(device)
    model.eval()
    
    with torch.no_grad():
        # Run the model
        _, _ = model(sample_data)
        
        # Get neuromodulator levels
        neurotransmitter_levels = model.get_neurotransmitter_levels()
        
        # Extract values
        dopamine = neurotransmitter_levels['dopamine'].cpu().numpy()
        serotonin = neurotransmitter_levels['serotonin'].cpu().numpy()
        norepinephrine = neurotransmitter_levels['norepinephrine'].cpu().numpy()
        acetylcholine = neurotransmitter_levels['acetylcholine'].cpu().numpy()
        
        # Plot the results
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(dopamine.flatten())
        plt.title('Dopamine Activity (Reward Prediction)')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(serotonin.flatten())
        plt.title('Serotonin Activity (Risk Assessment)')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(norepinephrine.flatten())
        plt.title('Norepinephrine Activity (Market Volatility Attention)')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(acetylcholine.flatten())
        plt.title('Acetylcholine Activity (Pattern Memory)')
        plt.grid(True)
        
        plt.tight_layout()
        return plt


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train the model for one epoch with proper reward handling."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target, reward) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        # Move data to device
        data = data.to(device)
        target = target.to(device)
        reward = reward.to(device)  # Critical fix: Always provide reward
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass
        output, predicted_reward = model(data, external_reward=reward)
        
        # Calculate prediction loss
        task_loss = criterion(output, target)
        
        # Calculate reward prediction loss
        # The neuromodulator system uses this to adjust its internal state
        reward_loss = criterion(predicted_reward, reward)
        
        # Combined loss - critical for neuromodulation to work correctly
        # Fix: Adjust weight of reward_loss to prevent it from dominating
        loss = task_loss + 0.5 * reward_loss
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
    
    # Return average loss
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """Validate the model on the validation set."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for data, target, reward in tqdm(dataloader, desc="Validation"):
            # Move data to device
            data = data.to(device)
            target = target.to(device)
            reward = reward.to(device) if reward is not None else None
            
            # Forward pass
            output, predicted_reward = model(data, external_reward=reward)
            
            # Calculate loss
            task_loss = nn.functional.mse_loss(output, target)
            
            # Add reward prediction loss if external reward is provided
            if reward is not None:
                reward_loss = nn.functional.mse_loss(predicted_reward, reward)
                loss = task_loss + reward_loss
            else:
                loss = task_loss
            
            # Accumulate loss
            total_loss += loss.item()
    
    # Return average loss
    return total_loss / len(dataloader)


def llm_validation(model, llm_interface, validation_prompts, device):
    """Validate the model using LLM integration."""
    # Use the enhanced LLM interface's validate_with_llm method
    return llm_interface.validate_with_llm(model, validation_prompts, device)


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


def create_datasets(config):
    """
    Create datasets for stock prediction instead of LLM-based datasets.
    This function replaces the original LLM-based implementation.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (train_dataset, val_dataset) PyTorch datasets
    """
    # Load stock data
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
    
    # Extract the train and validation datasets from the data_info
    train_dataloader = data_info['dataloaders']['train']
    val_dataloader = data_info['dataloaders']['val']
    
    # Get the underlying datasets from the dataloaders
    train_dataset = train_dataloader.dataset
    val_dataset = val_dataloader.dataset
    
    return train_dataset, val_dataset

def main(args):
    """Main training function."""
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
    
    # Load and preprocess data
    try:
        # Create datasets and dataloaders
        train_dataset, val_dataset = create_datasets(config)
        
        batch_size = config['training'].get('batch_size', 32)
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config['training'].get('num_workers', 2) if 'num_workers' in config['training'] else 2
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config['training'].get('num_workers', 2) if 'num_workers' in config['training'] else 2
        )
    except KeyError as e:
        print(f"Configuration error: Missing key {e} in config. Check your config file.")
        return
    
    # Set up model using the shape from actual data
    input_shape = train_dataset.tensors[0].shape  # Get shape from the dataset
    model = setup_model(config, input_shape)
    model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Set up optimizer and scheduler
    optimizer, scheduler = setup_optimizer(model, config)
    
    # Set up loss function
    criterion = nn.MSELoss()
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        # Train epoch
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device, epoch)
        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.6f}")
        
        # Validate
        val_loss, val_mse, val_mae = validate(model, val_dataloader, criterion, device)
        print(f"Epoch {epoch}/{num_epochs}, Validation Loss: {val_loss:.6f}, MSE: {val_mse:.6f}, MAE: {val_mae:.6f}")
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(config['general']['checkpoint_dir'], "best_model.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_mse': val_mse,
                'val_mae': val_mae
            }, save_path)
            print(f"New best model saved! Validation Loss: {val_loss:.6f}")
        
        # Save checkpoint
        if epoch % args.checkpoint_interval == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss, config, 
                config['general']['checkpoint_dir']
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Brain-Inspired Neural Network")
    parser.add_argument("--config", type=str, default="config/config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--use-llm", action="store_true", 
                        help="Enable LLM integration for validation")
    parser.add_argument("--llm-validation-interval", type=int, default=5, 
                        help="Interval (in epochs) for LLM validation")
    parser.add_argument("--checkpoint-interval", type=int, default=10, 
                        help="Interval (in epochs) for saving checkpoints")
    
    args = parser.parse_args()
    main(args)
