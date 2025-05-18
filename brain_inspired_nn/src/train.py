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


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target, reward) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        # Move data to device
        data = data.to(device)
        target = target.to(device)
        reward = reward.to(device) if reward is not None else None
        
        # Reset gradients
        optimizer.zero_grad()
        
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
    class LLMDataset(torch.utils.data.Dataset):
        def __init__(self, llm_interface, prompts, size=1000, mode='train'):
            self.llm_interface = llm_interface
            self.size = size
            self.mode = mode
            self.prompts = prompts
            self.input_size = config.get('input_size', 128)
            self.output_size = config.get('output_size', 64)
            
            # Cache for storing LLM responses to avoid repeated API calls
            self.cache = {}
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            if idx in self.cache:
                return self.cache[idx]
            
            # Get prompt for current index
            prompt = self.prompts[idx % len(self.prompts)]
            
            try:
                # Get response from LLM
                llm_response = self.llm_interface.get_response(prompt)
                
                # Process LLM response into tensors
                input_tensor = self.llm_interface.process_input(llm_response, self.input_size)
                target_tensor = self.llm_interface.process_target(llm_response, self.output_size)
                reward = self.llm_interface.calculate_reward(llm_response)
                
                # Cache the processed results
                result = (input_tensor, target_tensor, reward)
                self.cache[idx] = result
                
                return result
                
            except Exception as e:
                print(f"Error processing LLM response for index {idx}: {e}")
                # Return zero tensors as fallback
                return (
                    torch.zeros(self.input_size),
                    torch.zeros(self.output_size),
                    torch.zeros(1)
                )

    # Initialize LLM interface
    llm_interface = LLMInterface(config['llm'])
    
    # Load prompts from configuration
    train_prompts = config['llm'].get('train_prompts', [
        "Default training prompt 1",
        "Default training prompt 2"
    ])
    val_prompts = config['llm'].get('val_prompts', [
        "Default validation prompt 1",
        "Default validation prompt 2"
    ])
    
    # Create train and validation datasets
    train_dataset = LLMDataset(
        llm_interface=llm_interface,
        prompts=train_prompts,
        size=config['training'].get('train_size', 1000),
        mode='train'
    )
    
    val_dataset = LLMDataset(
        llm_interface=llm_interface,
        prompts=val_prompts,
        size=config['training'].get('val_size', 200),
        mode='val'
    )
    
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
    
    # Create datasets and dataloaders
    train_dataset, val_dataset = create_datasets(config)
    batch_size = config['training'].get('batch_size', 32)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['training'].get('num_workers', 2)
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['training'].get('num_workers', 2)
    )
    
    # Set up model
    model = setup_model(config)
    model.to(device)
    
    # Set up optimizer and scheduler
    optimizer, scheduler = setup_optimizer(model, config)
    
    # Training loop
    for epoch in range(1, config['training']['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
        
        # Train for one epoch
        train_loss = train_epoch(model, train_dataloader, optimizer, device, epoch)
        print(f"Train loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_dataloader, device)
        print(f"Validation loss: {val_loss:.4f}")
        
        # LLM validation
        if args.use_llm and epoch % args.llm_validation_interval == 0:
            llm_val_loss = llm_validation(model, llm_interface, val_dataset.prompts, device)
            print(f"LLM Validation loss: {llm_val_loss:.4f}")
        
        # Save checkpoint
        if epoch % args.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, config, "checkpoints")
        
        # Step the scheduler
        if scheduler is not None:
            scheduler.step()

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
