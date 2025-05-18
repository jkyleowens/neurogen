"""
Training Script for Brain-Inspired Neural Network

This script handles the training process for the brain-inspired neural network,
including data loading, training loop, and LLM integration for validation.
"""

import os
import yaml
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import BrainInspiredNN
from utils.llm_interface import LLMInterface


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
    
    # Set up model
    model = setup_model(config)
    model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Set up optimizer and scheduler
    optimizer, scheduler = setup_optimizer(model, config)
    
    # Set up LLM interface if enabled
    llm_interface = None
    if args.use_llm:
        provider = config['llm'].get('provider', 'openai')
        provider_config = config['llm'].get(provider, {})
        
        llm_interface = LLMInterface(
            api_endpoint=config['llm'].get('api_endpoint', ''),
            model_name=provider_config.get('model_name', config['llm']['model_name']),
            max_tokens=config['llm'].get('max_tokens', 1024),
            temperature=config['llm'].get('temperature', 0.7),
            provider=provider,
            api_key=provider_config.get('api_key', None),
            embedding_dim=config['llm'].get('embedding_dim', 768)
        )
        print(f"LLM interface initialized with {provider} provider")
    
    # TODO: Load dataset
    # For now, we'll use dummy data
    # In a real implementation, you would load your actual dataset here
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    for epoch in range(1, num_epochs + 1):
        # Train epoch
        train_loss = train_epoch(model, train_dataloader, optimizer, device, epoch)
        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.6f}")
        
        # Validate
        val_loss = validate(model, val_dataloader, device)
        print(f"Epoch {epoch}/{num_epochs}, Validation Loss: {val_loss:.6f}")
        
        # LLM validation if enabled
        if args.use_llm and llm_interface is not None and epoch % args.llm_validation_interval == 0:
            # Get validation prompts from config
            validation_prompts = config['llm'].get('validation', {}).get('prompts', [
                "Explain how neural networks process information similar to the human brain.",
                "Describe the role of neuromodulators in learning and memory."
            ])
            
            # Run LLM validation
            llm_score, llm_results = llm_validation(model, llm_interface, validation_prompts, device)
            print(f"Epoch {epoch}/{num_epochs}, LLM Validation Score: {llm_score:.4f}")
            
            # Log LLM validation results
            for i, result in enumerate(llm_results):
                print(f"  Prompt {i+1} Score: {result['evaluation']['score']:.2f}")
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Save checkpoint
        if epoch % args.checkpoint_interval == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss, config, 
                config['general']['checkpoint_dir']
            )
    
    # Save final model
    final_checkpoint_path = os.path.join(config['general']['checkpoint_dir'], "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_checkpoint_path)
    print(f"Final model saved to {final_checkpoint_path}")


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
