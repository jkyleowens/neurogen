#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the neural component pretraining functionality.
This script loads a small subset of data and tests the pretraining process
to ensure it works correctly.
"""

import os
import sys
import torch
import yaml
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.model import BrainInspiredNN
from src.utils.pretrain_utils import create_pretrain_dataloader
from src.utils.financial_data_utils import load_financial_data

def test_pretraining():
    """Test the pretraining functionality"""
    print("=== Testing Neural Component Pretraining ===\n")
    
    # Load config
    config_path = 'config/financial_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config for testing
    config['data']['ticker_symbol'] = 'AAPL'  # Use Apple stock data
    config['data']['start_date'] = '2020-01-01'  # Use smaller date range
    config['data']['end_date'] = '2020-12-31'
    config['training']['batch_size'] = 16
    
    # Modify pretraining config
    config['pretraining'] = {
        'enabled': True,
        'epochs': 2,  # Short pretraining for testing
        'controller': {
            'enabled': True,
            'learning_rate': 0.001,
            'epochs': 2
        },
        'neuromodulator': {
            'enabled': True,
            'learning_rate': 0.0005,
            'epochs': 2
        }
    }
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load a small subset of data
    print("Loading data...")
    train_loader, val_loader, test_loader, data_info = load_financial_data(config)
    
    # Create an even smaller dataloader for pretraining test
    pretrain_loader = create_pretrain_dataloader(train_loader, batch_size=8)
    
    # Check if we have data
    try:
        batch = next(iter(pretrain_loader))
        print(f"Data loaded successfully. Batch shape: {batch[0].shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Adjust input size based on data
    input_features = batch[0].shape[-1]
    if input_features != config['model']['input_size']:
        print(f"Adjusting model input size from {config['model']['input_size']} to {input_features}")
        config['model']['input_size'] = input_features
    
    # Create model
    print("\nCreating model...")
    model = BrainInspiredNN(config).to(device)
    
    # Test pretraining
    print("\nTesting pretraining...")
    try:
        model.pretrain_components(pretrain_loader, device, config['pretraining'])
        print("\nPretraining completed successfully!")
    except Exception as e:
        print(f"\nError during pretraining: {e}")
    
    # Try a forward pass to ensure the model still works
    print("\nTesting forward pass after pretraining...")
    try:
        x, y = next(iter(train_loader))
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            output = model(x)
        print(f"Forward pass successful. Output shape: {output.shape}")
        
        # Test with reward
        output_with_reward = model(x, reward=torch.tensor(-0.5, device=device))
        print(f"Forward pass with reward successful. Output shape: {output_with_reward.shape}")
        
    except Exception as e:
        print(f"Error during forward pass: {e}")
    
    print("\n=== Pretraining Test Complete ===")

if __name__ == "__main__":
    test_pretraining()
