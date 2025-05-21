#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify financial data loading functionality
"""

import sys
import os
import yaml
import numpy as np
import pandas as pd
import torch
from src.utils.financial_data_utils import load_financial_data

# Load config from file
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    print("Testing financial data loading...")
    
    # Load configuration
    config = load_config('config/financial_config.yaml')
    
    try:
        # Load financial data
        train_loader, val_loader, test_loader, data_info = load_financial_data(config)
        
        # Print info about the data
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Testing batches: {len(test_loader)}")
        print(f"Feature count: {data_info['feature_count']}")
        print(f"Sample features: {data_info['feature_names'][:5]}")
        
        # Check a sample batch
        x, y = next(iter(train_loader))
        print(f"Sample batch - X shape: {x.shape}, Y shape: {y.shape}")
        
        print("Data loading successful!")
        
    except Exception as e:
        print(f"Error testing data loading: {str(e)}")
        import traceback
        traceback.print_exc()
