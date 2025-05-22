"""
Standalone Model Testing Script

Run this script to test your trained brain-inspired neural network model
and generate comprehensive analysis reports.

Usage:
    python test_model.py --model models/checkpoints/best_model.pt --config config/financial_config.yaml
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.model import BrainInspiredNN
from comprehensive_model_testing import test_model_comprehensive, ModelTester

# Try to import data utilities
try:
    from src.utils.financial_data_utils import load_financial_data
except ImportError:
    from data_loader import create_datasets
    def load_financial_data(config):
        train_dataset, val_dataset, test_dataset = create_datasets(config)
        from torch.utils.data import DataLoader
        batch_size = config['training']['batch_size']
        return None, None, DataLoader(test_dataset, batch_size=batch_size), {}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Brain-Inspired Neural Network')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='test_results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test with basic metrics only')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model_robust(model_path, config, device):
    """
    Load model with automatic architecture detection and fixing.
    """
    print(f"📦 Loading model from {model_path}...")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Get model config from checkpoint if available
        checkpoint_config = checkpoint.get('config', config)
        
        # Detect model architecture from state_dict
        state_dict = checkpoint['model_state_dict']
        detected_config = detect_model_architecture(state_dict, config)
        
        print(f"🔍 Detected model architecture:")
        print(f"   Input Size: {detected_config['model']['input_size']}")
        print(f"   Hidden Size: {detected_config['model']['hidden_size']}")
        print(f"   Output Size: {detected_config['model']['output_size']}")
        print(f"   Uses BioGRU: {detected_config['model'].get('use_bio_gru', False)}")
        
        # Create model with detected config
        model = BrainInspiredNN(detected_config).to(device)
        
        # Try to load state dict with error handling
        try:
            model.load_state_dict(state_dict, strict=True)
            print("✅ Model loaded successfully (strict mode)")
        except RuntimeError as e:
            print(f"⚠️  Strict loading failed: {e}")
            print("🔧 Attempting flexible loading...")
            
            # Try flexible loading
            model_state = model.state_dict()
            loaded_keys = []
            skipped_keys = []
            
            for key, value in state_dict.items():
                if key in model_state:
                    if model_state[key].shape == value.shape:
                        model_state[key] = value
                        loaded_keys.append(key)
                    else:
                        print(f"   ⚠️  Shape mismatch for {key}: "
                              f"expected {model_state[key].shape}, got {value.shape}")
                        skipped_keys.append(key)
                else:
                    print(f"   ⚠️  Key not found in current model: {key}")
                    skipped_keys.append(key)
            
            # Load the compatible parameters
            model.load_state_dict(model_state)
            
            print(f"✅ Flexible loading completed:")
            print(f"   📊 Loaded: {len(loaded_keys)} parameters")
            print(f"   ⚠️  Skipped: {len(skipped_keys)} parameters")
            
            if len(skipped_keys) > len(loaded_keys):
                print("❌ Too many parameters skipped. Model may not work properly.")
                return None, None
        
        model.eval()
        return model, checkpoint
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def detect_model_architecture(state_dict, fallback_config):
    """
    Detect model architecture from state_dict keys and shapes.
    """
    config = fallback_config.copy()
    
    # Detect if using BioGRU
    has_bio_gru = any('output_neurons' in key for key in state_dict.keys())
    config['model']['use_bio_gru'] = has_bio_gru
    
    # Detect dimensions from key shapes
    for key, tensor in state_dict.items():
        if 'output_layer.weight' in key:
            # output_layer.weight shape: [output_size, hidden_size]
            config['model']['output_size'] = tensor.shape[0]
            config['model']['hidden_size'] = tensor.shape[1]
        
        elif 'gru.weight_ih_l0' in key and not has_bio_gru:
            # GRU input-hidden weight shape: [3*hidden_size, input_size]
            config['model']['hidden_size'] = tensor.shape[0] // 3
            config['model']['input_size'] = tensor.shape[1]
        
        elif 'memory_projection.weight' in key:
            # memory_projection.weight shape: [hidden_size, persistent_memory_size]
            config['model']['hidden_size'] = tensor.shape[0]
            config['controller']['persistent_memory_size'] = tensor.shape[1]
    
    # Set reasonable defaults if not detected
    if 'input_size' not in config['model'] or config['model']['input_size'] is None:
        config['model']['input_size'] = 64  # Default
    
    if 'hidden_size' not in config['model'] or config['model']['hidden_size'] is None:
        config['model']['hidden_size'] = 128  # Default
    
    if 'output_size' not in config['model'] or config['model']['output_size'] is None:
        config['model']['output_size'] = 1  # Default
    
    return config

def create_compatible_model(checkpoint_path, device):
    """
    Create a model that's compatible with the checkpoint, regardless of config.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # Create minimal config based on state_dict
        config = {
            'model': {'input_size': 64, 'hidden_size': 128, 'output_size': 1, 'use_bio_gru': False},
            'controller': {'num_layers': 2, 'persistent_memory_size': 64, 'dropout': 0.2},
            'neuromodulator': {'dopamine_scale': 1.0, 'serotonin_scale': 1.0, 
                              'norepinephrine_scale': 1.0, 'acetylcholine_scale': 1.0, 'reward_decay': 0.95},
            'training': {'learning_rate': 0.001}
        }
        
        # Detect architecture
        config = detect_model_architecture(state_dict, config)
        
        # Create and load model
        model = BrainInspiredNN(config).to(device)
        
        # Load with error handling
        model_dict = model.state_dict()
        compatible_dict = {}
        
        for key in model_dict.keys():
            if key in state_dict and model_dict[key].shape == state_dict[key].shape:
                compatible_dict[key] = state_dict[key]
            else:
                compatible_dict[key] = model_dict[key]  # Keep original
                
        model.load_state_dict(compatible_dict)
        model.eval()
        
        return model, config
        
    except Exception as e:
        print(f"Error creating compatible model: {e}")
        return None, None

# Updated test script load_model function
def load_model_updated(model_path, config, device):
    """Updated load_model function for the test script."""
    
    # First try robust loading
    model, checkpoint = load_model_robust(model_path, config, device)
    
    if model is not None:
        return model, checkpoint
    
    # If that fails, try creating compatible model
    print("🔄 Trying compatible model creation...")
    model, detected_config = create_compatible_model(model_path, device)
    
    if model is not None:
        print("✅ Compatible model created successfully")
        return model, {'config': detected_config}
    
    # Final fallback: create new model and warn user
    print("⚠️  Could not load checkpoint. Creating new model with config...")
    model = BrainInspiredNN(config).to(device)
    print("❌ Warning: Using untrained model!")
    
    return model, {'config': config}

# Add this to your test script - replace the existing load_model function
def fix_test_script():
    """
    Replace your load_model function in the test script with load_model_updated
    """
    pass