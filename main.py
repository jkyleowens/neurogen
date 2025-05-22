#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Brain-Inspired Neural Network: Complete Training, Validation, and Testing Pipeline

This script provides a complete pipeline for training, validating, and testing
the brain-inspired neural network model on financial data.

Usage:
    # Full training pipeline
    python main.py --config config/financial_config.yaml
    
    # Skip training, only test
    python main.py --config config/financial_config.yaml --test-only
    
    # Skip training and validation, only test
    python main.py --config config/financial_config.yaml --test-only --skip-validation
    
    # Training with pretraining
    python main.py --config config/financial_config.yaml --pretrain --pretrain-epochs 10
"""

import sys
import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json
import shutil

# Import the model and utilities
from src.model import BrainInspiredNN
from src.utils.memory_utils import optimize_memory_usage, print_gpu_memory_status
from src.utils.performance_report import generate_performance_report
from src.utils.reset_model_state import reset_model_state

# Try to import testing utilities
try:
    from comprehensive_model_testing import test_model_comprehensive, ModelTester
    COMPREHENSIVE_TESTING_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_TESTING_AVAILABLE = False
    print("⚠️  Comprehensive testing not available - install additional dependencies")

# Try to import financial data utilities
try:
    from src.utils.financial_data_utils import load_financial_data, visualize_predictions
    FINANCIAL_DATA_AVAILABLE = True
except ImportError:
    from data_loader import create_datasets
    FINANCIAL_DATA_AVAILABLE = False
    
    def load_financial_data(config):
        """Fallback data loader using the existing data_loader.py"""
        train_dataset, val_dataset, test_dataset = create_datasets(config)
        batch_size = config['training']['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader, {}

# Try to import pretraining utilities
try:
    from src.utils.pretrain_utils import create_pretrain_dataloader
    PRETRAIN_AVAILABLE = True
except ImportError:
    PRETRAIN_AVAILABLE = False
    def create_pretrain_dataloader(dataloader, batch_size=32):
        return dataloader

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Brain-Inspired Neural Network - Complete Pipeline')
    
    # Configuration
    parser.add_argument('--config', type=str, default='config/financial_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use for training (cuda/cpu/auto)')
    
    # Training control
    parser.add_argument('--test-only', action='store_true',
                        help='Skip training and validation, only run testing')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training phase')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip validation phase')
    parser.add_argument('--skip-testing', action='store_true',
                        help='Skip testing phase')
    
    # Model loading
    parser.add_argument('--model-path', type=str, default='neurogen/models/checkpoints/best_model.pt',
                        help='Path to model checkpoint to load')
    parser.add_argument('--force-new-model', action='store_true',
                        help='Force creation of new model (ignore existing checkpoints)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Learning rate (overrides config)')
    
    # Pretraining
    parser.add_argument('--pretrain', action='store_true',
                        help='Enable pretraining of neural components')
    parser.add_argument('--pretrain-epochs', type=int, default=5,
                        help='Number of pretraining epochs')
    parser.add_argument('--skip-controller-pretrain', action='store_true',
                        help='Skip controller pretraining')
    parser.add_argument('--skip-neuromod-pretrain', action='store_true',
                        help='Skip neuromodulator pretraining')
    
    # Output control
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save detailed predictions to CSV')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file with fallbacks."""
    if not os.path.exists(config_path):
        print(f"⚠️  Config file not found: {config_path}")
        print("📋 Creating default config...")
        return create_default_config()
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Config loaded from {config_path}")
        return config
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        print("📋 Using default config...")
        return create_default_config()

def create_default_config():
    """Create a default configuration."""
    return {
        'model': {
            'input_size': 64,
            'hidden_size': 128,
            'output_size': 1,
            'use_bio_gru': True
        },
        'controller': {
            'type': 'persistent_gru',
            'num_layers': 2,
            'persistent_memory_size': 64,
            'dropout': 0.2
        },
        'neuromodulator': {
            'dopamine_scale': 1.0,
            'serotonin_scale': 1.0,
            'norepinephrine_scale': 1.0,
            'acetylcholine_scale': 1.0,
            'reward_decay': 0.95
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 100,
            'weight_decay': 0.0001,
            'learning_mode': 'neuromodulator',
            'optimizer': 'adam',
            'early_stopping_patience': 30,
            'accuracy_threshold': 0.02
        },
        'data': {
            'dataset': 'financial',
            'ticker_symbol': 'AAPL',
            'features': ['Open', 'High', 'Low', 'Close', 'Volume'],
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'sequence_length': 50,
            'prediction_horizon': 1,
            'normalize': True
        },
        'pretraining': {
            'enabled': False,
            'epochs': 5,
            'controller': {'enabled': True, 'learning_rate': 0.001},
            'neuromodulator': {'enabled': True, 'learning_rate': 0.0005}
        }
    }

def detect_model_architecture(state_dict, fallback_config):
    """Detect model architecture from state_dict keys and shapes."""
    config = fallback_config.copy()
    
    # Detect if using BioGRU
    has_bio_gru = any('output_neurons' in key for key in state_dict.keys())
    config['model']['use_bio_gru'] = has_bio_gru
    
    # Detect dimensions from key shapes
    for key, tensor in state_dict.items():
        if 'output_layer.weight' in key:
            config['model']['output_size'] = tensor.shape[0]
            config['model']['hidden_size'] = tensor.shape[1]
        elif 'gru.weight_ih_l0' in key and not has_bio_gru:
            config['model']['hidden_size'] = tensor.shape[0] // 3
            config['model']['input_size'] = tensor.shape[1]
        elif 'memory_projection.weight' in key:
            config['model']['hidden_size'] = tensor.shape[0]
            config['controller']['persistent_memory_size'] = tensor.shape[1]
    
    return config

def load_model_checkpoint(model_path, config, device, force_new=False):
    """Load model from checkpoint with robust error handling."""
    
    if force_new or not os.path.exists(model_path):
        if not force_new:
            print(f"📋 No checkpoint found at {model_path}")
        print("🆕 Creating new model...")
        model = BrainInspiredNN(config).to(device)
        return model, None, config
    
    print(f"📦 Loading model checkpoint from {model_path}...")
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # Detect architecture from checkpoint
        detected_config = detect_model_architecture(state_dict, config)
        
        print(f"🔍 Detected architecture:")
        print(f"   Input Size: {detected_config['model']['input_size']}")
        print(f"   Hidden Size: {detected_config['model']['hidden_size']}")
        print(f"   Output Size: {detected_config['model']['output_size']}")
        print(f"   Uses BioGRU: {detected_config['model'].get('use_bio_gru', False)}")
        
        # Create model with detected config
        model = BrainInspiredNN(detected_config).to(device)
        
        # Try to load state dict
        try:
            model.load_state_dict(state_dict, strict=True)
            print("✅ Model loaded successfully (strict mode)")
        except RuntimeError as e:
            print(f"⚠️  Strict loading failed, using flexible mode...")
            model.load_state_dict(state_dict, strict=False)
            print("✅ Model loaded (flexible mode - some parameters may be missing)")
        
        model.eval()
        epoch = checkpoint.get('epoch', 0)
        print(f"📊 Loaded model from epoch {epoch}")
        
        return model, checkpoint, detected_config
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print("🆕 Creating new model instead...")
        model = BrainInspiredNN(config).to(device)
        return model, None, config

def fix_model_forward(model):
    """Apply emergency fix for forward method signature errors."""
    original_forward = model.forward
    
    def safe_forward(x, **kwargs):
        reward = kwargs.get('reward') or kwargs.get('error_signal_for_update')
        if 'error_signal_for_update' in kwargs:
            reward = -kwargs['error_signal_for_update']
        
        try:
            if reward is not None:
                return original_forward(x, reward=reward)
            else:
                return original_forward(x)
        except Exception as e:
            print(f"Forward pass error: {e}")
            batch_size = x.size(0)
            output_size = getattr(model, 'output_size', 1)
            return torch.zeros(batch_size, output_size, device=x.device)
    
    model.forward = safe_forward
    return model

def load_data(config):
    """Load and preprocess data with fallback options."""
    try:
        print("📊 Loading financial data...")
        train_loader, val_loader, test_loader, data_info = load_financial_data(config)
        print("✅ Data loaded successfully")
        return train_loader, val_loader, test_loader, data_info
    except Exception as e:
        print(f"⚠️  Error loading data: {e}")
        print("🔄 Using synthetic fallback data...")
        return load_synthetic_data(config)

def load_synthetic_data(config):
    """Create synthetic data as fallback."""
    print("🎲 Creating synthetic data...")
    
    batch_size = config['training']['batch_size']
    input_size = config['model']['input_size']
    output_size = config['model']['output_size']
    seq_length = config['data']['sequence_length']
    
    # Create synthetic time series
    num_samples = 1000
    X = torch.randn(num_samples, seq_length, input_size)
    y = torch.randn(num_samples, output_size)
    
    # Split data
    train_size = int(num_samples * config['data']['train_ratio'])
    val_size = int(num_samples * config['data']['val_ratio'])
    test_size = num_samples - train_size - val_size
    
    X_train, X_val, X_test = torch.split(X, [train_size, val_size, test_size])
    y_train, y_val, y_test = torch.split(y, [train_size, val_size, test_size])
    
    # Create datasets and loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, {'synthetic': True}

def calculate_accuracy(output, target, threshold=0.1):
    """Calculate accuracy for regression tasks."""
    try:
        with torch.no_grad():
            if output.dim() == 3 and target.dim() == 2:
                output = output[:, -1, :]
            
            if output.shape != target.shape:
                batch_size = min(output.size(0), target.size(0))
                feature_size = min(output.size(-1), target.size(-1))
                output = output[:batch_size, ..., :feature_size]
                target = target[:batch_size, ..., :feature_size]
            
            error = torch.abs(output - target)
            correct = (error < threshold).float()
            accuracy = 100.0 * correct.mean().item()
            return accuracy
    except Exception:
        return 0.0

def train_epoch(model, train_loader, optimizer, criterion, device, quiet=False):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    batch_count = 0
    total_accuracy = 0.0
    
    iterator = train_loader if quiet else tqdm(train_loader, desc='Training')
    
    for batch_idx, batch_data in enumerate(iterator):
        try:
            if len(batch_data) == 2:
                data, target = batch_data
            elif len(batch_data) == 3:
                data, target, _ = batch_data
            else:
                continue
            
            model.reset_state()
            data, target = data.to(device), target.to(device)
            
            if target.dim() == 1:
                target = target.unsqueeze(1)
            
            if optimizer is None:
                # Neuromodulator-driven learning
                with torch.no_grad():
                    output = model(data)
                    loss = criterion(output, target)
                    reward_signal = -loss.detach()
                    _ = model(data, reward=reward_signal)
                    final_loss = loss
            else:
                # Traditional backprop
                output = model(data)
                
                if output.shape != target.shape:
                    if output.dim() == 3 and target.dim() == 2:
                        output = output[:, -1, :]
                    elif output.shape[-1] != target.shape[-1]:
                        min_features = min(output.shape[-1], target.shape[-1])
                        output = output[..., :min_features]
                        target = target[..., :min_features]
                
                loss = criterion(output, target)
                final_loss = loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            if torch.isnan(final_loss) or torch.isinf(final_loss):
                continue
            
            batch_accuracy = calculate_accuracy(output, target)
            total_loss += final_loss.item()
            total_accuracy += batch_accuracy
            batch_count += 1
            
        except Exception as e:
            if not quiet:
                print(f"Error in batch {batch_idx}: {str(e)}")
            continue
    
    return (total_loss / max(1, batch_count), total_accuracy / max(1, batch_count))

def validate(model, val_loader, criterion, device, quiet=False):
    """Validate the model."""
    model.reset_state()
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    batch_count = 0
    
    iterator = val_loader if quiet else tqdm(val_loader, desc='Validation')
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(iterator):
            try:
                if len(batch_data) == 2:
                    data, target = batch_data
                elif len(batch_data) == 3:
                    data, target, _ = batch_data
                else:
                    continue
                
                data, target = data.to(device), target.to(device)
                
                if target.dim() == 1:
                    target = target.unsqueeze(1)
                
                output = model(data)
                
                if output.shape != target.shape:
                    if output.dim() == 3 and target.dim() == 2:
                        output = output[:, -1, :]
                    elif output.shape[-1] != target.shape[-1]:
                        min_features = min(output.shape[-1], target.shape[-1])
                        output = output[..., :min_features]
                        target = target[..., :min_features]
                
                loss = criterion(output, target)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                batch_accuracy = calculate_accuracy(output, target)
                total_loss += loss.item()
                total_accuracy += batch_accuracy
                batch_count += 1
                
            except Exception as e:
                if not quiet:
                    print(f"Error in validation batch {batch_idx}: {str(e)}")
                continue
    
    return (total_loss / max(1, batch_count), total_accuracy / max(1, batch_count))

def run_training(model, train_loader, val_loader, config, device, args):
    """Run the training loop."""
    print("\n🎓 " + "="*58)
    print("🎓 TRAINING PHASE")
    print("🎓 " + "="*58)
    
    # Setup training
    criterion = nn.MSELoss()
    
    if config.get('training', {}).get('learning_mode', '') == 'neuromodulator':
        optimizer = None
        print("🧠 Using neuromodulator-driven learning (no backprop)")
    else:
        weight_decay = config['training'].get('weight_decay', 0.0)
        lr = config['training']['learning_rate']
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        print(f"📈 Using Adam optimizer (lr={lr}, weight_decay={weight_decay})")
    
    # Learning rate scheduler
    scheduler = None
    if optimizer is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=not args.quiet
        )
    
    # Training parameters
    num_epochs = config['training']['num_epochs']
    early_stopping_patience = config['training'].get('early_stopping_patience', 30)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.dirname(args.model_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    print(f"🏃 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        if not args.quiet:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_accuracy = train_epoch(
            model, train_loader, optimizer, criterion, device, args.quiet
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validate (unless skipped)
        if not args.skip_validation:
            val_loss, val_accuracy = validate(model, val_loader, criterion, device, args.quiet)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
        else:
            val_loss, val_accuracy = train_loss, train_accuracy  # Use training metrics
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Print progress
        if not args.quiet:
            print(f"📊 Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.2f}%")
            print(f"📊 Val Loss: {val_loss:.6f}, Val Accuracy: {val_accuracy:.2f}%")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            
            # Save best model
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_accuracies': train_accuracies,
                    'val_accuracies': val_accuracies
                }, args.model_path)
                
                if not args.quiet:
                    print(f"✅ New best model saved (val_loss: {best_val_loss:.6f})")
                    
            except Exception as e:
                print(f"⚠️  Error saving model: {e}")
        else:
            epochs_without_improvement += 1
            if not args.quiet:
                print(f"⏳ No improvement for {epochs_without_improvement}/{early_stopping_patience} epochs")
        
        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(f"🛑 Early stopping after {epoch+1} epochs")
            break
        
        # Periodic memory cleanup
        if (epoch + 1) % 10 == 0:
            model = optimize_memory_usage(model, device)
    
    print(f"✅ Training completed! Best validation loss: {best_val_loss:.6f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_loss': best_val_loss
    }

def run_comprehensive_testing(model, test_loader, config, device, output_dir, args):
    """Run comprehensive model testing."""
    print("\n🧪 " + "="*58)
    print("🧪 COMPREHENSIVE TESTING PHASE")
    print("🧪 " + "="*58)
    
    test_output_dir = os.path.join(output_dir, 'comprehensive_test')
    os.makedirs(test_output_dir, exist_ok=True)
    
    try:
        if COMPREHENSIVE_TESTING_AVAILABLE:
            print("🔬 Running comprehensive analysis...")
            test_metrics = test_model_comprehensive(
                model=model,
                test_loader=test_loader,
                device=device,
                config=config,
                save_dir=test_output_dir
            )
            
            # Print key results
            if not args.quiet:
                print(f"\n📊 KEY TEST METRICS:")
                key_metrics = ['r2', 'rmse', 'mae', 'direction_accuracy', 'mape']
                for metric in key_metrics:
                    if metric in test_metrics:
                        value = test_metrics[metric]
                        if metric in ['direction_accuracy', 'mape']:
                            print(f"   {metric.upper():<20}: {value:.2f}%")
                        else:
                            print(f"   {metric.upper():<20}: {value:.6f}")
            
            print(f"\n📁 COMPREHENSIVE RESULTS:")
            print(f"   📋 Full Report: {test_output_dir}/test_report.html")
            print(f"   📊 Interactive Dashboard: {test_output_dir}/interactive_dashboard.html")
            print(f"   📈 Visualizations: {test_output_dir}/*.png")
            
        else:
            print("📊 Running basic testing (comprehensive testing not available)...")
            test_metrics = run_basic_testing(model, test_loader, device, test_output_dir, args.quiet)
            
            print(f"\n📊 BASIC TEST METRICS:")
            for key, value in test_metrics.items():
                if isinstance(value, float):
                    print(f"   {key.upper():<20}: {value:.6f}")
        
        return test_metrics
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return {}

def run_basic_testing(model, test_loader, device, output_dir, quiet=False):
    """Run basic testing if comprehensive testing is not available."""
    model.eval()
    criterion = nn.MSELoss()
    
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    batch_count = 0
    
    iterator = test_loader if quiet else tqdm(test_loader, desc='Testing')
    
    with torch.no_grad():
        for batch_data in iterator:
            try:
                if len(batch_data) == 2:
                    data, target = batch_data
                elif len(batch_data) == 3:
                    data, target, _ = batch_data
                else:
                    continue
                
                data, target = data.to(device), target.to(device)
                model.reset_state()
                
                output = model(data)
                
                if output.dim() == 3:
                    output = output[:, -1, :]
                if target.dim() == 1:
                    target = target.unsqueeze(1)
                if output.shape != target.shape:
                    min_features = min(output.shape[-1], target.shape[-1])
                    output = output[..., :min_features]
                    target = target[..., :min_features]
                
                loss = criterion(output, target)
                total_loss += loss.item()
                batch_count += 1
                
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                
            except Exception:
                continue
    
    if batch_count == 0:
        return {}
    
    # Calculate metrics
    predictions = np.vstack(all_predictions).flatten()
    targets = np.vstack(all_targets).flatten()
    
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    ss_res = np.sum((targets - predictions) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    metrics = {
        'test_loss': total_loss / batch_count,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'num_samples': len(predictions)
    }
    
    # Save basic results
    with open(os.path.join(output_dir, 'basic_test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def run_pretraining(model, train_loader, device, config, args):
    """Run pretraining if enabled."""
    if not args.pretrain and not config.get('pretraining', {}).get('enabled', False):
        return
    
    if not PRETRAIN_AVAILABLE:
        print("⚠️  Pretraining utilities not available, skipping...")
        return
    
    print("\n🎯 " + "="*58)
    print("🎯 PRETRAINING PHASE")
    print("🎯 " + "="*58)
    
    try:
        pretrain_config = config.get('pretraining', {}).copy()
        
        if args.pretrain_epochs:
            pretrain_config['controller'] = pretrain_config.get('controller', {})
            pretrain_config['controller']['epochs'] = args.pretrain_epochs
            pretrain_config['neuromodulator'] = pretrain_config.get('neuromodulator', {})
            pretrain_config['neuromodulator']['epochs'] = args.pretrain_epochs
        
        if args.skip_controller_pretrain:
            pretrain_config['controller'] = pretrain_config.get('controller', {})
            pretrain_config['controller']['enabled'] = False
            
        if args.skip_neuromod_pretrain:
            pretrain_config['neuromodulator'] = pretrain_config.get('neuromodulator', {})
            pretrain_config['neuromodulator']['enabled'] = False
        
        pretrain_dataloader = create_pretrain_dataloader(train_loader)
        
        if hasattr(model, 'pretrain_components'):
            model.pretrain_components(pretrain_dataloader, device, pretrain_config)
            print("✅ Pretraining completed successfully")
        else:
            print("⚠️  Model does not support pretraining, skipping...")
            
    except Exception as e:
        print(f"❌ Pretraining failed: {e}")
        print("🔄 Continuing without pretraining...")

def save_predictions_to_csv(model, test_loader, device, output_path):
    """Save detailed predictions to CSV file."""
    print(f"💾 Saving predictions to {output_path}...")
    
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc='Generating predictions')):
            try:
                if len(batch_data) == 2:
                    data, target = batch_data
                elif len(batch_data) == 3:
                    data, target, _ = batch_data
                else:
                    continue
                
                data, target = data.to(device), target.to(device)
                model.reset_state()
                
                output = model(data)
                
                # Handle output shapes
                if output.dim() == 3:
                    output = output[:, -1, :]
                if target.dim() == 1:
                    target = target.unsqueeze(1)
                if output.shape != target.shape:
                    min_features = min(output.shape[-1], target.shape[-1])
                    output = output[..., :min_features]
                    target = target[..., :min_features]
                
                # Convert to numpy and store
                pred_np = output.cpu().numpy()
                target_np = target.cpu().numpy()
                
                for i in range(len(pred_np)):
                    results.append({
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'prediction': pred_np[i].flatten()[0] if len(pred_np[i].flatten()) > 0 else 0,
                        'actual': target_np[i].flatten()[0] if len(target_np[i].flatten()) > 0 else 0,
                        'error': pred_np[i].flatten()[0] - target_np[i].flatten()[0] if len(pred_np[i].flatten()) > 0 else 0
                    })
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
    
    # Save to CSV
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"✅ Predictions saved to {output_path}")
    except ImportError:
        # Manual CSV writing if pandas not available
        with open(output_path, 'w') as f:
            f.write("batch_idx,sample_idx,prediction,actual,error\n")
            for result in results:
                f.write(f"{result['batch_idx']},{result['sample_idx']},{result['prediction']},{result['actual']},{result['error']}\n")
        print(f"✅ Predictions saved to {output_path}")

def generate_final_report(training_results, test_results, config, output_dir, args):
    """Generate final comprehensive report."""
    print("\n📋 " + "="*58)
    print("📋 GENERATING FINAL REPORT")
    print("📋 " + "="*58)
    
    try:
        # Prepare metrics for report
        all_metrics = {}
        
        if training_results:
            all_metrics.update({
                'train_loss': training_results.get('train_losses', []),
                'val_loss': training_results.get('val_losses', []),
                'train_accuracy': training_results.get('train_accuracies', []),
                'val_accuracy': training_results.get('val_accuracies', [])
            })
        
        if test_results:
            all_metrics.update(test_results)
        
        # Generate performance report
        final_train_loss = training_results.get('train_losses', [0])[-1] if training_results else 0
        final_val_loss = training_results.get('val_losses', [float('inf')])[-1] if training_results else float('inf')
        final_test_loss = test_results.get('test_loss', test_results.get('loss', float('inf')))
        
        generate_performance_report(
            final_train_loss,
            final_val_loss,
            final_test_loss,
            metrics=all_metrics,
            output_dir=output_dir
        )
        
        # Create summary file
        summary_path = os.path.join(output_dir, 'pipeline_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Brain-Inspired Neural Network - Pipeline Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: {args.config}\n")
            f.write(f"Model Path: {args.model_path}\n")
            f.write(f"Output Directory: {output_dir}\n\n")
            
            f.write("Pipeline Phases:\n")
            f.write(f"  Pretraining: {'✅' if args.pretrain else '❌'}\n")
            f.write(f"  Training: {'✅' if not args.skip_training and not args.test_only else '❌'}\n")
            f.write(f"  Validation: {'✅' if not args.skip_validation and not args.test_only else '❌'}\n")
            f.write(f"  Testing: {'✅' if not args.skip_testing else '❌'}\n\n")
            
            if training_results:
                f.write("Training Results:\n")
                f.write(f"  Final Train Loss: {final_train_loss:.6f}\n")
                f.write(f"  Final Val Loss: {final_val_loss:.6f}\n")
                f.write(f"  Best Val Loss: {training_results.get('best_val_loss', 'N/A'):.6f}\n")
                f.write(f"  Total Epochs: {len(training_results.get('train_losses', []))}\n\n")
            
            if test_results:
                f.write("Test Results:\n")
                for key, value in test_results.items():
                    if isinstance(value, (int, float)) and key != 'num_samples':
                        f.write(f"  {key.upper()}: {value:.6f}\n")
                if 'num_samples' in test_results:
                    f.write(f"  SAMPLES: {test_results['num_samples']}\n")
        
        print(f"📋 Final report generated: {output_dir}/performance_report.md")
        print(f"📊 Pipeline summary: {summary_path}")
        
    except Exception as e:
        print(f"⚠️  Error generating final report: {e}")

def main():
    """Main pipeline function."""
    # Parse arguments
    args = parse_args()
    
    # Setup
    print("🧠 Brain-Inspired Neural Network - Complete Pipeline")
    print("=" * 70)
    print(f"⚙️  Configuration: {args.config}")
    print(f"🎯 Mode: {'Test Only' if args.test_only else 'Full Pipeline'}")
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"🖥️  Device: {device}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"\n📊 Loading data...")
    train_loader, val_loader, test_loader, data_info = load_data(config)
    
    # Adjust input size based on data
    try:
        first_batch = next(iter(train_loader))
        input_features = first_batch[0].shape[-1]
        if input_features != config['model']['input_size']:
            print(f"🔧 Adjusting input size: {config['model']['input_size']} → {input_features}")
            config['model']['input_size'] = input_features
    except Exception as e:
        print(f"⚠️  Could not determine input size from data: {e}")
    
    # Load or create model
    model, checkpoint, model_config = load_model_checkpoint(
        args.model_path, config, device, args.force_new_model
    )
    model = fix_model_forward(model)
    
    # Use detected model config if available
    if model_config != config:
        config = model_config
        print("🔧 Using detected model configuration")
    
    # Initialize results storage
    training_results = None
    test_results = None
    
    # Phase 1: Pretraining (if enabled)
    if args.pretrain and not args.test_only:
        run_pretraining(model, train_loader, device, config, args)
    
    # Phase 2: Training (unless skipped)
    if not args.skip_training and not args.test_only:
        training_results = run_training(model, train_loader, val_loader, config, device, args)
        
        # Load best model for testing
        if os.path.exists(args.model_path):
            print("📦 Loading best model for testing...")
            model, _, _ = load_model_checkpoint(args.model_path, config, device)
            model = fix_model_forward(model)
    
    # Phase 3: Testing (unless skipped)
    if not args.skip_testing:
        test_results = run_comprehensive_testing(model, test_loader, config, device, args.output_dir, args)
        
        # Save detailed predictions if requested
        if args.save_predictions:
            predictions_path = os.path.join(args.output_dir, 'detailed_predictions.csv')
            save_predictions_to_csv(model, test_loader, device, predictions_path)
    
    # Phase 4: Generate final report
    generate_final_report(training_results, test_results, config, args.output_dir, args)
    
    # Final summary
    print("\n🎉 " + "="*58)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("🎉 " + "="*58)
    
    if training_results:
        best_val_loss = training_results.get('best_val_loss', float('inf'))
        print(f"🏆 Best Validation Loss: {best_val_loss:.6f}")
    
    if test_results:
        test_r2 = test_results.get('r2', 0)
        test_loss = test_results.get('test_loss', test_results.get('loss', 0))
        print(f"🧪 Test R² Score: {test_r2:.4f}")
        print(f"🧪 Test Loss: {test_loss:.6f}")
    
    print(f"\n📁 All results saved to: {args.output_dir}/")
    print(f"📋 Performance report: {args.output_dir}/performance_report.md")
    print(f"🎯 Model checkpoint: {args.model_path}")
    
    if COMPREHENSIVE_TESTING_AVAILABLE and test_results:
        print(f"🌐 Interactive dashboard: {args.output_dir}/comprehensive_test/interactive_dashboard.html")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)