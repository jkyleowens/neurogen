"""
Model State Reset Script

This script resets the internal states of a trained BrainInspiredNN model. 
It can be used to:
1. Fix models that have encountered training errors
2. Reset the persistent memory, hidden states, and neurotransmitter levels
3. Optionally reset only specific components while preserving others
4. Prepare a model for continued training with a clean slate

Usage:
python reset_model_state.py --model path/to/checkpoint.pt --output path/to/output.pt [options]
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
from datetime import datetime

# Try different import approaches to handle various project structures
try:
    from memory_tensor_fixes import optimize_memory_usage, print_gpu_memory_status
except ImportError:
    print("Warning: Could not import memory optimization functions")


def load_model(model_path, device):
    """
    Load a model from a checkpoint file, extracting the model, config, and other metadata.
    
    Args:
        model_path (str): Path to the model checkpoint
        device (torch.device): Device to load the model on
        
    Returns:
        tuple: (model, config, checkpoint_data)
    """
    print(f"Loading model from {model_path}...")
    
    # Load the checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print(f"Checkpoint loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)
    
    # Get config from checkpoint or create default
    config = checkpoint.get('config', {})
    
    # Try to load the model class
    try:
        # Try multiple import paths
        try:
            from src.model import BrainInspiredNN
        except ImportError:
            try:
                from model import BrainInspiredNN
            except ImportError:
                print("Error: Could not import BrainInspiredNN. Please check your project structure.")
                sys.exit(1)
        
        # Try to get model parameters from config
        input_size = config.get('input_size', 128)
        if 'model' in config and 'hidden_size' in config['model']:
            hidden_size = config['model']['hidden_size']
        else:
            hidden_size = config.get('controller', {}).get('hidden_size', 256)
        
        output_size = config.get('output_size', 1)
        
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
        
        # Create model with correct architecture
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
        
        # Load weights
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model state loaded successfully")
        except Exception as e:
            print(f"Warning: Error loading model state: {e}")
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
        model.eval()  # Set to eval mode
        
        return model, config, checkpoint
    
    except Exception as e:
        print(f"Error creating model: {e}")
        sys.exit(1)


def reset_model_state(model, reset_options):
    """
    Reset the internal state of the model according to specified options.
    
    Args:
        model (BrainInspiredNN): The model to reset
        reset_options (dict): Options specifying what to reset
        
    Returns:
        BrainInspiredNN: The reset model
    """
    print("Resetting model state...")
    
    # Reset hidden states if requested
    if reset_options['hidden_states']:
        if hasattr(model, 'hidden_states'):
            print("Resetting hidden states")
            model.hidden_states = None
        if hasattr(model, 'controller') and hasattr(model.controller, 'hidden'):
            print("Resetting controller hidden states")
            model.controller.hidden = None
    
    # Reset persistent memory if requested
    if reset_options['persistent_memory']:
        if hasattr(model, 'persistent_memories'):
            print("Resetting persistent memories")
            model.persistent_memories = None
        if hasattr(model, 'controller') and hasattr(model.controller, 'persistent_memory'):
            print("Resetting controller persistent memory")
            model.controller.persistent_memory = None
    
    # Reset neuromodulator levels if requested
    if reset_options['neuromodulator_levels']:
        if hasattr(model, 'neurotransmitter_levels'):
            print("Resetting neurotransmitter levels")
            model.neurotransmitter_levels = None
        
        # If requested, also reset the neuromodulator parameters
        if reset_options['neuromodulator_params']:
            if hasattr(model, 'neuromodulator'):
                print("Resetting neuromodulator parameters to defaults")
                if hasattr(model.neuromodulator, 'dopamine_scale'):
                    model.neuromodulator.dopamine_scale = torch.nn.Parameter(
                        torch.tensor([1.0], device=model.neuromodulator.dopamine_scale.device)
                    )
                if hasattr(model.neuromodulator, 'serotonin_scale'):
                    model.neuromodulator.serotonin_scale = torch.nn.Parameter(
                        torch.tensor([0.8], device=model.neuromodulator.serotonin_scale.device)
                    )
                if hasattr(model.neuromodulator, 'norepinephrine_scale'):
                    model.neuromodulator.norepinephrine_scale = torch.nn.Parameter(
                        torch.tensor([0.6], device=model.neuromodulator.norepinephrine_scale.device)
                    )
                if hasattr(model.neuromodulator, 'acetylcholine_scale'):
                    model.neuromodulator.acetylcholine_scale = torch.nn.Parameter(
                        torch.tensor([0.7], device=model.neuromodulator.acetylcholine_scale.device)
                    )
                if hasattr(model.neuromodulator, 'reward_decay'):
                    model.neuromodulator.reward_decay = torch.nn.Parameter(
                        torch.tensor([0.95], device=model.neuromodulator.reward_decay.device)
                    )
    
    # Reset input metadata if requested
    if reset_options['input_metadata']:
        if hasattr(model, 'input_metadata'):
            print("Resetting input metadata")
            model.input_metadata = None
    
    # Reset model weights if requested (reinitialization)
    if reset_options['model_weights']:
        print("Reinitializing model weights")
        # Reinitialize each module
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.GRU)):
                print(f"  Reinitializing {name}")
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, torch.nn.Conv1d):
                    torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, torch.nn.GRU):
                    for name, param in module.named_parameters():
                        if 'weight' in name:
                            torch.nn.init.orthogonal_(param)
                        elif 'bias' in name:
                            torch.nn.init.zeros_(param)
    
    # If requested, fix NaN values in parameters
    if reset_options['fix_nan_values']:
        print("Checking for and fixing NaN values in parameters")
        nan_found = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                nan_found = True
                print(f"  Found NaN values in {name}")
                # Replace NaN with zeros
                nan_mask = torch.isnan(param)
                param.data[nan_mask] = 0.0
        
        if not nan_found:
            print("  No NaN values found")
    
    print("Model state reset complete")
    return model


def main(args):
    """Main function to reset model state."""
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    print(f"Using device: {device}")
    
    # Try to optimize memory if available
    if 'optimize_memory_usage' in globals():
        optimize_memory_usage()
    
    # Load model
    model, config, checkpoint = load_model(args.model, device)
    
    # Prepare reset options
    reset_options = {
        'hidden_states': args.reset_hidden,
        'persistent_memory': args.reset_persistent_memory,
        'neuromodulator_levels': args.reset_neuromodulator_levels,
        'neuromodulator_params': args.reset_neuromodulator_params,
        'input_metadata': args.reset_input_metadata,
        'model_weights': args.reset_model_weights,
        'fix_nan_values': args.fix_nan_values
    }
    
    # Reset model state
    model = reset_model_state(model, reset_options)
    
    # Update checkpoint data
    checkpoint_data = {
        'epoch': checkpoint.get('epoch', 0),
        'model_state_dict': model.state_dict(),
        'config': config,
        'reset_info': {
            'original_model': args.model,
            'reset_options': reset_options,
            'reset_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    # Keep optimizer state dict if needed
    if 'optimizer_state_dict' in checkpoint and not args.reset_optimizer:
        checkpoint_data['optimizer_state_dict'] = checkpoint['optimizer_state_dict']
    
    # Keep history if needed
    if 'history' in checkpoint and not args.reset_history:
        checkpoint_data['history'] = checkpoint['history']
    
    # Keep data info if needed
    if 'data_info' in checkpoint:
        checkpoint_data['data_info'] = checkpoint['data_info']
    
    # Save reset model
    output_path = args.output
    if not output_path:
        # Generate default output path
        base_dir = os.path.dirname(args.model)
        base_name = os.path.basename(args.model)
        output_path = os.path.join(base_dir, f"reset_{base_name}")
    
    print(f"Saving reset model to {output_path}")
    torch.save(checkpoint_data, output_path)
    
    # Save reset info
    reset_info_path = f"{os.path.splitext(output_path)[0]}_reset_info.json"
    with open(reset_info_path, 'w') as f:
        json.dump({
            'original_model': args.model,
            'reset_options': reset_options,
            'reset_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    print(f"Reset info saved to {reset_info_path}")
    print("Model reset complete!")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset the internal state of a BrainInspiredNN model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="",
                        help="Path to save reset model (default: reset_[original_name])")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    
    # Reset options
    parser.add_argument("--reset-all", action="store_true",
                        help="Reset all model state components")
    parser.add_argument("--reset-hidden", action="store_true",
                        help="Reset hidden states")
    parser.add_argument("--reset-persistent-memory", action="store_true",
                        help="Reset persistent memory")
    parser.add_argument("--reset-neuromodulator-levels", action="store_true",
                        help="Reset neurotransmitter levels")
    parser.add_argument("--reset-neuromodulator-params", action="store_true",
                        help="Reset neuromodulator parameters to defaults")
    parser.add_argument("--reset-input-metadata", action="store_true",
                        help="Reset input metadata")
    parser.add_argument("--reset-model-weights", action="store_true",
                        help="Reinitialize model weights")
    parser.add_argument("--reset-optimizer", action="store_true",
                        help="Remove optimizer state from checkpoint")
    parser.add_argument("--reset-history", action="store_true",
                        help="Remove training history from checkpoint")
    parser.add_argument("--fix-nan-values", action="store_true",
                        help="Fix NaN values in parameters")
    
    args = parser.parse_args()
    
    # If reset-all is specified, set all individual reset options to True
    if args.reset_all:
        args.reset_hidden = True
        args.reset_persistent_memory = True
        args.reset_neuromodulator_levels = True
        args.reset_neuromodulator_params = True
        args.reset_input_metadata = True
        args.fix_nan_values = True
        # Note: We don't set reset_model_weights to True as it's more destructive
    
    # Ensure at least one reset option is selected
    if not any([
        args.reset_hidden, args.reset_persistent_memory, 
        args.reset_neuromodulator_levels, args.reset_neuromodulator_params,
        args.reset_input_metadata, args.reset_model_weights,
        args.fix_nan_values
    ]):
        print("Warning: No reset options selected. The model will be loaded and saved without changes.")
        print("Use --reset-all or specific reset options to modify the model state.")
    
    sys.exit(main(args))
