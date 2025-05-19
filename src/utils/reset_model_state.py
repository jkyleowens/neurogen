"""
Model State Reset Utilities

This module provides functions to reset the internal states of a BrainInspiredNN model.
"""

import torch
import numpy as np

def reset_model_state(model):
    """
    Reset the internal state of the model.
    
    Args:
        model (BrainInspiredNN): The model to reset
        
    Returns:
        BrainInspiredNN: The reset model
    """
    # Reset hidden states
    if hasattr(model, 'hidden'):
        model.hidden = None
    
    # Reset neurotransmitter levels
    if hasattr(model, 'neurotransmitter_levels'):
        model.neurotransmitter_levels = None
    
    # Reset controller hidden states if it exists
    if hasattr(model, 'controller') and hasattr(model.controller, 'hidden'):
        model.controller.hidden = None
    
    # Reset controller persistent memory if it exists
    if hasattr(model, 'controller') and hasattr(model.controller, 'persistent_memory'):
        model.controller.persistent_memory = None
    
    # Fix NaN values in parameters
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"Found NaN values in {name}")
            # Replace NaN with zeros
            nan_mask = torch.isnan(param)
            param.data[nan_mask] = 0.0
    
    return model
