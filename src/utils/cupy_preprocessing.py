"""
CuPy-exclusive data preprocessing utilities

This module contains functions for preprocessing data using only CuPy operations,
without relying on PyTorch until the actual neural network computation.
"""

import warnings
try:
    import cupy as cp
    USING_CUPY = True
except ImportError:
    import numpy as cp
    USING_CUPY = False
    warnings.warn("CuPy not available. Using NumPy instead.")

def normalize_features(data):
    """
    Normalize features using only CuPy operations.
    
    Args:
        data: CuPy array of shape (samples, features)
        
    Returns:
        Normalized data
    """
    mean = cp.mean(data, axis=0)
    std = cp.std(data, axis=0)
    
    # Handle zero standard deviation
    std = cp.where(std == 0, 1.0, std)
    
    return (data - mean) / std

def create_sequences(data, sequence_length):
    """
    Create sequences from data using only CuPy operations.
    
    Args:
        data: CuPy array of shape (samples, features)
        sequence_length: Length of each sequence
        
    Returns:
        Sequences of shape (n_sequences, sequence_length, features)
    """
    n_samples, n_features = data.shape
    n_sequences = n_samples - sequence_length + 1
    
    sequences = cp.zeros((n_sequences, sequence_length, n_features), dtype=data.dtype)
    
    for i in range(n_sequences):
        sequences[i] = data[i:i+sequence_length]
        
    return sequences

def split_xy(sequences, target_idx=-1, predict_offset=1):
    """
    Split sequences into input and target.
    
    Args:
        sequences: CuPy array of shape (n_sequences, sequence_length, features)
        target_idx: Index of the target feature
        predict_offset: Offset for prediction (1 = next step)
        
    Returns:
        x_sequences, y_targets
    """
    n_sequences = sequences.shape[0]
    
    if predict_offset > 0:
        # For future prediction
        if predict_offset >= sequences.shape[1]:
            raise ValueError(f"Prediction offset {predict_offset} is too large for sequence length {sequences.shape[1]}")
        
        x_sequences = sequences[:n_sequences-predict_offset]
        y_targets = sequences[predict_offset:, -1, target_idx]
    else:
        # For current step prediction
        x_sequences = sequences
        y_targets = sequences[:, -1, target_idx]
        
    return x_sequences, y_targets

def create_batches(x_sequences, y_targets, batch_size):
    """
    Create batches from sequences using only CuPy operations.
    
    Args:
        x_sequences: CuPy array of input sequences
        y_targets: CuPy array of target values
        batch_size: Batch size
        
    Returns:
        List of (x_batch, y_batch) tuples
    """
    n_samples = len(x_sequences)
    n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division
    
    batches = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        
        x_batch = x_sequences[start_idx:end_idx]
        y_batch = y_targets[start_idx:end_idx]
        
        batches.append((x_batch, y_batch))
        
    return batches

def preprocess_financial_data(data, config):
    """
    Preprocess financial data using only CuPy operations.
    
    Args:
        data: Dictionary or DataFrame with financial data
        config: Configuration dictionary
        
    Returns:
        x_sequences, y_targets ready for neural network (but still as CuPy arrays)
    """
    # Extract features
    feature_names = config.get('features', ['Open', 'High', 'Low', 'Close', 'Volume'])
    target_name = config.get('target', 'Close')
    sequence_length = config.get('sequence_length', 20)
    predict_next = config.get('predict_next_step', True)
    predict_offset = 1 if predict_next else 0
    normalize = config.get('normalize', True)
    
    # Extract feature arrays
    feature_arrays = []
    for feature in feature_names:
        if feature in data:
            # Convert to CuPy array
            feature_arrays.append(cp.asarray(data[feature]))
        else:
            warnings.warn(f"Feature {feature} not found in data")
            
    if not feature_arrays:
        raise ValueError("No valid features found in data")
        
    # Stack features
    X = cp.column_stack(feature_arrays)
    
    # Normalize if requested
    if normalize:
        X = normalize_features(X)
        
    # Create sequences
    sequences = create_sequences(X, sequence_length)
    
    # Find target index
    target_idx = feature_names.index(target_name) if target_name in feature_names else -1
    
    # Split into X and y
    x_sequences, y_targets = split_xy(sequences, target_idx=target_idx, predict_offset=predict_offset)
    
    return x_sequences, y_targets
