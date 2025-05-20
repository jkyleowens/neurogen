#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Financial Data Utilities for Brain-Inspired Neural Network

This module provides utilities for loading, preprocessing, and preparing
financial time series data for training and evaluation.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class TimeSeriesDataset(Dataset):
    """
    Dataset for time series prediction with additional features and indicators.
    
    This dataset creates sequences of historical price data and indicators,
    and targets of the next price point to predict.
    """
    def __init__(self, data, seq_length, target_column='Close', 
                 normalize=True, scale_targets=True, scaler=None):
        """
        Initialize the dataset.
        
        Args:
            data (pd.DataFrame): DataFrame containing price and indicator data
            seq_length (int): Length of historical sequence to use for prediction
            target_column (str): Column name to use as the prediction target
            normalize (bool): Whether to normalize the data
            scale_targets (bool): Whether to scale the target values
            scaler: Optional pre-fitted scaler
        """
        self.seq_length = seq_length
        self.target_column = target_column
        
        # Store column names for reference
        self.feature_columns = list(data.columns)
        self.target_idx = self.feature_columns.index(target_column)
        
        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Normalize data if requested
        if normalize:
            if scaler is None:
                if isinstance(normalize, str) and normalize.lower() == 'minmax':
                    self.scaler = MinMaxScaler()
                else:
                    self.scaler = StandardScaler()
                self.data = self.scaler.fit_transform(data.values)
            else:
                self.scaler = scaler
                self.data = self.scaler.transform(data.values)
        else:
            self.scaler = None
            self.data = data.values
            
        # Store whether targets are scaled
        self.scale_targets = scale_targets
        
        # Convert to tensor
        self.data_tensor = torch.tensor(self.data, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        # Get sequence
        x = self.data_tensor[idx:idx+self.seq_length]
        
        # Get target (next time step)
        if self.scale_targets or self.scaler is None:
            y = self.data_tensor[idx+self.seq_length, self.target_idx].reshape(-1)
        else:
            # Return unscaled target
            y = torch.tensor(self.data[idx+self.seq_length, self.target_idx], dtype=torch.float32).reshape(-1)
        
        return x, y
    
    def inverse_transform(self, y_pred, target_only=True):
        """
        Transform predictions back to original scale.
        
        Args:
            y_pred: Tensor or numpy array of predictions
            target_only: Whether to return only the target column values
        
        Returns:
            numpy array: Inverse-transformed predictions
        """
        if self.scaler is None:
            return y_pred
            
        # Convert to numpy if tensor
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
            
        if target_only:
            # Create dummy array for inverse transform
            if len(y_pred.shape) == 1:
                dummy = np.zeros((len(y_pred), len(self.feature_columns)))
                dummy[:, self.target_idx] = y_pred
                result = self.scaler.inverse_transform(dummy)
                return result[:, self.target_idx]
            else:
                # Handle batched predictions
                batch_size, seq_len = y_pred.shape
                dummy = np.zeros((batch_size, len(self.feature_columns)))
                dummy[:, self.target_idx] = y_pred[:, -1] if seq_len > 1 else y_pred.squeeze()
                result = self.scaler.inverse_transform(dummy)
                return result[:, self.target_idx]
        else:
            # Inverse transform whole array
            return self.scaler.inverse_transform(y_pred)


def add_technical_indicators(df):
    """
    Add technical indicators to the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with added indicators
    """
    # Create copy to avoid modifying the original
    result = df.copy()
    
    # Extract price data
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # Moving Averages
    result['MA5'] = close.rolling(window=5).mean()
    result['MA10'] = close.rolling(window=10).mean()
    result['MA20'] = close.rolling(window=20).mean()
    
    # Relative Strength Index (RSI)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    result['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    result['MACD'] = ema12 - ema26
    result['MACD_Signal'] = result['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    ma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    result['BB_Upper'] = ma20 + (std20 * 2)
    result['BB_Lower'] = ma20 - (std20 * 2)
    
    # Stochastic Oscillator
    low_14 = low.rolling(window=14).min()
    high_14 = high.rolling(window=14).max()
    result['%K'] = 100 * ((close - low_14) / (high_14 - low_14))
    result['%D'] = result['%K'].rolling(window=3).mean()
    
    # Average Directional Index (ADX)
    # Simple implementation - for a full implementation, consider using TA-Lib
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    result['ATR'] = tr.rolling(window=14).mean()
    
    # Price Rate of Change
    result['ROC'] = close.pct_change(periods=10) * 100
    
    # Volume Features
    result['Volume_MA10'] = volume.rolling(window=10).mean()
    result['Volume_Change'] = volume.pct_change() * 100
    
    # Price to Moving Average Ratios
    # Use pandas division which ensures element-wise operation
    result['Price_to_MA5'] = close / result['MA5']
    result['Price_to_MA20'] = close / result['MA20']
    
    # Handle any potential NaN values from division or other calculations
    result = result.replace([np.inf, -np.inf], np.nan)
    
    # For each column, use appropriate fillna method
    for col in result.columns:
        if col.startswith('Price_to_') or col.endswith('_Change'):
            # For ratio columns, fill NaN with 1.0
            result[col] = result[col].fillna(1.0)
        elif col.startswith('RSI') or col.startswith('%'):
            # For oscillators, fill with 50 (neutral)
            result[col] = result[col].fillna(50.0)
        else:
            # For other indicators, forward-fill then backfill
            result[col] = result[col].fillna(method='ffill').fillna(method='bfill')
    
    return result


def load_financial_data(config):
    """
    Load and preprocess financial data as specified in the config.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, scalers)
    """
    # Extract data parameters
    ticker = config['data']['ticker_symbol']
    start_date = config['data'].get('start_date', '2015-01-01')
    end_date = config['data'].get('end_date', datetime.now().strftime('%Y-%m-%d'))
    seq_length = config['data']['sequence_length']
    batch_size = config['training']['batch_size']
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    features = config['data'].get('features', ['Open', 'High', 'Low', 'Close', 'Volume'])
    include_indicators = config['data'].get('include_indicators', True)
    target_column = config['data'].get('target_column', 'Close')
    
    print(f"Loading financial data for {ticker} from {start_date} to {end_date}...")
    
    # Ensure dates are properly formatted for yfinance
    try:
        # Parse and reformat dates if they're strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
            
        # Download data using yfinance
        df = yf.download(ticker, start=start_date, end=end_date)
        
        # Check if data was successfully downloaded
        if df.empty:
            raise ValueError(f"No data downloaded for ticker {ticker}. Please check if the ticker symbol is valid.")
    except Exception as e:
        print(f"Error downloading data: {e}")
        # Create a minimal synthetic dataset for testing if download fails
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        df = pd.DataFrame(index=dates)
        df['Open'] = np.random.randn(len(df)) + 100
        df['High'] = df['Open'] + abs(np.random.randn(len(df)))
        df['Low'] = df['Open'] - abs(np.random.randn(len(df)))
        df['Close'] = df['Open'] + np.random.randn(len(df))
        df['Volume'] = np.random.randint(1000, 10000, size=len(df))
        print("WARNING: Using synthetic data due to download failure")
    
    # Ensure all required basic columns exist
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in downloaded data.")
    
    # Add technical indicators if requested
    if include_indicators:
        df = add_technical_indicators(df)
        
    # Select only the specified features
    if features == 'all':
        features = df.columns
    else:
        # Ensure target column is included
        if target_column not in features:
            features = list(features) + [target_column]
            
    # Select only requested features
    df = df[features]
    
    # Drop any rows with NaN (usually at the beginning due to indicators)
    df = df.dropna()
    
    # Split into train, val, test
    train_end = int(len(df) * train_ratio)
    val_end = train_end + int(len(df) * val_ratio)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Create datasets
    normalize_method = config['data'].get('normalize', 'standard')  # 'standard', 'minmax', or False
    
    train_dataset = TimeSeriesDataset(
        train_df, seq_length, target_column, normalize=normalize_method
    )
    
    # Use the same scaler for validation and test data
    val_dataset = TimeSeriesDataset(
        val_df, seq_length, target_column, 
        normalize=normalize_method, scaler=train_dataset.scaler
    )
    
    test_dataset = TimeSeriesDataset(
        test_df, seq_length, target_column, 
        normalize=normalize_method, scaler=train_dataset.scaler
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=config['data'].get('pin_memory', False)
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=config['data'].get('pin_memory', False)
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=config['data'].get('pin_memory', False)
    )
    
    # Return loaders and train dataset for reference to the scaler
    return train_loader, val_loader, test_loader, {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'feature_names': train_dataset.feature_columns,
        'target_column': target_column,
        'target_idx': train_dataset.target_idx
    }


def visualize_predictions(true_values, predictions, dates=None, title="Model Predictions vs Actual Values"):
    """
    Visualize the model predictions against actual values.
    
    Args:
        true_values (array-like): True/actual values
        predictions (array-like): Predicted values
        dates (array-like, optional): Date indices for x-axis
        title (str): Title for the plot
    """
    plt.figure(figsize=(12, 6))
    
    if dates is not None:
        plt.plot(dates, true_values, 'b-', label='Actual')
        plt.plot(dates, predictions, 'r-', label='Predicted')
        plt.xticks(rotation=45)
    else:
        plt.plot(true_values, 'b-', label='Actual')
        plt.plot(predictions, 'r-', label='Predicted')
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    return plt.gcf()
