"""
Brain-Inspired Neural Network Model

This module implements the main model that integrates all components:
- Persistent GRU controller
- Neuromodulator system
- Input preprocessing
- LLM interface

The model is designed to mimic certain aspects of brain functionality,
particularly the interaction between neural processing, neuromodulation,
sensory processing, and higher-level cognitive functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

from src.controller.persistent_gru import PersistentGRUController
from src.neuromodulator.neuromodulator import NeuromodulatorSystem, EnhancedRewardPredictor
from src.preprocessing.input_processor import InputProcessor


class BrainInspiredNN(nn.Module):
    """
    A brain-inspired neural network that integrates all components into a cohesive system.

    This model is designed to mimic certain aspects of brain functionality:
    - The controller acts as a central processing unit (like prefrontal cortex)
    - The neuromodulator system regulates neural activity based on rewards (like dopaminergic systems)
    - The input processor handles sensory information (like sensory cortices)

    The system preserves the biologically-inspired nature with the controller acting as
    a central module that controls the neuromodulator based on preprocessed inputs and reward signals.
    """

    def __init__(self, input_size, hidden_size, output_size,
                 persistent_memory_size=128, num_layers=2, dropout=0.1,
                 dopamine_scale=1.0, serotonin_scale=0.8,
                 norepinephrine_scale=0.6, acetylcholine_scale=0.7,
                 reward_decay=0.95):
        """
        Initialize the Brain-Inspired Neural Network.

        Args:
            input_size (int): Size of the input features
            hidden_size (int): Size of the hidden state
            output_size (int): Size of the output
            persistent_memory_size (int): Size of the persistent memory
            num_layers (int): Number of GRU layers
            dropout (float): Dropout probability
            dopamine_scale (float): Scaling factor for dopamine effects
            serotonin_scale (float): Scaling factor for serotonin effects
            norepinephrine_scale (float): Scaling factor for norepinephrine effects
            acetylcholine_scale (float): Scaling factor for acetylcholine effects
            reward_decay (float): Decay factor for reward signals
        """
        super(BrainInspiredNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Input Processor - handles sensory information preprocessing
        self.input_processor = InputProcessor(
            input_size=input_size,
            output_size=input_size,  # Maintain same dimensionality for controller input
            hidden_size=hidden_size,
            feature_size=hidden_size // 2,
            encoding_size=input_size,
            adaptation_rate=0.1
        )

        # GRU Controller - central processing unit
        self.controller = PersistentGRUController(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=hidden_size,  # Output to neuromodulator, not final output
            num_layers=num_layers,
            persistent_memory_size=persistent_memory_size,
            dropout=dropout
        )

        # Neuromodulator System - regulates neural activity based on rewards
        self.neuromodulator = NeuromodulatorSystem(
            input_size=input_size,
            hidden_size=hidden_size,
            dopamine_scale=dopamine_scale,
            serotonin_scale=serotonin_scale,
            norepinephrine_scale=norepinephrine_scale,
            acetylcholine_scale=acetylcholine_scale,
            reward_decay=reward_decay
        )

        # Reward Predictor - predicts rewards for reinforcement learning
        self.reward_predictor = EnhancedRewardPredictor(
            state_size=hidden_size,
            action_size=output_size,
            hidden_size=hidden_size
        )

        # Output Layer - generates final output
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Internal state tracking
        self.hidden_states = None
        self.persistent_memories = None
        self.neurotransmitter_levels = None
        self.input_metadata = None

    @staticmethod
    def preprocess_data(stock_data, config):
        """
        Neuromorphic preprocessing for financial time series data.
        
        This method transforms raw market data into biologically-inspired 
        representations for neural processing, with robust error handling and
        comprehensive validation to ensure model stability.
        
        Args:
            stock_data (pd.DataFrame): Raw financial time series data
            config (dict): Configuration parameters for neural processing
                
        Returns:
            dict: Structured data and metadata for neural architecture
        """
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        from torch.utils.data import TensorDataset, DataLoader
        import torch
        
        # Create a copy to avoid modifying the original data
        df = stock_data.copy()
        
        # CRITICAL FIX: Handle multi-index columns from Yahoo Finance
        if isinstance(df.columns, pd.MultiIndex):
            print("Detected multi-index columns - flattening column structure")
            # Extract just the first level of the multi-index (the attribute names)
            df.columns = df.columns.get_level_values(0)
        
        # Now we have a flattened DataFrame with standard column names
        available_columns = df.columns.tolist()
        print(f"Available data columns after processing: {available_columns}")
        
        # Extract and validate feature columns
        if 'feature_columns' in config['data']:
            feature_columns = config['data']['feature_columns']
            
            if feature_columns == 'all' or (isinstance(feature_columns, list) and 
                                        len(feature_columns) > 0 and 
                                        feature_columns[0] == 'all'):
                feature_columns = available_columns
            else:
                # IMPORTANT FIX: Handle tuple column references in config
                # Convert any tuple references to their string equivalents
                cleaned_feature_columns = []
                for feature in feature_columns:
                    if isinstance(feature, tuple) and len(feature) > 0:
                        # If it's a tuple like ('Close', 'AAPL'), take just the first element
                        cleaned_feature = feature[0]
                        print(f"Converting tuple column reference {feature} to {cleaned_feature}")
                    else:
                        cleaned_feature = feature
                    cleaned_feature_columns.append(cleaned_feature)
                
                feature_columns = cleaned_feature_columns
                
                # Validate that requested features exist
                valid_features = []
                for feature in feature_columns:
                    if feature in available_columns:
                        valid_features.append(feature)
                    else:
                        print(f"Warning: Feature '{feature}' not found in data. Skipping.")
                
                if not valid_features:
                    # Fall back to essential columns
                    basic_features = ['Open', 'High', 'Low', 'Close', 'Volume']
                    valid_features = [col for col in basic_features if col in available_columns]
                    print(f"No valid features found. Using basic features: {valid_features}")
                
                feature_columns = valid_features
        else:
            feature_columns = available_columns
        
        # Ensure we have at least one feature
        if not feature_columns:
            raise ValueError("No valid features could be identified in the data. Cannot proceed.")
        
        # Validate target column - CRITICAL FIX for tuple/string mismatch
        target_column = config['data']['target_column']
        
        # Convert target column from tuple to string if necessary
        if isinstance(target_column, tuple) and len(target_column) > 0:
            target_column = target_column[0]
            print(f"Converting tuple target column to {target_column}")
        
        if target_column not in available_columns:
            print(f"Warning: Target column '{target_column}' not found in data")
            
            # Try to set a fallback target
            if 'Close' in available_columns:
                target_column = 'Close'
                print(f"Using 'Close' as target column instead")
            elif len(available_columns) > 0:
                target_column = available_columns[0]
                print(f"Using '{available_columns[0]}' as target column instead")
            else:
                raise ValueError("No suitable target column found")
            
            # Update config
            config['data']['target_column'] = target_column
        
        print(f"Using {len(feature_columns)} features: {feature_columns}")
        print(f"Target column: {target_column}")
        
        # Calculate technical indicators
        if config['data'].get('use_technical_indicators', True):
            print("Calculating technical indicators...")
            
            try:
                # Moving averages - temporal integration like in visual cortex
                window_sizes = config['data'].get('ma_windows', [5, 10, 20, 50])
                for window in window_sizes:
                    df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
                
                # RSI - adaptive gain control mechanism
                rsi_window = config['data'].get('rsi_window', 14)
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=rsi_window).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=rsi_window).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # Bollinger Bands calculation with neuromorphic vectorization
                bb_window = config['data'].get('bb_window', 20)
                std_dev = config['data'].get('bb_std_dev', 2)
                
                # Middle band
                df['BB_Middle'] = df['Close'].rolling(window=bb_window).mean()
                
                # Standard deviation
                std_series = df['Close'].rolling(window=bb_window).std()
                
                # Pre-allocate arrays
                upper_values = np.full(len(df), np.nan)
                lower_values = np.full(len(df), np.nan)
                
                # Create boolean mask for valid entries
                valid_mask = (~df['BB_Middle'].isna()) & (~std_series.isna())
                
                # Extract valid indices
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_indices) > 0:  # CRITICAL: Check if we have valid data
                    # Extract corresponding values
                    middle_values = df['BB_Middle'].values[valid_indices]
                    std_values = std_series.values[valid_indices]
                    
                    # Calculate bands
                    upper_values[valid_indices] = middle_values + (std_dev * std_values)
                    lower_values[valid_indices] = middle_values - (std_dev * std_values)
                
                # Assign to dataframe
                df['BB_Upper'] = upper_values
                df['BB_Lower'] = lower_values
                
                # MACD
                fast = config['data'].get('macd_fast', 12)
                slow = config['data'].get('macd_slow', 26)
                signal = config['data'].get('macd_signal', 9)
                df['MACD'] = df['Close'].ewm(span=fast, adjust=False).mean() - df['Close'].ewm(span=slow, adjust=False).mean()
                df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
                df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
                
                # Additional indicators
                df['Momentum_5'] = df['Close'].pct_change(5)
                df['Momentum_10'] = df['Close'].pct_change(10)
                df['Volatility'] = df['Close'].rolling(window=bb_window).std() / df['Close'].rolling(window=bb_window).mean()
                df['ROC_5'] = df['Close'].pct_change(5) * 100
                df['ROC_10'] = df['Close'].pct_change(10) * 100
                
                if 'Volume' in available_columns:
                    df['Volume_ROC'] = df['Volume'].pct_change() * 100
                    df['Volume_MA_Ratio'] = df['Volume'] / df['Volume'].rolling(window=10).mean()
            
            except Exception as e:
                print(f"Error calculating technical indicators: {e}")
                print("Proceeding with basic features only")
                # Drop any columns that might have been partially calculated
                for col in df.columns:
                    if col not in available_columns:
                        df = df.drop(columns=[col])
            
            # Drop NaN values
            initial_rows = len(df)
            enhanced_data = df.dropna()
            dropped_rows = initial_rows - len(enhanced_data)
            
            # CRITICAL FIX: Ensure we have data after dropping NaNs
            if len(enhanced_data) == 0:
                print("Warning: No data remains after dropping NaN values. Using original data with forward-filled NaNs.")
                enhanced_data = df.fillna(method='ffill').fillna(method='bfill')  # Forward then backward fill NaNs
            elif dropped_rows > 0:
                print(f"Dropped {dropped_rows} rows with NaN values ({dropped_rows/initial_rows*100:.1f}% of data)")
        else:
            enhanced_data = df
        
        # Extract parameters
        sequence_length = config['data']['sequence_length']
        prediction_horizon = config['data']['prediction_horizon']
        train_ratio = config['data']['train_ratio']
        val_ratio = config['data']['val_ratio']
        
        # CRITICAL FIX: Ensure feature_columns contains only columns that exist in enhanced_data
        feature_columns = [col for col in feature_columns if col in enhanced_data.columns]
        
        # Find target column index
        if target_column not in feature_columns:
            feature_columns.append(target_column)
            print(f"Added target column '{target_column}' to feature list")
        
        target_idx = feature_columns.index(target_column)
        
        # CRITICAL CHECK: Ensure we have enough data for sequence + prediction
        min_required_length = sequence_length + prediction_horizon
        available_data_points = len(enhanced_data)
        
        if available_data_points < min_required_length:
            raise ValueError(
                f"Not enough data points ({available_data_points}) for sequence length "
                f"({sequence_length}) + prediction horizon ({prediction_horizon}). "
                f"Need at least {min_required_length} points.\n"
                f"Available: {available_data_points} data points after preprocessing.\n"
                f"Consider: 1) Extending the date range, 2) Reducing sequence length, or "
                f"3) Reducing prediction horizon."
            )
        
        print(f"Data has {available_data_points} points, requiring minimum of {min_required_length} (sequence + horizon)")
        
        # Extract features
        features = enhanced_data[feature_columns].values
        
        # Normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        # CRITICAL FIX: Ensure we have at least one sample to fit
        if len(features) == 0:
            raise ValueError("No data samples available for scaling")
        
        features_scaled = scaler.fit_transform(features)
        
        # Create sequences and targets
        X, y, rewards = [], [], []
        for i in range(len(features_scaled) - sequence_length - prediction_horizon + 1):
            # Input sequence
            X.append(features_scaled[i:i+sequence_length])
            
            # Target
            current_price = features_scaled[i+sequence_length-1, target_idx]
            future_price = features_scaled[i+sequence_length+prediction_horizon-1, target_idx]
            y.append(future_price)
            
            # Generate reward signal
            price_change = future_price - current_price
            reward_signal = np.sign(price_change) * np.sqrt(np.abs(price_change))
            
            # Volatility-adjusted reward
            if i >= 10:
                recent_prices = features_scaled[i+sequence_length-10:i+sequence_length, target_idx]
                volatility = np.std(recent_prices) / (np.mean(recent_prices) + 1e-8)
                reward_signal = reward_signal * (1 + volatility)
            
            rewards.append(reward_signal)
        
        # CRITICAL FIX: Ensure we have sequences
        if len(X) == 0:
            raise ValueError(
                "No sequences could be created. Check sequence length and prediction horizon.\n"
                f"Available data points: {len(features_scaled)}, Sequence length: {sequence_length}, "
                f"Prediction horizon: {prediction_horizon}"
            )
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        rewards = np.array(rewards).reshape(-1, 1)
        
        # Split data
        total_samples = len(X)
        train_size = max(1, int(total_samples * train_ratio))  # Ensure at least 1 sample
        val_size = max(1, int(total_samples * val_ratio))      # Ensure at least 1 sample
        
        # Adjust sizes if we don't have enough samples
        if train_size + val_size > total_samples:
            if total_samples > 1:
                train_size = total_samples // 2
                val_size = total_samples - train_size
            else:
                train_size = 1
                val_size = 0
        
        # Create splits
        X_train, y_train, rewards_train = X[:train_size], y[:train_size], rewards[:train_size]
        X_val, y_val, rewards_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size], rewards[train_size:train_size+val_size]
        X_test, y_test, rewards_test = X[train_size+val_size:], y[train_size+val_size:], rewards[train_size+val_size:]
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        rewards_train = torch.FloatTensor(rewards_train)
        
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)
        rewards_val = torch.FloatTensor(rewards_val)
        
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)
        rewards_test = torch.FloatTensor(rewards_test)
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train, rewards_train)
        val_dataset = TensorDataset(X_val, y_val, rewards_val)
        test_dataset = TensorDataset(X_test, y_test, rewards_test)
        
        # Create DataLoaders with adaptive batch sizes
        batch_size = min(config['training'].get('batch_size', 32), len(train_dataset))  # Ensure batch size <= dataset size
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=torch.cuda.is_available()
        )
        
        val_batch_size = min(batch_size, len(val_dataset)) if len(val_dataset) > 0 else 1
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=val_batch_size, 
            shuffle=False,
            pin_memory=torch.cuda.is_available()
        )
        
        test_batch_size = min(batch_size, len(test_dataset)) if len(test_dataset) > 0 else 1
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=test_batch_size, 
            shuffle=False,
            pin_memory=torch.cuda.is_available()
        )
        
        # Calculate statistics
        try:
            train_mean = X_train.mean(dim=(0, 1)).numpy()
            train_std = X_train.std(dim=(0, 1)).numpy()
            reward_mean = rewards_train.mean().item()
            reward_std = rewards_train.std().item()
        except (RuntimeError, ValueError) as e:
            print(f"Warning: Could not calculate statistics: {e}")
            train_mean = np.zeros(X_train.shape[2]) if X_train.shape[0] > 0 else np.array([])
            train_std = np.ones(X_train.shape[2]) if X_train.shape[0] > 0 else np.array([])
            reward_mean = 0.0
            reward_std = 1.0
        
        # Compile data info
        data_info = {
            'scaler': scaler,
            'feature_columns': feature_columns,
            'target_column': target_column,
            'target_idx': target_idx,
            'dataloaders': {
                'train': train_dataloader,
                'val': val_dataloader,
                'test': test_dataloader
            },
            'shapes': {
                'X_train': X_train.shape,
                'y_train': y_train.shape,
                'rewards_train': rewards_train.shape
            },
            'statistics': {
                'train_mean': train_mean,
                'train_std': train_std,
                'reward_mean': reward_mean,
                'reward_std': reward_std
            },
            'sequence_info': {
                'sequence_length': sequence_length,
                'prediction_horizon': prediction_horizon,
                'available_points': available_data_points,
                'created_sequences': len(X)
            }
        }
        
        print(f"Data preparation complete:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        
        return data_info


    @staticmethod
    def setup_model(config, input_shape):
        """
        Set up the model with financial-specific parameters.
        """
        input_size = input_shape[2]  # Number of features per time step
        hidden_size = config['controller']['hidden_size']
        output_size = 1  # Single value prediction (target price)
        persistent_memory_size = config['controller']['persistent_memory_size']

        dopamine_scale = config['neuromodulator']['dopamine_scale']
        serotonin_scale = config['neuromodulator']['serotonin_scale']
        norepinephrine_scale = config['neuromodulator']['norepinephrine_scale']
        acetylcholine_scale = config['neuromodulator']['acetylcholine_scale']
        reward_decay = config['neuromodulator']['reward_decay']

        num_layers = config['controller']['num_layers']
        dropout = config['controller']['dropout']

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

    def forward(self, x, hidden=None, persistent_memory=None, external_reward=None):
        """
        Forward pass of the Brain-Inspired Neural Network with robust neuromorphic processing.
        
        This method implements a biologically-inspired information flow that mimics cortical 
        processing hierarchies, neuromodulatory dynamics, and persistent memory integration:
        
        1. Input preprocessing (sensory cortex-like processing)
        2. Controller processing (prefrontal cortex-like central executive)
        3. Reward prediction (basal ganglia-like function)
        4. Neuromodulation (brainstem-like neurotransmitter regulation)
        5. Output generation (motor cortex-like response selection)
        
        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
                            or (batch_size, input_size)
            hidden (torch.Tensor, optional): Initial hidden state for recurrent network
            persistent_memory (torch.Tensor, optional): Initial persistent memory state
            external_reward (torch.Tensor, optional): External reward signal for neuromodulation
                
        Returns:
            tuple: (outputs, predicted_rewards)
        """
        # Ensure input has correct dimensions (batch_size, sequence_length, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension for single-step inputs
        
        # Extract dimensions
        batch_size, seq_length, _ = x.size()
        device = x.device
        
        # Validate input dimensions and adapt if necessary
        if x.size(2) != self.input_size:
            print(f"Input dimension mismatch. Expected {self.input_size}, got {x.size(2)}. Adapting.")
            if x.size(2) < self.input_size:
                # Pad with zeros if input is smaller than expected
                padding = torch.zeros(batch_size, seq_length, self.input_size - x.size(2), device=device)
                x = torch.cat([x, padding], dim=2)
            else:
                # Truncate if input is larger than expected
                x = x[:, :, :self.input_size]
        
        # Initialize states if not provided
        if hidden is None or persistent_memory is None:
            try:
                hidden, persistent_memory = self.controller.init_hidden(batch_size, device)
            except Exception as e:
                print(f"State initialization error: {e}. Using default initialization.")
                # Default initialization
                if isinstance(self.controller, nn.Module):
                    if hasattr(self.controller, 'hidden_size') and hasattr(self.controller, 'num_layers'):
                        hidden_size = self.controller.hidden_size
                        num_layers = self.controller.num_layers
                        hidden = torch.zeros(num_layers, batch_size, hidden_size, device=device)
                        persistent_memory = torch.zeros(batch_size, 
                                                    self.controller.persistent_memory_size if hasattr(self.controller, 'persistent_memory_size') else hidden_size, 
                                                    device=device)
                    else:
                        hidden = torch.zeros(batch_size, self.hidden_size, device=device)
                        persistent_memory = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # Preprocess input through input processor
        try:
            processed_x, input_metadata = self.input_processor(x)
            self.input_metadata = input_metadata
        except Exception as e:
            print(f"Input processing error: {e}. Using simplified preprocessing.")
            # Simplified preprocessing fallback
            processed_x = (x - x.mean(dim=(0, 1), keepdim=True)) / (x.std(dim=(0, 1), keepdim=True) + 1e-8)
            self.input_metadata = {"error": "Input processing failed, using normalized input"}
        
        # Process through controller
        try:
            controller_outputs, hidden_states, persistent_memories = self.controller(
                processed_x, hidden, persistent_memory
            )
            
            # Store states for next forward pass or analysis
            self.hidden_states = hidden_states
            self.persistent_memories = persistent_memories
        except Exception as e:
            print(f"Controller processing error: {e}. Using emergency linear projection.")
            # Emergency linear projection fallback
            controller_outputs = torch.zeros(batch_size, seq_length, self.hidden_size, device=device)
            for t in range(seq_length):
                controller_outputs[:, t, :] = F.linear(
                    processed_x[:, t, :],
                    torch.ones(self.hidden_size, processed_x.size(2), device=device) / processed_x.size(2),
                    torch.zeros(self.hidden_size, device=device)
                )
            
            # Keep any valid states from previous iterations or use defaults
            self.hidden_states = hidden if isinstance(hidden, torch.Tensor) else torch.zeros(batch_size, self.hidden_size, device=device)
            self.persistent_memories = persistent_memory if isinstance(persistent_memory, torch.Tensor) else torch.zeros(batch_size, self.hidden_size, device=device)
        
        # Process each time step with neuromodulation
        outputs = []
        predicted_rewards = []
        neurotransmitter_states = []
        
        for t in range(seq_length):
            try:
                # Get controller output for current time step
                controller_output_t = controller_outputs[:, t, :]
                
                # Generate preliminary action
                action_t = self.output_layer(controller_output_t)
                
                # Predict or use external reward
                if external_reward is None:
                    # Predict reward if external reward not provided
                    reward_t = self.reward_predictor(controller_output_t, action_t)
                else:
                    # Handle various external reward formats
                    if external_reward.dim() > 1 and external_reward.size(1) > t:
                        # If reward has time dimension and it's long enough
                        reward_t = external_reward[:, t].unsqueeze(1) if external_reward[:, t].dim() == 1 else external_reward[:, t]
                    else:
                        # Use the same reward for all time steps
                        reward_t = external_reward.unsqueeze(1) if external_reward.dim() == 1 else external_reward
                
                # Apply neuromodulation
                modulated_output_t, neurotransmitter_levels_t = self.neuromodulator(
                    processed_x[:, min(t, processed_x.size(1)-1), :], 
                    controller_output_t, 
                    reward_t
                )
                
                # Generate final output
                final_output_t = self.output_layer(modulated_output_t)
                
                # Store outputs, rewards, and neuromodulator states
                outputs.append(final_output_t)
                predicted_rewards.append(reward_t)
                neurotransmitter_states.append(neurotransmitter_levels_t)
                
            except Exception as e:
                print(f"Error in time step {t}: {e}. Using failsafe output.")
                # Emergency output generation
                emergency_output = torch.zeros(batch_size, self.output_size, device=device)
                emergency_reward = torch.zeros(batch_size, 1, device=device)
                default_neurotransmitters = {
                    'dopamine': torch.ones(batch_size, 1, device=device) * 0.5,
                    'serotonin': torch.ones(batch_size, 1, device=device) * 0.5,
                    'norepinephrine': torch.ones(batch_size, 1, device=device) * 0.5,
                    'acetylcholine': torch.ones(batch_size, 1, device=device) * 0.5
                }
                
                outputs.append(emergency_output)
                predicted_rewards.append(emergency_reward)
                neurotransmitter_states.append(default_neurotransmitters)
        
        # Store final neurotransmitter state
        self.neurotransmitter_levels = neurotransmitter_states[-1]
        
        # Stack outputs and rewards along sequence dimension
        outputs = torch.stack(outputs, dim=1)
        predicted_rewards = torch.stack(predicted_rewards, dim=1)
        
        # Ensure outputs have the expected shape
        if outputs.size(2) != self.output_size:
            print(f"Output dimension mismatch. Reshaping from {outputs.size()} to match expected output size {self.output_size}.")
            outputs = outputs.view(batch_size, seq_length, self.output_size)
        
        # Ensure rewards have consistent shape
        if predicted_rewards.size(2) != 1 and predicted_rewards.size(2) != self.output_size:
            predicted_rewards = predicted_rewards.view(batch_size, seq_length, 1)
        
        return outputs, predicted_rewards

    def get_neurotransmitter_levels(self):
        """
        Retrieve current neurotransmitter levels for analysis or visualization.
        
        Returns:
            dict: Dictionary containing levels of dopamine, serotonin, norepinephrine, and acetylcholine
        """
        if hasattr(self, 'neurotransmitter_levels') and self.neurotransmitter_levels is not None:
            return self.neurotransmitter_levels
        else:
            # Return baseline levels if not available
            device = next(self.parameters()).device
            return {
                'dopamine': torch.tensor(0.5, device=device),
                'serotonin': torch.tensor(0.5, device=device),
                'norepinephrine': torch.tensor(0.5, device=device),
                'acetylcholine': torch.tensor(0.5, device=device)
            }
    
    # Add to the BrainInspiredNN class
    def reshape_output_for_loss(self, outputs, targets):
        """
        Reshape model outputs to match target shape for loss calculation.
        
        Args:
            outputs (torch.Tensor): Model outputs
            targets (torch.Tensor): Target values
            
        Returns:
            torch.Tensor: Reshaped outputs compatible with targets for loss calculation
        """
        # If outputs has sequence dimension but targets don't
        if outputs.dim() == 3 and targets.dim() == 2:
            batch_size, seq_len, output_dim = outputs.size()
            
            # Get last time step predictions (most relevant for future prediction)
            return outputs[:, -1, :]
        else:
            # No reshape needed
            return outputs

    def get_neurotransmitter_levels(self):
        """
        Get the current neurotransmitter levels for analysis and visualization.
        
        Returns:
            dict: Dictionary of neurotransmitter levels
        """
        if hasattr(self, 'neurotransmitter_levels') and self.neurotransmitter_levels is not None:
            return self.neurotransmitter_levels
        else:
            # Return default levels if not available
            device = next(self.parameters()).device
            return {
                'dopamine': torch.tensor(0.5, device=device),
                'serotonin': torch.tensor(0.5, device=device),
                'norepinephrine': torch.tensor(0.5, device=device),
                'acetylcholine': torch.tensor(0.5, device=device)
            }

    def reset_computational_graph(self):
        """Reset stateful components to prevent graph retention between batches."""
        self.hidden_states = None
        self.persistent_memories = None
        self.neurotransmitter_levels = None
        self.input_metadata = None

        if hasattr(self.input_processor, 'reset_adaptation'):
            self.input_processor.reset_adaptation()

        if hasattr(self.controller, 'reset_states'):
            self.controller.reset_states()

        if hasattr(self.neuromodulator, 'reset'):
            self.neuromodulator.reset()

    def get_neurotransmitter_levels(self):
        """
        Get current neurotransmitter levels with safe tensor extraction.

        Returns:
            dict: Dictionary of neurotransmitter levels as Python scalars
        """
        if self.neurotransmitter_levels is None:
            return {
                'dopamine': 0.5,
                'serotonin': 0.5,
                'norepinephrine': 0.5,
                'acetylcholine': 0.5
            }

        nt_levels = {}
        for name, level in self.neurotransmitter_levels.items():
            if isinstance(level, torch.Tensor):
                if level.numel() == 1:
                    nt_levels[name] = level.item()
                else:
                    nt_levels[name] = level.mean().item()
            else:
                nt_levels[name] = float(level)

        return nt_levels

    def implement_neurogenesis(self, reward_prediction_error):
        """
        Dynamically grow new neurons in response to high prediction errors.
        Areas with consistent errors receive more computational resources.
        """
        # Threshold for significant error
        error_threshold = 0.5
        
        if abs(reward_prediction_error) > error_threshold:
            # Identify which brain region needs expansion based on error type
            target_layer = "hidden" if reward_prediction_error > 0 else "persistent_memory"
            
            # Add new neurons with random initial connections
            if target_layer == "hidden":
                # Expand hidden layer dimensionality
                new_hidden_dim = self.hidden_size + int(10 * abs(reward_prediction_error))
                self._expand_layer(self.hidden_size, new_hidden_dim)
                self.hidden_size = new_hidden_dim
            else:
                # Expand persistent memory
                new_memory_dim = self.persistent_memory_size + int(5 * abs(reward_prediction_error))
                self._expand_memory(self.persistent_memory_size, new_memory_dim)
                self.persistent_memory_size = new_memory_dim
    def neuromodulator_guided_pruning(self, loss_per_neuron):
        """
        Prune neurons based on their contribution to loss, guided by neuromodulator levels.
        
        Higher dopamine -> more aggressive pruning of poorly performing neurons
        Higher serotonin -> more protection of consistently performing neurons
        Higher acetylcholine -> stronger consolidation of successful pathways
        """
        # Get current neuromodulator levels
        dopamine = self.neurotransmitter_levels['dopamine']
        serotonin = self.neurotransmitter_levels['serotonin']
        acetylcholine = self.neurotransmitter_levels['acetylcholine']
        
        # Calculate pruning threshold - higher dopamine allows more aggressive pruning
        pruning_threshold = 0.7 * (1 + dopamine)
        
        # Calculate protection threshold - higher serotonin protects more neurons
        protection_threshold = 0.3 * (1 + serotonin)
        
        # Apply pruning: neurons with loss > pruning_threshold get suppressed
        # Neurons with loss < protection_threshold get strengthened
        for i, loss in enumerate(loss_per_neuron):
            if loss > pruning_threshold:
                # Suppress this neuron by applying a mask in forward pass
                self.pruning_mask[i] = 0.1  # Reduce to 10% activity
            elif loss < protection_threshold:
                # Strengthen this neuron proportional to acetylcholine
                self.consolidation_factor[i] = 1.0 + (0.5 * acetylcholine)
    def homeostatic_regulation(self, layer_activities):
        """
        Regulate neural activity to maintain stable firing rates.
        This prevents neural saturation or silence.
        """
        target_activity = 0.3  # Target activity level (30% of neurons active)
        
        for layer_name, activity in layer_activities.items():
            current_activity = torch.mean(activity > 0).item()
            
            # If layer is too active, increase inhibition
            if current_activity > target_activity * 1.2:
                self.layer_inhibition[layer_name] *= 1.05
            # If layer is too quiet, decrease inhibition
            elif current_activity < target_activity * 0.8:
                self.layer_inhibition[layer_name] *= 0.95