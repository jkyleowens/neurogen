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

from controller.persistent_gru import PersistentGRUController
from neuromodulator.neuromodulator import NeuromodulatorSystem, RewardPredictor
from preprocessing.input_processor import InputProcessor
from utils.llm_interface import LLMInterface


class BrainInspiredNN(nn.Module):
    """
    A brain-inspired neural network that integrates all components into a cohesive system.
    
    This model is designed to mimic certain aspects of brain functionality:
    - The controller acts as a central processing unit (like prefrontal cortex)
    - The neuromodulator system regulates neural activity based on rewards (like dopaminergic systems)
    - The input processor handles sensory information (like sensory cortices)
    - The LLM interface provides higher-level cognitive capabilities (like language areas)
    
    The system preserves the biologically-inspired nature with the controller acting as
    a central module that controls the neuromodulator based on preprocessed inputs and reward signals.
    """
    
    def __init__(self, input_size, hidden_size, output_size, 
                 persistent_memory_size=128, num_layers=2, dropout=0.1,
                 dopamine_scale=1.0, serotonin_scale=0.8, 
                 norepinephrine_scale=0.6, acetylcholine_scale=0.7,
                 reward_decay=0.95, llm_provider="openai", llm_api_key=None):
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
            llm_provider (str): LLM provider to use ('openai', 'huggingface', 'anthropic')
            llm_api_key (str, optional): API key for the LLM provider
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
        self.reward_predictor = RewardPredictor(
            state_size=hidden_size,
            action_size=output_size,
            hidden_size=hidden_size
        )
        
        # Output Layer - generates final output
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # LLM Interface - provides higher-level cognitive capabilities
        self.llm_interface = LLMInterface(
            provider=llm_provider,
            api_key=llm_api_key,
            embedding_dim=hidden_size
        )
        
        # Internal state tracking
        self.hidden_states = None
        self.persistent_memories = None
        self.neurotransmitter_levels = None
        self.input_metadata = None

    def setup_model(config, input_shape):
        """Set up the model with financial-specific parameters."""
        # Get input dimensions from actual preprocessed data
        input_size = input_shape[2]  # Number of features per time step
        hidden_size = config['controller']['hidden_size']
        output_size = 1  # Single value prediction (target price)
        
        # Critical fix: Configure persistent memory size based on financial data patterns
        persistent_memory_size = config['controller']['persistent_memory_size']
        
        # Neuromodulator tuning specifically for financial data
        # Dopamine is crucial for reward prediction in market movements
        dopamine_scale = config['neuromodulator']['dopamine_scale'] 
        # Serotonin regulates risk assessment behavior
        serotonin_scale = config['neuromodulator']['serotonin_scale']
        # Norepinephrine increases attention to market volatility
        norepinephrine_scale = config['neuromodulator']['norepinephrine_scale'] 
        # Acetylcholine enhances memory formation for market patterns
        acetylcholine_scale = config['neuromodulator']['acetylcholine_scale']
        
        reward_decay = config['neuromodulator']['reward_decay']
        
        # Fix: Set num_layers and dropout from config
        num_layers = config['controller']['num_layers']
        dropout = config['controller']['dropout']
        
        # Create model with appropriate parameters
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
            reward_decay=reward_decay,
            # Fix: Remove LLM parameters
        )
        
        return model
    
    def forward(self, x, hidden=None, persistent_memory=None, external_reward=None):
        """
        Forward pass of the Brain-Inspired Neural Network.
        
        The information flow follows a biologically-inspired pattern:
        1. Input preprocessing (sensory processing)
        2. Controller processing (central neural processing)
        3. Reward prediction (basal ganglia function)
        4. Neuromodulation (neurotransmitter effects)
        5. Output generation (motor/response output)
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
            hidden (torch.Tensor, optional): Initial hidden state tensor
            persistent_memory (torch.Tensor, optional): Initial persistent memory tensor
            external_reward (torch.Tensor, optional): External reward signal
            
        Returns:
            tuple: (outputs, predicted_rewards)
        """
        batch_size, seq_length, _ = x.size()
        device = x.device
        
        # Initialize states if not provided
        if hidden is None or persistent_memory is None:
            hidden, persistent_memory = self.controller.init_hidden(batch_size, device)
        
        # Preprocess input through the input processor
        processed_x, input_metadata = self.input_processor(x)
        self.input_metadata = input_metadata
        
        # Process through controller
        controller_outputs, hidden_states, persistent_memories = self.controller(
            processed_x, hidden, persistent_memory
        )
        
        # Store states
        self.hidden_states = hidden_states
        self.persistent_memories = persistent_memories
        
        # Process each time step with neuromodulation
        outputs = []
        predicted_rewards = []
        
        for t in range(seq_length):
            # Get controller output for current time step
            controller_output_t = controller_outputs[:, t, :]
            
            # Generate action (preliminary output)
            action_t = self.output_layer(controller_output_t)
            
            # Predict reward if external reward not provided
            if external_reward is None:
                reward_t = self.reward_predictor(controller_output_t, action_t)
            else:
                reward_t = external_reward[:, t].unsqueeze(1) if external_reward.dim() > 1 else external_reward.unsqueeze(1)
            
            # Apply neuromodulation
            modulated_output_t, neurotransmitter_levels_t = self.neuromodulator(
                processed_x[:, t, :], controller_output_t, reward_t
            )
            
            # Generate final output
            final_output_t = self.output_layer(modulated_output_t)
            
            # Store outputs and rewards
            outputs.append(final_output_t)
            predicted_rewards.append(reward_t)
            
            # Store neurotransmitter levels for the last time step
            if t == seq_length - 1:
                self.neurotransmitter_levels = neurotransmitter_levels_t
        
        # Stack outputs and rewards along sequence dimension
        outputs = torch.stack(outputs, dim=1)
        predicted_rewards = torch.stack(predicted_rewards, dim=1)
        
        return outputs, predicted_rewards

    def preprocess_data(stock_data, config):
        """
        Preprocess stock data for training the brain-inspired neural network.
        
        Args:
            stock_data (pd.DataFrame): Raw stock data from yfinance
            config (dict): Configuration parameters
            
        Returns:
            dict: Dictionary containing DataLoaders and metadata
        """
        # Calculate technical indicators if enabled
        if config['data'].get('use_technical_indicators', True):
            df = stock_data.copy()
            
            # Moving averages
            window_sizes = config['data'].get('ma_windows', [5, 10, 20, 50])
            for window in window_sizes:
                df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            
            # Relative Strength Index (RSI)
            rsi_window = config['data'].get('rsi_window', 14)
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=rsi_window).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=rsi_window).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands - Fixed implementation
            bb_window = config['data'].get('bb_window', 20)
            std_dev = config['data'].get('bb_std_dev', 2)
            
            # Calculate middle band
            df['BB_Middle'] = df['Close'].rolling(window=bb_window).mean()

            # Calculate standard deviation as a separate Series
            rolling_std = df['Close'].rolling(window=bb_window).std()

            # Calculate upper and lower bands using explicit Series operations
            # Force the calculation to produce a single-column Series
            upper_band = df['BB_Middle'] + (std_dev * rolling_std)
            lower_band = df['BB_Middle'] - (std_dev * rolling_std)

            # Now assign these Series to the DataFrame columns
            # Even more explicit approach if needed
            df['BB_Upper'] = pd.Series(df['BB_Middle'].values + (std_dev * rolling_std.values), index=df.index)
            df['BB_Lower'] = pd.Series(df['BB_Middle'].values - (std_dev * rolling_std.values), index=df.index)
            
            # MACD
            fast = config['data'].get('macd_fast', 12)
            slow = config['data'].get('macd_slow', 26)
            signal = config['data'].get('macd_signal', 9)
            df['MACD'] = df['Close'].ewm(span=fast, adjust=False).mean() - df['Close'].ewm(span=slow, adjust=False).mean()
            df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # Momentum indicators
            df['Momentum_5'] = df['Close'].pct_change(5)
            df['Momentum_10'] = df['Close'].pct_change(10)
            
            # Volatility indicator
            df['Volatility'] = df['Close'].rolling(window=bb_window).std() / df['Close'].rolling(window=bb_window).mean()
            
            # Price rate of change
            df['ROC_5'] = df['Close'].pct_change(5) * 100
            df['ROC_10'] = df['Close'].pct_change(10) * 100
            
            # Volume indicators
            df['Volume_ROC'] = df['Volume'].pct_change() * 100
            df['Volume_MA_Ratio'] = df['Volume'] / df['Volume'].rolling(window=10).mean()
            
            # Drop NaN values resulting from calculations
            enhanced_data = df.dropna()
        else:
            enhanced_data = stock_data.copy()
        
        # Extract preprocessing parameters
        sequence_length = config['data']['sequence_length']
        target_column = config['data']['target_column']
        feature_columns = config['data']['feature_columns']
        prediction_horizon = config['data']['prediction_horizon']
        train_ratio = config['data']['train_ratio']
        val_ratio = config['data']['val_ratio']
        
        # If feature_columns is 'all', use all available columns
        if feature_columns == 'all' or feature_columns[0] == 'all':
            feature_columns = enhanced_data.columns.tolist()
        
        print(f"Using {len(feature_columns)} features: {feature_columns}")
        
        # Extract features
        features = enhanced_data[feature_columns].values
        
        # Normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        features_scaled = scaler.fit_transform(features)
        
        # Find target column index
        target_idx = feature_columns.index(target_column)
        
        # Create sequences and targets
        X, y, rewards = [], [], []
        for i in range(len(features_scaled) - sequence_length - prediction_horizon + 1):
            # Input sequence
            X.append(features_scaled[i:i+sequence_length])
            
            # Target: price at prediction_horizon days ahead
            current_price = features_scaled[i+sequence_length-1, target_idx]
            future_price = features_scaled[i+sequence_length+prediction_horizon-1, target_idx]
            y.append(future_price)
            
            # Generate reward signal
            price_change = future_price - current_price
            price_change_pct = price_change / (current_price + 1e-8)
            reward_signal = np.sign(price_change_pct) * np.sqrt(np.abs(price_change_pct))
            
            # Apply volatility adjustment
            if i >= 10:
                recent_prices = features_scaled[i+sequence_length-10:i+sequence_length, target_idx]
                volatility = np.std(recent_prices) / (np.mean(recent_prices) + 1e-8)
                reward_signal = reward_signal * (1 + volatility)
            
            rewards.append(reward_signal)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        rewards = np.array(rewards).reshape(-1, 1)
        
        # Split data into train, validation, and test sets
        total_samples = len(X)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
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
        
        # Create PyTorch datasets
        from torch.utils.data import TensorDataset, DataLoader
        
        train_dataset = TensorDataset(X_train, y_train, rewards_train)
        val_dataset = TensorDataset(X_val, y_val, rewards_val)
        test_dataset = TensorDataset(X_test, y_test, rewards_test)
        
        # Create DataLoaders
        batch_size = config['training']['batch_size']
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=config.get('num_workers', 0),
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=config.get('num_workers', 0),
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=config.get('num_workers', 0),
            pin_memory=True if torch.cuda.is_available() else False
        )
        
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
            }
        }
        
        return data_info
