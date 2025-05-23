import torch
try:
    import cupy as cp
    USING_CUPY = True
    print("Using CuPy for financial data processing")
except ImportError:
    import numpy as cp
    USING_CUPY = False
    print("Using NumPy for financial data (consider installing CuPy for better performance)")
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class AugmentedTimeSeriesDataset(Dataset):
    """
    Dataset for time series prediction with built-in data augmentation.
    
    This dataset applies multiple augmentation techniques to financial time series
    to provide robust training examples while preserving temporal characteristics.
    """
    def __init__(self, base_dataset, augment_prob=0.5, augment_strength=0.05, 
                target_column=None, sequence_first=True):
        """
        Initialize the dataset.
        
        Args:
            base_dataset: Original dataset to augment
            augment_prob: Probability of applying augmentation to a sample
            augment_strength: Strength of augmentation (0-1, higher = stronger)
            target_column: Column to preserve during augmentation
            sequence_first: Whether data has sequence as first dimension
        """
        self.base_dataset = base_dataset
        self.augment_prob = augment_prob
        self.augment_strength = augment_strength
        self.target_column = target_column
        self.sequence_first = sequence_first
        
        # Set RNG seed for reproducibility while allowing randomness
        self.rng = cp.random.RandomState(42)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        
        # Apply augmentation with probability
        if self.rng.random() < self.augment_prob and self.augment_strength > 0:
            # Ensure x is in the right format for augmentation
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32)
                
            # Apply augmentation
            x = self._augment_timeseries(x)
        
        return x, y
    
    def _augment_timeseries(self, x):
        """Apply multiple augmentation techniques to time series data."""
        # Clone to avoid modifying original
        augmented = x.clone()
        
        # Determine data shape: [batch, seq, features] or [seq, features]
        is_batched = augmented.dim() > 2
        
        # Reshape for consistent processing if needed
        if is_batched:
            batch_size, seq_len, features = augmented.shape
        elif self.sequence_first:
            seq_len, features = augmented.shape
            augmented = augmented.unsqueeze(0)  # Add batch dimension
        else:
            features, seq_len = augmented.shape
            augmented = augmented.t().unsqueeze(0)  # Transpose and add batch
        
        # Apply a random subset of augmentations
        aug_types = self.rng.choice([
            'noise', 'scale', 'magnitude_warp', 'time_warp', 'window_slice'
        ], size=2, replace=False)
        
        for aug_type in aug_types:
            if aug_type == 'noise':
                # Add random noise
                noise = torch.randn_like(augmented) * self.augment_strength
                augmented = augmented + noise
                
            elif aug_type == 'scale':
                # Random scaling of feature magnitudes
                scale_factor = 1.0 + (torch.rand(1) * 2 - 1) * self.augment_strength
                augmented = augmented * scale_factor
                
            elif aug_type == 'magnitude_warp':
                # Smooth, sequence-correlated distortion of feature magnitudes
                smooth_noise = self._generate_smooth_noise(augmented.shape)
                warp = smooth_noise * self.augment_strength * 2
                augmented = augmented * (1 + warp)
                
            elif aug_type == 'time_warp':
                # Subtle time warping (slower/faster segments)
                if seq_len > 10:
                    # Number of warping points
                    n_points = max(2, int(seq_len * self.augment_strength))
                    
                    # Generate warping points and create displacement field
                    orig_steps = cp.arange(seq_len)
                    warp_steps = cp.zeros(seq_len)
                    
                    # Generate random warp points
                    warp_points = self.rng.choice(seq_len, size=n_points)
                    warp_offsets = self.rng.uniform(-self.augment_strength * 3, 
                                                 self.augment_strength * 3, 
                                                 size=n_points)
                    
                    # Create displacement field by interpolation
                    for p, o in zip(warp_points, warp_offsets):
                        warp_steps[p] = o
                    
                    # Smooth the displacement field
                    from scipy.ndimage import gaussian_filter1d
                    warp_steps = gaussian_filter1d(warp_steps, sigma=3)
                    
                    # Apply displacement (resample)
                    new_steps = orig_steps + warp_steps
                    
                    # Ensure steps are within bounds
                    new_steps = cp.clip(new_steps, 0, seq_len - 1)
                    
                    # Interpolate data at new time steps
                    for b in range(augmented.shape[0]):
                        for f in range(features):
                            series = augmented[b, :, f].numpy()
                            augmented[b, :, f] = torch.tensor(
                                cp.interp(orig_steps, new_steps, series),
                                dtype=augmented.dtype
                            )
                            
            elif aug_type == 'window_slice':
                # Randomly slice a window and stretch it to original size
                if seq_len > 10:
                    # Determine window size (percent of original)
                    window_size = int(seq_len * (1 - self.augment_strength/2))
                    window_size = max(window_size, seq_len // 2)  # Minimum 50% of original
                    
                    # Random starting point
                    start = self.rng.randint(0, seq_len - window_size + 1)
                    end = start + window_size
                    
                    # Extract window
                    window = augmented[:, start:end, :].clone()
                    
                    # Resize window to original sequence length using interpolation
                    for b in range(augmented.shape[0]):
                        for f in range(features):
                            orig_series = window[b, :, f].numpy()
                            orig_steps = cp.linspace(0, 1, num=window_size)
                            new_steps = cp.linspace(0, 1, num=seq_len)
                            augmented[b, :, f] = torch.tensor(
                                cp.interp(new_steps, orig_steps, orig_series),
                                dtype=augmented.dtype
                            )
        
        # Reshape back to original format
        if not is_batched:
            if self.sequence_first:
                augmented = augmented.squeeze(0)  # Remove batch dimension
            else:
                augmented = augmented.squeeze(0).t()  # Remove batch and transpose back
        
        return augmented
    
    def _generate_smooth_noise(self, shape):
        """Generate smooth noise for time-correlated augmentation."""
        batch_size, seq_len, features = shape
        
        # Generate random frequency and phase for each feature
        freqs = torch.rand(features) * 3  # 0-3 cycles over the sequence
        phases = torch.rand(features) * 2 * cp.pi
        
        # Create time steps
        t = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(2)  # [1, seq, 1]
        t = t.repeat(batch_size, 1, features)  # [batch, seq, features]
        
        # Generate smooth sinusoidal noise
        noise = torch.sin(2 * cp.pi * freqs * t + phases)
        
        return noise


def create_augmented_train_loader(train_dataset, config):
    """
    Create a training data loader with augmentation.
    
    Args:
        train_dataset: Original training dataset
        config: Configuration dictionary
        
    Returns:
        DataLoader with augmentation applied
    """
    # Create augmented dataset wrapper
    augmented_dataset = AugmentedTimeSeriesDataset(
        train_dataset,
        augment_prob=config['data'].get('augment_prob', 0.5),
        augment_strength=config['data'].get('augment_strength', 0.05),
        target_column=config['data'].get('target_column', 'Close')
    )
    
    # Create DataLoader with augmented dataset
    train_loader = DataLoader(
        augmented_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=config['data'].get('pin_memory', False)
    )
    
    return train_loader


# Advanced preprocessing tools

def add_financial_features(df):
    """
    Add advanced financial features for better predictive power.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with added financial features
    """
    # Create copy to avoid modifying the original
    result = df.copy()
    
    # Verify required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        available_cols = list(df.columns)
        print(f"Warning: Missing required columns. Available: {available_cols}")
        # Try to find case-insensitive matches
        for req_col in required_cols:
            if req_col not in df.columns:
                matches = [col for col in df.columns if col.upper() == req_col.upper()]
                if matches:
                    # Rename matched column to expected name
                    result.rename(columns={matches[0]: req_col}, inplace=True)
                    print(f"Renamed '{matches[0]}' to '{req_col}'")
    
    # Re-check if required columns are now available
    missing_cols = [col for col in required_cols if col not in result.columns]
    if missing_cols:
        print(f"Error: Still missing required columns: {missing_cols}")
        # Skip features that require missing columns but continue with what we can
    
    # Extract data series if columns are available
    if 'Close' in result.columns:
        # --- Price-based features ---
        close = result['Close']
        
        # Log returns (daily percentage change, more normally distributed than raw price)
        result['LogReturn'] = cp.log(close / close.shift(1))
        
        # Price momentum (absolute price change over different windows)
        for window in [5, 10, 20]:
            if len(close) > window:
                result[f'Momentum_{window}'] = close - close.shift(window)
        
        # Volatility measures
        result['Volatility_5'] = result['LogReturn'].rolling(window=5).std()
        result['Volatility_20'] = result['LogReturn'].rolling(window=20).std()
        
        # Price position within recent range
        for window in [10, 20]:
            if len(close) > window:
                high_window = result['High'].rolling(window=window).max()
                low_window = result['Low'].rolling(window=window).min()
                result[f'PricePosition_{window}'] = (close - low_window) / (high_window - low_window + 1e-8)
    
    if all(col in result.columns for col in ['High', 'Low', 'Close']):
        # --- Candlestick pattern features ---
        high = result['High']
        low = result['Low']
        open_price = result['Open'] if 'Open' in result.columns else close
        
        # Body and shadow ratios
        body = abs(close - open_price)
        total_range = high - low
        upper_shadow = high - torch.maximum(close, open_price)
        lower_shadow = torch.minimum(close, open_price) - low
        
        result['BodyRatio'] = body / (total_range + 1e-8)
        result['UpperShadowRatio'] = upper_shadow / (total_range + 1e-8)
        result['LowerShadowRatio'] = lower_shadow / (total_range + 1e-8)
        
        # Doji detection (small body relative to range)
        result['IsDoji'] = (body / (total_range + 1e-8) < 0.1).astype(float)
    
    if 'Volume' in result.columns:
        # --- Volume-based features ---
        volume = result['Volume']
        
        # Volume moving averages
        result['VolumeMA_10'] = volume.rolling(window=10).mean()
        result['VolumeMA_20'] = volume.rolling(window=20).mean()
        
        # Volume rate of change
        result['VolumeROC'] = volume.pct_change(periods=5)
        
        # Price-volume relationships
        if 'Close' in result.columns:
            # On-Balance Volume (OBV)
            price_change = result['Close'].diff()
            obv_direction = torch.sign(price_change)
            result['OBV'] = (volume * obv_direction).cumsum()
            
            # Volume-Weighted Average Price (VWAP) approximation
            typical_price = (result['High'] + result['Low'] + result['Close']) / 3
            result['VWAP'] = (typical_price * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
    
    # --- Advanced technical indicators ---
    if 'Close' in result.columns:
        close = result['Close']
        
        # Commodity Channel Index (CCI)
        if all(col in result.columns for col in ['High', 'Low']):
            typical_price = (result['High'] + result['Low'] + close) / 3
            sma_tp = typical_price.rolling(window=20).mean()
            mad = typical_price.rolling(window=20).apply(lambda x: abs(x - x.mean()).mean())
            result['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
        
        # Williams %R
        if all(col in result.columns for col in ['High', 'Low']):
            highest_high = result['High'].rolling(window=14).max()
            lowest_low = result['Low'].rolling(window=14).min()
            result['WilliamsR'] = (highest_high - close) / (highest_high - lowest_low + 1e-8) * -100
        
        # Money Flow Index (MFI) - RSI with volume
        if 'Volume' in result.columns and all(col in result.columns for col in ['High', 'Low']):
            typical_price = (result['High'] + result['Low'] + close) / 3
            money_flow = typical_price * result['Volume']
            
            # Separate positive and negative money flows
            price_change = typical_price.diff()
            positive_mf = money_flow.where(price_change > 0, 0).rolling(window=14).sum()
            negative_mf = money_flow.where(price_change < 0, 0).rolling(window=14).sum()
            
            mfi_ratio = positive_mf / (negative_mf + 1e-8)
            result['MFI'] = 100 - (100 / (1 + mfi_ratio))
    
    # --- Market structure features ---
    if 'Close' in result.columns:
        close = result['Close']
        
        # Support and resistance levels (approximate)
        for window in [20, 50]:
            if len(close) > window:
                # Local maxima and minima
                rolling_max = close.rolling(window=window, center=True).max()
                rolling_min = close.rolling(window=window, center=True).min()
                
                # Distance to support/resistance
                result[f'DistanceToResistance_{window}'] = (rolling_max - close) / close
                result[f'DistanceToSupport_{window}'] = (close - rolling_min) / close
        
        # Trend strength
        for window in [10, 20, 50]:
            if len(close) > window:
                # Linear regression slope as trend indicator
                x = torch.arange(window, dtype=torch.float32)
                
                def calculate_slope(prices):
                    if len(prices) < window:
                        return 0.0
                    y = torch.tensor(prices.values, dtype=torch.float32)
                    
                    # Calculate slope using least squares
                    n = len(y)
                    sum_x = x.sum()
                    sum_y = y.sum()
                    sum_xy = (x * y).sum()
                    sum_x2 = (x * x).sum()
                    
                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x + 1e-8)
                    return slope.item()
                
                result[f'TrendSlope_{window}'] = close.rolling(window=window).apply(calculate_slope)
    
    # Fill NaN values with appropriate methods
    for col in result.columns:
        if col not in df.columns:  # Only fill new columns
            # Determine fill method based on column type
            col_str = str(col)
            if any(keyword in col_str.lower() for keyword in ['ratio', 'position', 'isdoji']):
                # For ratios and binary indicators, fill with neutral values
                result[col] = result[col].fillna(0.5 if 'ratio' in col_str.lower() else 0.0)
            elif any(keyword in col_str.lower() for keyword in ['return', 'roc', 'momentum']):
                # For returns and changes, fill with 0 (no change)
                result[col] = result[col].fillna(0.0)
            elif any(keyword in col_str.lower() for keyword in ['volatility', 'atr']):
                # For volatility measures, fill with small positive value
                result[col] = result[col].fillna(0.01)
            else:
                # For other indicators, forward fill then backward fill
                result[col] = result[col].fillna(method='ffill').fillna(method='bfill')
    
    return result


def create_cross_validation_splits(df, n_splits=5, test_size=0.2):
    """
    Create time-aware cross-validation splits for financial data.
    
    Args:
        df (pd.DataFrame): Time series data
        n_splits (int): Number of CV splits
        test_size (float): Proportion of data for testing in each split
        
    Returns:
        list: List of (train_indices, test_indices) tuples
    """
    total_len = len(df)
    test_len = int(total_len * test_size)
    
    # For time series, we use forward-chaining validation
    # Each fold uses progressively more historical data for training
    splits = []
    
    for i in range(n_splits):
        # Calculate test period for this fold
        test_end = total_len - (n_splits - i - 1) * (test_len // n_splits)
        test_start = test_end - test_len
        
        # Ensure we have enough training data
        train_start = max(0, test_start - (total_len - test_len))
        train_end = test_start
        
        if train_end > train_start and test_end > test_start:
            train_indices = list(range(train_start, train_end))
            test_indices = list(range(test_start, test_end))
            splits.append((train_indices, test_indices))
    
    return splits


def detect_market_regime(df, window=50):
    """
    Detect market regime (bull/bear/sideways) for regime-aware training.
    
    Args:
        df (pd.DataFrame): Financial data with Close prices
        window (int): Window for regime detection
        
    Returns:
        pd.Series: Market regime labels (0=bear, 1=sideways, 2=bull)
    """
    if 'Close' not in df.columns:
        return pd.Series([1] * len(df), index=df.index)  # Default to sideways
    
    close = df['Close']
    
    # Calculate trend strength using linear regression slope
    def trend_strength(prices):
        if len(prices) < window:
            return 0.0
        
        x = cp.arange(len(prices))
        coeffs = cp.polyfit(x, prices, 1)
        slope = coeffs[0]
        
        # Normalize by price level
        avg_price = cp.mean(prices)
        normalized_slope = slope / avg_price if avg_price > 0 else 0
        
        return normalized_slope
    
    # Calculate rolling trend strength
    trend = close.rolling(window=window).apply(trend_strength)
    
    # Calculate volatility for regime detection
    returns = close.pct_change()
    volatility = returns.rolling(window=window).std()
    
    # Define regime thresholds
    trend_threshold = 0.001  # 0.1% per period
    vol_threshold = volatility.quantile(0.7)  # High volatility threshold
    
    # Classify regimes
    regimes = []
    for i in range(len(df)):
        if pd.isna(trend.iloc[i]) or pd.isna(volatility.iloc[i]):
            regimes.append(1)  # Default to sideways
        elif trend.iloc[i] > trend_threshold:
            regimes.append(2)  # Bull market
        elif trend.iloc[i] < -trend_threshold:
            regimes.append(0)  # Bear market
        else:
            regimes.append(1)  # Sideways market
    
    return pd.Series(regimes, index=df.index)


class RegimeAwareDataset(Dataset):
    """
    Dataset that adjusts augmentation based on market regime.
    """
    def __init__(self, base_dataset, regime_labels, regime_augment_factors=None):
        """
        Initialize regime-aware dataset.
        
        Args:
            base_dataset: Base dataset
            regime_labels: Market regime labels for each sample
            regime_augment_factors: Dict of augmentation factors per regime
        """
        self.base_dataset = base_dataset
        self.regime_labels = regime_labels
        
        # Default augmentation factors per regime
        self.regime_factors = regime_augment_factors or {
            0: 0.08,  # Bear market - more augmentation (more volatile)
            1: 0.05,  # Sideways - moderate augmentation
            2: 0.03   # Bull market - less augmentation (more stable)
        }
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        
        # Get regime for this sample
        regime = self.regime_labels[idx] if idx < len(self.regime_labels) else 1
        
        # Apply regime-specific augmentation
        if regime in self.regime_factors:
            augment_strength = self.regime_factors[regime]
            
            # Apply augmentation with regime-specific strength
            if torch.rand(1) < 0.6:  # 60% chance of augmentation
                augmented_dataset = AugmentedTimeSeriesDataset(
                    base_dataset=type('DummyDataset', (), {
                        '__getitem__': lambda self, i: (x, y),
                        '__len__': lambda self: 1
                    })(),
                    augment_prob=1.0,
                    augment_strength=augment_strength
                )
                x, y = augmented_dataset[0]
        
        return x, y


def create_ensemble_datasets(base_dataset, n_ensembles=3, augment_variations=None):
    """
    Create multiple variations of the dataset for ensemble training.
    
    Args:
        base_dataset: Original dataset
        n_ensembles: Number of ensemble variations
        augment_variations: List of augmentation parameters for each ensemble
        
    Returns:
        list: List of augmented datasets for ensemble training
    """
    if augment_variations is None:
        # Default variations with different augmentation strengths
        augment_variations = [
            {'augment_prob': 0.3, 'augment_strength': 0.02},  # Conservative
            {'augment_prob': 0.5, 'augment_strength': 0.05},  # Moderate
            {'augment_prob': 0.7, 'augment_strength': 0.08},  # Aggressive
        ]
    
    ensemble_datasets = []
    
    for i in range(n_ensembles):
        # Use cycling through variations if we have more ensembles than variations
        variation = augment_variations[i % len(augment_variations)]
        
        augmented = AugmentedTimeSeriesDataset(
            base_dataset,
            augment_prob=variation['augment_prob'],
            augment_strength=variation['augment_strength']
        )
        
        ensemble_datasets.append(augmented)
    
    return ensemble_datasets


def create_balanced_batches(dataset, regime_labels, batch_size=32):
    """
    Create batches that are balanced across market regimes.
    
    Args:
        dataset: Dataset to batch
        regime_labels: Market regime labels
        batch_size: Size of each batch
        
    Returns:
        DataLoader with regime-balanced batches
    """
    from torch.utils.data.sampler import WeightedRandomSampler
    
    # Calculate regime distribution
    regime_counts = pd.Series(regime_labels).value_counts()
    
    # Calculate weights to balance regimes
    weights = []
    for regime in regime_labels:
        weight = 1.0 / regime_counts[regime] if regime in regime_counts else 1.0
        weights.append(weight)
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    # Create DataLoader with balanced sampling
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=False
    )
    
    return dataloader


# Integration function to update the main data loading
def load_financial_data_with_enhancements(config):
    """
    Enhanced version of load_financial_data with advanced preprocessing and augmentation.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, data_info)
    """
    # Load basic financial data (using existing function)
    from src.utils.financial_data_utils import load_financial_data
    
    try:
        train_loader, val_loader, test_loader, data_info = load_financial_data(config)
    except Exception as e:
        print(f"Error in basic data loading: {e}")
        raise
    
    # Extract the original datasets for enhancement
    train_dataset = data_info['train_dataset']
    val_dataset = data_info['val_dataset']
    test_dataset = data_info['test_dataset']
    
    # Apply enhanced preprocessing if enabled
    if config['data'].get('use_advanced_features', True):
        print("Applying advanced financial feature engineering...")
        
        # Load raw data again for feature engineering
        import yfinance as yf
        ticker = config['data']['ticker_symbol']
        start_date = config['data'].get('start_date', '2015-01-01')
        end_date = config['data'].get('end_date', datetime.now().strftime('%Y-%m-%d'))
        
        try:
            raw_df = yf.download(ticker, start=start_date, end=end_date)
            
            # Add advanced features
            enhanced_df = add_financial_features(raw_df)
            
            # Recreate datasets with enhanced features
            # Note: This is a simplified version - in practice, you'd want to
            # rebuild the TimeSeriesDataset with the new features
            print(f"Enhanced features: {len(enhanced_df.columns)} total columns")
            
        except Exception as e:
            print(f"Warning: Could not apply advanced features: {e}")
    
    # Apply regime-aware augmentation if enabled
    if config['data'].get('regime_aware_training', False):
        print("Applying market regime detection...")
        
        try:
            # This would require access to the original DataFrame
            # For now, we'll create a placeholder
            regime_labels = [1] * len(train_dataset)  # Default to sideways market
            
            # Create regime-aware training dataset
            regime_train_dataset = RegimeAwareDataset(
                train_dataset,
                regime_labels
            )
            
            # Create new DataLoader
            train_loader = DataLoader(
                regime_train_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=True,
                num_workers=config['data'].get('num_workers', 0),
                pin_memory=config['data'].get('pin_memory', False)
            )
            
            print("Applied regime-aware training augmentation")
            
        except Exception as e:
            print(f"Warning: Could not apply regime-aware training: {e}")
    
    # Create ensemble datasets if requested
    elif config['data'].get('ensemble_augmentation', False):
        print("Creating ensemble training datasets...")
        
        n_ensembles = config['data'].get('n_ensembles', 3)
        ensemble_datasets = create_ensemble_datasets(train_dataset, n_ensembles)
        
        # For now, just use the first ensemble dataset
        # In practice, you might want to train multiple models
        train_loader = DataLoader(
            ensemble_datasets[0],
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['data'].get('num_workers', 0),
            pin_memory=config['data'].get('pin_memory', False)
        )
        
        # Store ensemble datasets in data_info for potential use
        data_info['ensemble_datasets'] = ensemble_datasets
        
        print(f"Created {n_ensembles} ensemble training datasets")
    
    # Apply standard augmentation if not using regime-aware or ensemble
    else:
        # Create augmented training dataset
        augmented_train_dataset = AugmentedTimeSeriesDataset(
            train_dataset,
            augment_prob=config['data'].get('augment_prob', 0.5),
            augment_strength=config['data'].get('augment_strength', 0.05)
        )
        
        # Create new training DataLoader with augmentation
        train_loader = DataLoader(
            augmented_train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['data'].get('num_workers', 0),
            pin_memory=config['data'].get('pin_memory', False)
        )
        
        print("Applied standard data augmentation to training set")
    
    return train_loader, val_loader, test_loader, data_info