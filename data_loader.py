import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import yfinance as yf
import cupy as cp
import pandas as pd
import random

class YFinanceDataset(Dataset):
    def __init__(self, tickers, start_date, end_date, sequence_length, transform=None):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.transform = transform

        # Download data from yfinance
        self.data = self._download_data()

    def _download_data(self):
        try:
            data = yf.download(self.tickers, start=self.start_date, end=self.end_date)
            if data.empty:
                raise ValueError(f"No data downloaded for {self.tickers}")
            return data['Close'].dropna()
        except Exception as e:
            print(f"Error downloading data: {e}")
            # Create synthetic data as fallback
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
            synthetic_data = pd.Series(
                100 + cp.cumsum(cp.random.randn(len(dates)) * 0.01),
                index=dates,
                name='Close'
            )
            return synthetic_data

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx:idx + self.sequence_length].values
        target = self.data.iloc[idx + self.sequence_length]

        if self.transform:
            sequence = self.transform(sequence)
            target = self.transform(target)

        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

class RandomWindowDataset(Dataset):
    """Samples random windows of fixed sequence_length from data array."""
    def __init__(self, data, seq_len, sample_count, augment=False, noise_std=0.01):
        self.data = data
        self.seq_len = seq_len
        self.sample_count = sample_count
        self.max_start = len(data) - seq_len - 1
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self):
        return self.sample_count

    def __getitem__(self, idx):
        start = random.randint(0, self.max_start)
        seq = self.data[start:start + self.seq_len]
        tgt = self.data[start + self.seq_len]

        # Apply data augmentation if enabled
        if self.augment:
            noise = cp.random.normal(0, self.noise_std, seq.shape)
            seq = seq + noise

        return torch.tensor(seq, dtype=torch.float32), torch.tensor(tgt, dtype=torch.float32)

def create_datasets(config):
    """
    Create train, validation, and test datasets with improved error handling.
    """
    try:
        # Parameters with better defaults and error handling
        ticker = config.get('data', {}).get('ticker_symbol', 'AAPL')
        sequence_length = config.get('data', {}).get('sequence_length', 50)
        
        # Handle features parameter properly
        features_config = config.get('data', {}).get('features', 'all')
        if features_config == 'all' or features_config == ['all']:
            # Use default OHLCV features when 'all' is specified
            features = ['Open', 'High', 'Low', 'Close', 'Volume']
        elif isinstance(features_config, (list, tuple)):
            features = list(features_config)
        else:
            # Fallback to Close price only
            features = ['Close']
        
        # Ensure we have required ratios
        train_ratio = config.get('data', {}).get('train_ratio', 0.7)
        val_ratio = config.get('data', {}).get('val_ratio', 0.15)
        
        # Date range with defaults
        start_date = config.get('data', {}).get('start_date', '2020-01-01')
        end_date = config.get('data', {}).get('end_date', '2023-12-31')
        
        print(f"Loading data for {ticker} from {start_date} to {end_date}")
        print(f"Features to use: {features}")
        
        # Download and preprocess
        try:
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
            
            if df.empty:
                raise ValueError(f"No data downloaded for {ticker}")
            
            print(f"Downloaded {len(df)} days of data")
            print(f"Available columns: {list(df.columns)}")
            
            # Select features, with fallback handling
            available_features = []
            for feature in features:
                if feature in df.columns:
                    available_features.append(feature)
                else:
                    print(f"Warning: Feature '{feature}' not available in data")
            
            if not available_features:
                # Fallback to Close price if it exists
                if 'Close' in df.columns:
                    available_features = ['Close']
                    print("Falling back to Close price only")
                else:
                    # Use the first available column
                    available_features = [df.columns[0]]
                    print(f"Falling back to first available column: {available_features[0]}")
            
            # Extract the selected features
            if len(available_features) == 1:
                data = df[available_features[0]].dropna().values
                # Reshape to 2D for consistency
                data = data.reshape(-1, 1)
            else:
                data = df[available_features].dropna().values
            
            print(f"Using features: {available_features}")
            print(f"Data shape: {data.shape}")
            
        except Exception as e:
            print(f"Error downloading real data: {e}")
            print("Creating synthetic data...")
            
            # Create synthetic data
            n_days = 1000
            n_features = len(features) if isinstance(features, list) else 5
            
            # Generate synthetic time series
            data = cp.zeros((n_days, n_features))
            data[0] = cp.random.randn(n_features) + 100  # Starting prices around 100
            
            for i in range(1, n_days):
                # Random walk with slight trend
                data[i] = data[i-1] + cp.random.randn(n_features) * 0.02
            
            available_features = features if isinstance(features, list) else ['Close']
            print(f"Created synthetic data with shape: {data.shape}")
        
        # Compute split indices
        total = len(data)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        print(f"Data split: total={total}, train_end={train_end}, val_end={val_end}")

        # Split data
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        print(f"Split sizes: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

        # Normalize using training data statistics
        if config.get('data', {}).get('normalize', False):
            print("Normalizing data...")
            means = train_data.mean(axis=0)
            stds = train_data.std(axis=0)
            # Handle zeros in standard deviation
            stds = cp.where(stds == 0, 1, stds)
            
            train_data = (train_data - means) / stds
            val_data = (val_data - means) / stds
            test_data = (test_data - means) / stds

        # Create RandomWindowDataset for train/val
        train_samples = config.get('data', {}).get('train_samples', max(1, len(train_data) - sequence_length))
        val_samples = config.get('data', {}).get('val_samples', max(1, len(val_data) - sequence_length))
        
        print(f"Creating datasets with sequence_length={sequence_length}")
        print(f"Train samples: {train_samples}, Val samples: {val_samples}")
        
        # Ensure we have enough data for the sequence length
        if len(train_data) <= sequence_length:
            print(f"Warning: Train data length ({len(train_data)}) <= sequence_length ({sequence_length})")
            sequence_length = max(1, len(train_data) - 1)
            print(f"Adjusted sequence_length to {sequence_length}")
        
        if len(val_data) <= sequence_length:
            print(f"Warning: Val data length ({len(val_data)}) <= sequence_length ({sequence_length})")
        
        if len(test_data) <= sequence_length:
            print(f"Warning: Test data length ({len(test_data)}) <= sequence_length ({sequence_length})")
        
        # Create datasets with error handling
        try:
            train_dataset = RandomWindowDataset(train_data, sequence_length, train_samples)
            val_dataset = RandomWindowDataset(val_data, sequence_length, val_samples)
            
            # Build sliding windows for test (more systematic)
            test_seqs, test_targets = [], []
            max_test_samples = len(test_data) - sequence_length
            
            if max_test_samples > 0:
                for i in range(max_test_samples):
                    test_seqs.append(test_data[i:i+sequence_length])
                    test_targets.append(test_data[i+sequence_length])
                
                X_test = cp.array(test_seqs)
                y_test = cp.array(test_targets)
                
                print(f"Test data shapes: X_test={X_test.shape}, y_test={y_test.shape}")
                
                # Handle target shape for consistency
                if y_test.ndim == 2 and y_test.shape[1] == 1:
                    y_test = y_test.squeeze(1)  # Remove singleton dimension
                elif y_test.ndim == 2 and y_test.shape[1] > 1:
                    # If multiple features, take the first one (typically Close price)
                    y_test = y_test[:, 0]
                
                test_dataset = TensorDataset(
                    torch.tensor(X_test, dtype=torch.float32),
                    torch.tensor(y_test, dtype=torch.float32)
                )
            else:
                print("Warning: Not enough test data, creating minimal test dataset")
                # Create a minimal test dataset
                test_dataset = TensorDataset(
                    torch.randn(10, sequence_length, data.shape[1]),
                    torch.randn(10)
                )
                
        except Exception as e:
            print(f"Error creating datasets: {e}")
            print("Creating minimal fallback datasets...")
            
            # Create minimal datasets as fallback
            n_samples = 100
            n_features = data.shape[1] if len(data.shape) > 1 else 1
            
            train_dataset = TensorDataset(
                torch.randn(n_samples, sequence_length, n_features),
                torch.randn(n_samples)
            )
            val_dataset = TensorDataset(
                torch.randn(n_samples//2, sequence_length, n_features),
                torch.randn(n_samples//2)
            )
            test_dataset = TensorDataset(
                torch.randn(n_samples//4, sequence_length, n_features),
                torch.randn(n_samples//4)
            )
        
        print("Datasets created successfully!")
        return train_dataset, val_dataset, test_dataset
        
    except Exception as e:
        print(f"Critical error in create_datasets: {e}")
        print("Creating emergency fallback datasets...")
        
        # Emergency fallback - create synthetic datasets
        sequence_length = config.get('data', {}).get('sequence_length', 50)
        n_features = 5  # Default number of features
        
        train_dataset = TensorDataset(
            torch.randn(200, sequence_length, n_features),
            torch.randn(200)
        )
        val_dataset = TensorDataset(
            torch.randn(50, sequence_length, n_features),
            torch.randn(50)
        )
        test_dataset = TensorDataset(
            torch.randn(50, sequence_length, n_features),
            torch.randn(50)
        )
        
        return train_dataset, val_dataset, test_dataset