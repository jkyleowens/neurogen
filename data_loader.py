import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import yfinance as yf
import numpy as np
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
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date)
        return data['Close'].dropna()

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
    def __init__(self, data, seq_len, sample_count):
        self.data = data
        self.seq_len = seq_len
        self.sample_count = sample_count
        self.max_start = len(data) - seq_len - 1

    def __len__(self):
        return self.sample_count

    def __getitem__(self, idx):
        start = random.randint(0, self.max_start)
        seq = self.data[start:start + self.seq_len]
        tgt = self.data[start + self.seq_len]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(tgt, dtype=torch.float32)

def create_datasets(config):
    """
    Create train, validation, and test datasets.
    """
    # Parameters
    ticker = config['data']['ticker_symbol']
    sequence_length = config['data']['sequence_length']
    features = config['data'].get('features', ['Close'])
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    # Download and preprocess
    df = yf.download(ticker, auto_adjust=True)[features].dropna()
    if config['data'].get('normalize', False):
        means = df.mean()
        stds = df.std().replace(0, 1)
        df = (df - means) / stds
    data = df.values
    # Compute split indices
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    # Create RandomWindowDataset for train/val
    train_samples = config['data'].get('train_samples', max(1, len(train_data) - sequence_length))
    val_samples = config['data'].get('val_samples', max(1, len(val_data) - sequence_length))
    train_dataset = RandomWindowDataset(train_data, sequence_length, train_samples)
    val_dataset = RandomWindowDataset(val_data, sequence_length, val_samples)
    # Build sliding windows for test
    seqs, targets = [], []
    for i in range(len(test_data) - sequence_length):
        seqs.append(test_data[i:i+sequence_length])
        targets.append(test_data[i+sequence_length])
    X_test = np.array(seqs)
    y_test = np.array(targets)
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.float32))
    return train_dataset, val_dataset, test_dataset
