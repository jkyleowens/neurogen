import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import yfinance as yf
import numpy as np

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

def create_datasets(config):
    """
    Create train, validation, and test datasets and dataloaders.
    """
    # Configuration keys
    ticker = config['data']['ticker_symbol']
    sequence_length = config['data']['sequence_length']
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    features = config['data'].get('features', ['Close'])
    # Download full history
    df = yf.download(ticker, auto_adjust=True)[features].dropna()
    # Normalize data if requested
    if config['data'].get('normalize', False):
        means = df.mean()
        stds = df.std().replace(0, 1)
        df = (df - means) / stds

    # Build sequences and targets
    data = df.values
    seqs, targets = [], []
    for i in range(len(data) - sequence_length):
        seqs.append(data[i:i+sequence_length])
        targets.append(data[i+sequence_length])
    X = np.array(seqs)
    y = np.array(targets)

    # Split indices
    total = len(X)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # Slice splits
    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:], y[val_end:]

    # Create TensorDatasets
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset   = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                  torch.tensor(y_val, dtype=torch.float32))
    test_dataset  = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                  torch.tensor(y_test, dtype=torch.float32))

    return train_dataset, val_dataset, test_dataset
