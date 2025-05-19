import torch
from torch.utils.data import TensorDataset, DataLoader

def prepare_test_data(config):
    """
    Prepare data loaders for training, validation, and testing.

    Args:
        config (dict): Configuration dictionary with data and training parameters.

    Returns:
        dict: Contains DataLoaders and metadata.
    """
    # This function should call the preprocess_data static method from your model
    from src.model import BrainInspiredNN

    # Load your raw stock data here (e.g., from CSV or DataFrame)
    # For example, assume config['data']['stock_data_path'] points to CSV
    import pandas as pd
    stock_data = pd.read_csv(config['data']['stock_data_path'], index_col=0, parse_dates=True)

    # Preprocess data using model's method
    data_info = BrainInspiredNN.preprocess_data(stock_data, config)

    return data_info