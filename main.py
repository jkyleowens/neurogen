#!/usr/bin/env python
import argparse
import yaml
import torch
import os
from torch.utils.data import DataLoader
from src.train import train_epoch, validate
from src.model import BrainInspiredNN
from data_loader import create_datasets
from src.utils.performance_report import generate_performance_report


def main():
    parser = argparse.ArgumentParser(description="Train, validate, and test the BrainInspiredNN model.")
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--model-path', type=str, default='models/best_model.pt', help='Path to save/load model')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Ensure model dims match data features
    feature_count = len(config['data'].get('features', []))
    if 'model' not in config:
        config['model'] = {}
    config['model']['input_size'] = feature_count
    config['model']['output_size'] = feature_count

    # Device (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data loaders
    train_ds, val_ds, test_ds = create_datasets(config)
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model
    best_model = None
    best_val_loss = float('inf')
    # Initialize or load
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        model = BrainInspiredNN.setup_model(config, input_shape=train_ds[0][0].shape)
        model.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=False)
        print("Loaded existing model checkpoint.")
    else:
        model = BrainInspiredNN.setup_model(config, input_shape=train_ds[0][0].shape)
        print("Initialized new model.")
    model.to(device)

    # Optimizer and loss
    # Remove optimizer if learning via neuromodulator only
    if config.get('learning_mode', '') == 'neuromodulator':
        optimizer = None
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = torch.nn.MSELoss()

    # Training + Validation
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'state_dict': model.state_dict()}, args.model_path)
            print(f"Saved new best model (Val Loss: {best_val_loss:.4f})")

    # Load best model for testing
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    # Testing
    test_loss = validate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f}")

    # Performance report
    generate_performance_report(train_loss, best_val_loss, test_loss, output_dir='docs/')
    print("Performance report available in docs/")


if __name__ == '__main__':
    main()
