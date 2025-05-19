import argparse
import yaml
import torch
import os
from src.train import train_epoch
from src.model import BrainInspiredNN as Model
from src.controller.persistent_gru import PersistentGRUController
from torch.utils.data import DataLoader
from data_loader import create_datasets  # Assume this is implemented for yfinance data
from src.utils.performance_report import generate_performance_report


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train, validate, and test the model.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs for training.")
    args = parser.parse_args()

    # Load configuration
    config_path = "config/config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    train_dataset, val_dataset, test_dataset = create_datasets(config)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # Load model
    model_path = "models/best_model.pt"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict):
            # If checkpoint is state_dict or dict containing 'state_dict'
            model = Model(config)
            state_dict = checkpoint.get('state_dict', checkpoint)
            model.load_state_dict(state_dict)
        else:
            # Loaded a full model instance
            model = checkpoint
        print("Loaded pre-trained model.")
    else:
        model = Model(config)
        print("Initialized new model.")
    model.to(device)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = torch.nn.MSELoss()

    # Training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Training Loss: {train_loss:.4f}")

    # Validation
    val_loss = train_epoch(model, val_loader, optimizer, criterion, device, train=False)
    print(f"Validation Loss: {val_loss:.4f}")

    # Testing
    test_loss = train_epoch(model, test_loader, optimizer, criterion, device, train=False)
    print(f"Test Loss: {test_loss:.4f}")

    # Save the model
    torch.save(model, model_path)
    print(f"Model saved to {model_path}")

    # Generate performance report
    generate_performance_report(train_loss, val_loss, test_loss, output_dir="docs/")
    print("Performance report generated in docs/ directory.")


if __name__ == "__main__":
    main()
