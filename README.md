# Brain-Inspired Neural Network System

This project aims to develop a biologically-inspired neural network system that resembles brain functionality. The system uses a persistent GRU neural network as a central controller, with different modules working together including a neuromodulator system based on reward signals.

## Project Structure

- `src/`: Source code for the neural network implementation
  - `controller/`: GRU-based controller implementation
  - `neuromodulator/`: Neuromodulation system based on reward signals
  - `utils/`: Utility functions and helper classes
    - `pretrain_utils.py`: Utilities for pretraining neural components
    - `bio_gru_pretraining.py`: Specialized pretraining for BioGRU
    - `financial_data_utils.py`: Financial data preprocessing utilities
- `config/`: Configuration files for different experiments
- `docs/`: Documentation files
  - `pretraining.md`: Documentation on neural component pretraining
- `tests/`: Test cases for the implementation

## Key Features

- **Biologically-Inspired Learning**: Uses neuromodulator-driven learning instead of backpropagation
- **BioGRU**: Biologically-plausible GRU implementation for sequence processing
- **Component Pretraining**: Pretraining for neuromodulator and controller components
- **Financial Prediction**: Specialized for financial time series prediction with technical indicators

## Dependencies

See `requirements.txt` for a list of dependencies.

## Getting Started

1. Install the required dependencies: `pip install -r requirements.txt`
2. Configure your experiment in `config/financial_config.yaml`
3. Run training with pretraining enabled:

   ```bash
   python main.py --config config/financial_config.yaml --pretrain
   ```

## Training with Pretraining

The system now includes pretraining for neural components to ensure stability in feedback loops:

```bash
# Run with default pretraining settings
python main.py --config config/financial_config.yaml --pretrain

# Specify number of pretraining epochs
python main.py --config config/financial_config.yaml --pretrain --pretrain-epochs 10

# Skip specific component pretraining
python main.py --config config/financial_config.yaml --pretrain --skip-controller-pretrain
python main.py --config config/financial_config.yaml --pretrain --skip-neuromod-pretrain
```

Pretraining can also be configured in the config file:

```yaml
pretraining:
  enabled: true
  epochs: 5
  controller:
    enabled: true
    learning_rate: 0.001
  neuromodulator:
    enabled: true
    learning_rate: 0.0005
```

See `docs/pretraining.md` for more detailed information on the pretraining mechanism.

## LLM Integration

The system is designed to interface with an LLM API endpoint for training and validation.

## References

- Documentation in the `docs/` folder
- Configuration examples in the `config/` folder
