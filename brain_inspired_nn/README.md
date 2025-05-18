# Brain-Inspired Neural Network System

This project aims to develop a biologically-inspired neural network system that resembles brain functionality. The system uses a persistent GRU neural network as a central controller, with different modules working together including a neuromodulator system based on reward signals.

## Project Structure

- `src/`: Source code for the neural network implementation
  - `controller/`: GRU-based controller implementation
  - `neuromodulator/`: Neuromodulation system based on reward signals
  - `utils/`: Utility functions and helper classes
- `config/`: Configuration files for different experiments
- `docs/`: Documentation files
- `tests/`: Test cases for the implementation

## Dependencies

See `requirements.txt` for a list of dependencies.

## Getting Started

1. Install the required dependencies: `pip install -r requirements.txt`
2. Configure the system parameters in `config/config.yaml`
3. Run the training script: `python src/train.py`

## LLM Integration

The system is designed to interface with an LLM API endpoint for training and validation.
