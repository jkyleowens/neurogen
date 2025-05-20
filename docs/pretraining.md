# Neural Component Pretraining

This document details the pretraining mechanism implemented for the neuromodulator and controller components of the brain-inspired neural network model.

## Overview

Pretraining neural components provides the following benefits:
1. Ensures components have a valid starting point before being used in feedback loops
2. Stabilizes learning during the main training phase
3. Reduces the likelihood of convergence to poor local optima
4. Improves overall model performance and generalization

## Pretraining Components

### 1. Controller Pretraining

The controller component (PersistentGRU or BioGRU) is pretrained to predict sequences effectively before it is used in the full model. This ensures that the controller has learned basic sequence processing capabilities.

For BioGRU specifically:
- **Temporal coherence loss**: Encourages stable representations over time
- **Representation diversity loss**: Prevents neuron death by ensuring diverse activation patterns
- **Pathway optimization**: Optimizes neuron pathways for efficient signal propagation

### 2. Neuromodulator Pretraining

The neuromodulator components are pretrained to generate appropriate feedback signals based on prediction errors. This ensures that the feedback mechanism is well-calibrated and drives learning in the right direction.

Key aspects:
- **Reward-error correlation**: Ensures rewards negatively correlate with prediction errors
- **Scale calibration**: Sets appropriate scaling factors for different neurotransmitters
- **Feedback stability**: Prevents excessive feedback oscillations

### 3. BioGRU Feedback Mechanism

For BioGRU, we separately pretrain the feedback response mechanism to ensure it responds appropriately to reward signals. This is critical for stable neuromodulator-driven learning.

## Configuration

Pretraining can be configured in the `financial_config.yaml` file:

```yaml
pretraining:
  enabled: true
  epochs: 5
  controller:
    enabled: true
    learning_rate: 0.001
    feedback_epochs: 3  # For BioGRU feedback mechanism
    feedback_learning_rate: 0.0005
  neuromodulator:
    enabled: true
    learning_rate: 0.0005
```

## Command-line Options

Pretraining can also be controlled via command-line arguments:

```bash
python main.py --config config/financial_config.yaml --pretrain --pretrain-epochs 10
```

To skip specific components:

```bash
python main.py --config config/financial_config.yaml --pretrain --skip-controller-pretrain
python main.py --config config/financial_config.yaml --pretrain --skip-neuromod-pretrain
```

## Implementation Details

1. **Controller Pretraining**: Uses sequence prediction with MSE loss
2. **Neuromodulator Pretraining**: Optimizes correlation between rewards and prediction errors
3. **BioGRU Feedback Pretraining**: Ensures feedback improves predictions on subsequent passes

## Expected Outcomes

After pretraining:
1. Controller component should be able to process sequences effectively
2. Neuromodulator components should generate rewards that correlate with prediction quality
3. Feedback mechanisms should respond appropriately to reward signals
4. The full model should train more stably and reach better performance
