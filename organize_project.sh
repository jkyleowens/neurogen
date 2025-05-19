#!/bin/bash

# Create necessary directories if they don't exist
mkdir -p .github/workflows scripts data/raw data/processed logs examples

# Make sure scripts are executable
chmod +x scripts/*.py

# Create a .gitignore file if it doesn't exist or update it
if [ ! -f .gitignore ]; then
    cat > .gitignore << 'GITIGNORE'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
logs/
data/processed/
models/*.pt
results/
GITIGNORE
fi

# Create a README for the scripts directory if it doesn't exist
if [ ! -f scripts/README.md ]; then
    cat > scripts/README.md << 'README'
# Scripts

This directory contains utility scripts for the Brain-Inspired Neural Network project:

- `create_model_checkpoint.py`: Creates an initial model checkpoint for testing
- `test_only.py`: Runs evaluation on a pre-trained model without training

## Usage

```bash
# Create a model checkpoint
python scripts/create_model_checkpoint.py

# Test a model
python scripts/test_only.py --model models/model_checkpoint.pt --config config/config.yaml
```
README
fi

# Create a README for the data directory if it doesn't exist
if [ ! -f data/README.md ]; then
    cat > data/README.md << 'README'
# Data

This directory contains data for the Brain-Inspired Neural Network project:

- `raw/`: Raw data files before preprocessing
- `processed/`: Processed data ready for training and evaluation

## Data Format

The model expects time series data with features and target values.
README
fi

echo "Project structure organized successfully!"
