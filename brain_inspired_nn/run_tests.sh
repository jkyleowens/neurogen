#!/bin/bash

# Run tests for the Brain-Inspired Neural Network

# Create __init__.py file in tests directory if it doesn't exist
mkdir -p tests
touch tests/__init__.py

# Run the test script
python tests/test_model.py
