"""
LLM Interface Example

This script demonstrates how to use the LLM interface for training and validation
of the brain-inspired neural network.
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.model import BrainInspiredNN
from src.utils.llm_interface import LLMInterface


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model(config):
    """Set up the model based on configuration."""
    # Extract model parameters from config
    input_size = config.get('input_size', 128)
    hidden_size = config['controller']['hidden_size']
    output_size = config.get('output_size', 64)
    persistent_memory_size = config['controller']['persistent_memory_size']
    num_layers = config['controller']['num_layers']
    dropout = config['controller']['dropout']
    
    # Neuromodulator parameters
    dopamine_scale = config['neuromodulator']['dopamine_scale']
    serotonin_scale = config['neuromodulator']['serotonin_scale']
    norepinephrine_scale = config['neuromodulator']['norepinephrine_scale']
    acetylcholine_scale = config['neuromodulator']['acetylcholine_scale']
    reward_decay = config['neuromodulator']['reward_decay']
    
    # Create model
    model = BrainInspiredNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        persistent_memory_size=persistent_memory_size,
        num_layers=num_layers,
        dropout=dropout,
        dopamine_scale=dopamine_scale,
        serotonin_scale=serotonin_scale,
        norepinephrine_scale=norepinephrine_scale,
        acetylcholine_scale=acetylcholine_scale,
        reward_decay=reward_decay
    )
    
    return model


def setup_llm_interface(config):
    """Set up the LLM interface based on configuration."""
    provider = config['llm']['provider']
    
    # Get provider-specific settings
    provider_config = config['llm'].get(provider, {})
    
    # Initialize LLM interface
    llm_interface = LLMInterface(
        api_endpoint=config['llm']['api_endpoint'],
        model_name=provider_config.get('model_name', config['llm']['model_name']),
        max_tokens=config['llm']['max_tokens'],
        temperature=config['llm']['temperature'],
        provider=provider,
        api_key=provider_config.get('api_key', None),
        embedding_dim=config['llm']['embedding_dim']
    )
    
    return llm_interface


def generate_dummy_data(batch_size, sequence_length, input_size, output_size):
    """Generate dummy data for demonstration purposes."""
    inputs = torch.randn(batch_size, sequence_length, input_size)
    targets = torch.randn(batch_size, sequence_length, output_size)
    rewards = torch.randn(batch_size, sequence_length, 1)
    
    return inputs, targets, rewards


def example_llm_validation(model, llm_interface, device):
    """Demonstrate LLM validation."""
    print("\n=== LLM Validation Example ===")
    
    # Sample validation prompts
    validation_prompts = [
        "Explain how neural networks process information similar to the human brain.",
        "Describe the role of neuromodulators in learning and memory."
    ]
    
    # Run validation
    avg_score, results = llm_interface.validate_with_llm(model, validation_prompts, device)
    
    print(f"Average validation score: {avg_score:.2f}")
    
    # Print detailed results for the first prompt
    print("\nDetailed results for first prompt:")
    print(f"Prompt: {results[0]['prompt']}")
    print(f"LLM Response (excerpt): {results[0]['llm_response'][:100]}...")
    print(f"Evaluation score: {results[0]['evaluation']['score']}")
    print(f"Feedback: {results[0]['evaluation']['feedback'][:200]}...")
    
    return results


def example_llm_training_data_generation(llm_interface):
    """Demonstrate generating training data using LLM."""
    print("\n=== LLM Training Data Generation Example ===")
    
    # Sample prompts for data generation
    prompts = [
        "Explain the concept of neural plasticity in learning.",
        "How do neurotransmitters affect signal transmission in the brain?"
    ]
    
    # Generate training data
    inputs, targets = llm_interface.generate_training_data(
        prompts, sequence_length=64, output_size=64
    )
    
    print(f"Generated input tensor shape: {inputs.shape}")
    print(f"Generated target tensor shape: {targets.shape}")
    
    return inputs, targets


def example_llm_feedback_training(model, llm_interface, device):
    """Demonstrate training with LLM feedback."""
    print("\n=== LLM Feedback Training Example ===")
    
    # Generate small batch of dummy data
    inputs, targets, _ = generate_dummy_data(2, 16, 128, 64)
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Define loss function
    loss_fn = torch.nn.MSELoss()
    
    # Train for a few epochs with LLM feedback
    results = llm_interface.train_with_llm_feedback(
        model, inputs, targets, optimizer, loss_fn, epochs=2
    )
    
    print("Training results:")
    for epoch, loss_info in enumerate(results["loss_history"]):
        print(f"Epoch {epoch+1}: Initial Loss = {loss_info['initial_loss']:.6f}, "
              f"Adjusted Loss = {loss_info['adjusted_loss']:.6f}, "
              f"LLM Score = {loss_info['llm_score']:.2f}")
    
    return results


def main():
    """Main function to demonstrate LLM interface usage."""
    # Load configuration
    config_path = os.path.join(project_root, "config", "config.yaml")
    config = load_config(config_path)
    
    # Set device
    device = torch.device(config['general']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(config['general']['seed'])
    np.random.seed(config['general']['seed'])
    
    # Set up model
    model = setup_model(config)
    model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Set up LLM interface
    llm_interface = setup_llm_interface(config)
    print(f"LLM interface initialized with {config['llm']['provider']} provider")
    
    # Example: Simple LLM response
    prompt = "Explain the concept of neuromodulation in simple terms."
    print("\n=== Simple LLM Response Example ===")
    print(f"Prompt: {prompt}")
    response = llm_interface.get_response(prompt)
    print(f"Response: {response[:200]}...")  # Print first 200 chars
    
    # Example: Streaming response
    print("\n=== Streaming Response Example ===")
    print(f"Prompt: {prompt}")
    print("Response (streaming):")
    
    def print_chunk(chunk):
        print(chunk, end="", flush=True)
    
    llm_interface.get_response(prompt, streaming=True, callback=print_chunk)
    print()  # Add newline after streaming
    
    # Example: LLM validation
    validation_results = example_llm_validation(model, llm_interface, device)
    
    # Example: Generate training data
    inputs, targets = example_llm_training_data_generation(llm_interface)
    
    # Example: Train with LLM feedback
    training_results = example_llm_feedback_training(model, llm_interface, device)
    
    print("\nAll examples completed successfully!")


if __name__ == "__main__":
    main()
