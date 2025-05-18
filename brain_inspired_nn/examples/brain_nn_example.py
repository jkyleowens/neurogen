"""
Brain-Inspired Neural Network Example

This example demonstrates how to use the complete brain-inspired neural network system.
It shows how to:
1. Initialize the model with all components
2. Process input data through the system
3. Observe the effects of neuromodulation and persistent memory
4. Use the LLM interface for enhanced processing
5. Visualize the internal states and dynamics of the system
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add the parent directory to the path so we can import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import BrainInspiredNN


def generate_synthetic_data(batch_size, seq_length, input_size, pattern_type='sine'):
    """
    Generate synthetic data for demonstration.
    
    Args:
        batch_size (int): Batch size
        seq_length (int): Sequence length
        input_size (int): Input size
        pattern_type (str): Type of pattern to generate ('sine', 'step', 'random')
        
    Returns:
        tuple: (inputs, targets, rewards)
    """
    # Initialize tensors
    inputs = torch.zeros(batch_size, seq_length, input_size)
    targets = torch.zeros(batch_size, seq_length, input_size)
    rewards = torch.zeros(batch_size, seq_length, 1)
    
    # Generate data based on pattern type
    for b in range(batch_size):
        if pattern_type == 'sine':
            # Generate sine wave patterns with different frequencies
            for i in range(input_size):
                freq = 1.0 + (i % 10) * 0.1
                phase = (b % 5) * 0.2
                inputs[b, :, i] = torch.sin(torch.linspace(0 + phase, 4*np.pi + phase, seq_length) * freq)
                targets[b, :, i] = torch.sin(torch.linspace(0.1 + phase, 4*np.pi + 0.1 + phase, seq_length) * freq)
            
            # Generate rewards based on alignment with target
            for t in range(seq_length):
                similarity = torch.cosine_similarity(inputs[b, t, :], targets[b, t, :], dim=0)
                rewards[b, t, 0] = similarity
                
        elif pattern_type == 'step':
            # Generate step function patterns
            for i in range(input_size):
                step_point = seq_length // 2 + (i % 5) - 2
                inputs[b, step_point:, i] = 1.0
                targets[b, step_point+1:, i] = 1.0
            
            # Generate rewards for correct predictions
            for t in range(1, seq_length):
                if torch.allclose(inputs[b, t, :], targets[b, t-1, :]):
                    rewards[b, t, 0] = 1.0
                    
        else:  # random
            # Generate random patterns with temporal correlation
            for t in range(seq_length):
                if t == 0:
                    inputs[b, t, :] = torch.randn(input_size)
                else:
                    inputs[b, t, :] = inputs[b, t-1, :] * 0.8 + torch.randn(input_size) * 0.2
                    
                # Targets are slightly shifted inputs
                if t < seq_length - 1:
                    targets[b, t+1, :] = inputs[b, t, :]
            
            # Random rewards with temporal correlation
            for t in range(seq_length):
                if t == 0:
                    rewards[b, t, 0] = torch.rand(1)
                else:
                    rewards[b, t, 0] = rewards[b, t-1, 0] * 0.7 + torch.rand(1) * 0.3
    
    return inputs, targets, rewards


def visualize_neurotransmitter_levels(levels, title="Neurotransmitter Levels"):
    """
    Visualize neurotransmitter levels.
    
    Args:
        levels (dict): Dictionary of neurotransmitter levels
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Extract values for each neurotransmitter
    neurotransmitters = list(levels.keys())
    values = [levels[nt].mean().item() for nt in neurotransmitters]
    
    # Create bar plot
    plt.bar(neurotransmitters, values, color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])
    plt.ylabel('Level')
    plt.title(title)
    
    # Add values on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"neurotransmitter_levels_{timestamp}.png")
    plt.close()


def visualize_dynamic_connections(connections, title="Dynamic Connection Strengths"):
    """
    Visualize dynamic connection strengths.
    
    Args:
        connections (list): List of connection strength tensors
        title (str): Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Plot each layer's connections
    for i, conn in enumerate(connections):
        plt.subplot(len(connections), 1, i+1)
        plt.imshow(conn.detach().cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title(f"Layer {i+1} Connection Strengths")
    
    plt.tight_layout()
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"dynamic_connections_{timestamp}.png")
    plt.close()


def visualize_input_attention(metadata, title="Input Attention Weights"):
    """
    Visualize input attention weights.
    
    Args:
        metadata (dict): Input metadata dictionary
        title (str): Plot title
    """
    if 'attention_weights' not in metadata:
        print("No attention weights found in metadata")
        return
    
    attention_weights = metadata['attention_weights'].detach().cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.imshow(attention_weights, cmap='hot', aspect='auto')
    plt.colorbar()
    plt.xlabel('Time Step')
    plt.ylabel('Attention Weight')
    plt.title(title)
    
    plt.tight_layout()
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"input_attention_{timestamp}.png")
    plt.close()


def run_example():
    """Run the brain-inspired neural network example."""
    print("Running Brain-Inspired Neural Network Example...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Model parameters
    input_size = 32
    hidden_size = 64
    output_size = 32
    
    # Create model
    model = BrainInspiredNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        persistent_memory_size=32,
        num_layers=2,
        dropout=0.1,
        dopamine_scale=1.0,
        serotonin_scale=0.8,
        norepinephrine_scale=0.6,
        acetylcholine_scale=0.7
    )
    
    # Generate synthetic data
    batch_size = 2
    seq_length = 20
    inputs, targets, rewards = generate_synthetic_data(
        batch_size=batch_size,
        seq_length=seq_length,
        input_size=input_size,
        pattern_type='sine'
    )
    
    print(f"Generated synthetic data: inputs shape {inputs.shape}, targets shape {targets.shape}, rewards shape {rewards.shape}")
    
    # Process data through the model
    print("\nProcessing data through the model...")
    outputs, predicted_rewards = model(inputs, external_reward=rewards)
    
    print(f"Model outputs shape: {outputs.shape}")
    print(f"Predicted rewards shape: {predicted_rewards.shape}")
    
    # Visualize neurotransmitter levels
    print("\nVisualizing neurotransmitter levels...")
    neurotransmitter_levels = model.get_neurotransmitter_levels()
    visualize_neurotransmitter_levels(neurotransmitter_levels)
    
    # Visualize dynamic connections
    print("\nVisualizing dynamic connections...")
    dynamic_connections = model.get_dynamic_connections()
    visualize_dynamic_connections(dynamic_connections)
    
    # Visualize input attention
    print("\nVisualizing input attention...")
    input_metadata = model.get_input_metadata()
    visualize_input_attention(input_metadata)
    
    # Demonstrate the effect of persistent memory
    print("\nDemonstrating the effect of persistent memory...")
    
    # Process the same input again without resetting states
    outputs2, predicted_rewards2 = model(inputs, external_reward=rewards)
    
    # Reset states and process again
    model.reset_states()
    outputs3, predicted_rewards3 = model(inputs, external_reward=rewards)
    
    # Compare outputs
    diff_with_memory = torch.mean(torch.abs(outputs2 - outputs)).item()
    diff_without_memory = torch.mean(torch.abs(outputs3 - outputs)).item()
    
    print(f"Difference with persistent memory: {diff_with_memory:.6f}")
    print(f"Difference after resetting memory: {diff_without_memory:.6f}")
    
    # Demonstrate reward-based learning
    print("\nDemonstrating reward-based learning...")
    
    # Reset states
    model.reset_states()
    
    # Process with positive rewards
    positive_rewards = torch.ones_like(rewards)
    outputs_pos, _ = model(inputs, external_reward=positive_rewards)
    
    # Get connection strengths
    connections_pos = model.get_dynamic_connections()
    
    # Reset states
    model.reset_states()
    
    # Process with negative rewards
    negative_rewards = -torch.ones_like(rewards)
    outputs_neg, _ = model(inputs, external_reward=negative_rewards)
    
    # Get connection strengths
    connections_neg = model.get_dynamic_connections()
    
    # Compare connection strengths
    connection_diff = 0
    for c_pos, c_neg in zip(connections_pos, connections_neg):
        connection_diff += torch.mean(torch.abs(c_pos - c_neg)).item()
    
    print(f"Connection strength difference due to reward: {connection_diff:.6f}")
    
    # Demonstrate LLM interface (if API key is available)
    try:
        print("\nDemonstrating LLM interface...")
        
        # Create a simple prompt for demonstration
        prompt = "Analyze the pattern in this neural activity:"
        
        # Process with LLM (this will use a mock response if no API key is available)
        llm_response = model.process_with_llm(outputs, prompt)
        
        print(f"LLM Response: {llm_response[:100]}...")  # Show first 100 chars
    except Exception as e:
        print(f"LLM interface demonstration skipped: {e}")
    
    print("\nExample completed successfully!")
    print("Visualization images have been saved to the current directory.")


if __name__ == "__main__":
    run_example()
