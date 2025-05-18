"""
Brain-Inspired Neural Network Model

This module implements the main model that integrates all components:
- Persistent GRU controller
- Neuromodulator system
- Input preprocessing
- LLM interface

The model is designed to mimic certain aspects of brain functionality,
particularly the interaction between neural processing, neuromodulation,
sensory processing, and higher-level cognitive functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from controller.persistent_gru import PersistentGRUController
from neuromodulator.neuromodulator import NeuromodulatorSystem, RewardPredictor
from preprocessing.input_processor import InputProcessor
from utils.llm_interface import LLMInterface


class BrainInspiredNN(nn.Module):
    """
    A brain-inspired neural network that integrates all components into a cohesive system.
    
    This model is designed to mimic certain aspects of brain functionality:
    - The controller acts as a central processing unit (like prefrontal cortex)
    - The neuromodulator system regulates neural activity based on rewards (like dopaminergic systems)
    - The input processor handles sensory information (like sensory cortices)
    - The LLM interface provides higher-level cognitive capabilities (like language areas)
    
    The system preserves the biologically-inspired nature with the controller acting as
    a central module that controls the neuromodulator based on preprocessed inputs and reward signals.
    """
    
    def __init__(self, input_size, hidden_size, output_size, 
                 persistent_memory_size=128, num_layers=2, dropout=0.1,
                 dopamine_scale=1.0, serotonin_scale=0.8, 
                 norepinephrine_scale=0.6, acetylcholine_scale=0.7,
                 reward_decay=0.95, llm_provider="openai", llm_api_key=None):
        """
        Initialize the Brain-Inspired Neural Network.
        
        Args:
            input_size (int): Size of the input features
            hidden_size (int): Size of the hidden state
            output_size (int): Size of the output
            persistent_memory_size (int): Size of the persistent memory
            num_layers (int): Number of GRU layers
            dropout (float): Dropout probability
            dopamine_scale (float): Scaling factor for dopamine effects
            serotonin_scale (float): Scaling factor for serotonin effects
            norepinephrine_scale (float): Scaling factor for norepinephrine effects
            acetylcholine_scale (float): Scaling factor for acetylcholine effects
            reward_decay (float): Decay factor for reward signals
            llm_provider (str): LLM provider to use ('openai', 'huggingface', 'anthropic')
            llm_api_key (str, optional): API key for the LLM provider
        """
        super(BrainInspiredNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Input Processor - handles sensory information preprocessing
        self.input_processor = InputProcessor(
            input_size=input_size,
            output_size=input_size,  # Maintain same dimensionality for controller input
            hidden_size=hidden_size,
            feature_size=hidden_size // 2,
            encoding_size=input_size,
            adaptation_rate=0.1
        )
        
        # GRU Controller - central processing unit
        self.controller = PersistentGRUController(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=hidden_size,  # Output to neuromodulator, not final output
            num_layers=num_layers,
            persistent_memory_size=persistent_memory_size,
            dropout=dropout
        )
        
        # Neuromodulator System - regulates neural activity based on rewards
        self.neuromodulator = NeuromodulatorSystem(
            input_size=input_size,
            hidden_size=hidden_size,
            dopamine_scale=dopamine_scale,
            serotonin_scale=serotonin_scale,
            norepinephrine_scale=norepinephrine_scale,
            acetylcholine_scale=acetylcholine_scale,
            reward_decay=reward_decay
        )
        
        # Reward Predictor - predicts rewards for reinforcement learning
        self.reward_predictor = RewardPredictor(
            state_size=hidden_size,
            action_size=output_size,
            hidden_size=hidden_size
        )
        
        # Output Layer - generates final output
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # LLM Interface - provides higher-level cognitive capabilities
        self.llm_interface = LLMInterface(
            provider=llm_provider,
            api_key=llm_api_key,
            embedding_dim=hidden_size
        )
        
        # Internal state tracking
        self.hidden_states = None
        self.persistent_memories = None
        self.neurotransmitter_levels = None
        self.input_metadata = None
    
    def forward(self, x, hidden=None, persistent_memory=None, external_reward=None):
        """
        Forward pass of the Brain-Inspired Neural Network.
        
        The information flow follows a biologically-inspired pattern:
        1. Input preprocessing (sensory processing)
        2. Controller processing (central neural processing)
        3. Reward prediction (basal ganglia function)
        4. Neuromodulation (neurotransmitter effects)
        5. Output generation (motor/response output)
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
            hidden (torch.Tensor, optional): Initial hidden state tensor
            persistent_memory (torch.Tensor, optional): Initial persistent memory tensor
            external_reward (torch.Tensor, optional): External reward signal
            
        Returns:
            tuple: (outputs, predicted_rewards)
        """
        batch_size, seq_length, _ = x.size()
        device = x.device
        
        # Initialize states if not provided
        if hidden is None or persistent_memory is None:
            hidden, persistent_memory = self.controller.init_hidden(batch_size, device)
        
        # Preprocess input through the input processor
        processed_x, input_metadata = self.input_processor(x)
        self.input_metadata = input_metadata
        
        # Process through controller
        controller_outputs, hidden_states, persistent_memories = self.controller(
            processed_x, hidden, persistent_memory
        )
        
        # Store states
        self.hidden_states = hidden_states
        self.persistent_memories = persistent_memories
        
        # Process each time step with neuromodulation
        outputs = []
        predicted_rewards = []
        
        for t in range(seq_length):
            # Get controller output for current time step
            controller_output_t = controller_outputs[:, t, :]
            
            # Generate action (preliminary output)
            action_t = self.output_layer(controller_output_t)
            
            # Predict reward if external reward not provided
            if external_reward is None:
                reward_t = self.reward_predictor(controller_output_t, action_t)
            else:
                reward_t = external_reward[:, t].unsqueeze(1) if external_reward.dim() > 1 else external_reward.unsqueeze(1)
            
            # Apply neuromodulation
            modulated_output_t, neurotransmitter_levels_t = self.neuromodulator(
                processed_x[:, t, :], controller_output_t, reward_t
            )
            
            # Generate final output
            final_output_t = self.output_layer(modulated_output_t)
            
            # Store outputs and rewards
            outputs.append(final_output_t)
            predicted_rewards.append(reward_t)
            
            # Store neurotransmitter levels for the last time step
            if t == seq_length - 1:
                self.neurotransmitter_levels = neurotransmitter_levels_t
        
        # Stack outputs and rewards along sequence dimension
        outputs = torch.stack(outputs, dim=1)
        predicted_rewards = torch.stack(predicted_rewards, dim=1)
        
        return outputs, predicted_rewards
    
    def process_with_llm(self, outputs, prompt=None):
        """
        Process model outputs with the LLM interface.
        
        This method allows the model to leverage LLM capabilities for
        interpreting and enhancing its outputs.
        
        Args:
            outputs (torch.Tensor): Model outputs to process
            prompt (str, optional): Additional prompt to guide the LLM
            
        Returns:
            str: LLM-processed response
        """
        # Convert model output to LLM prompt
        output_prompt = self.llm_interface.model_output_to_prompt(outputs)
        
        # Combine with additional prompt if provided
        if prompt:
            combined_prompt = f"{prompt}\n\n{output_prompt}"
        else:
            combined_prompt = output_prompt
        
        # Get LLM response
        response = self.llm_interface.get_response(combined_prompt)
        
        return response
    
    def train_with_llm_feedback(self, inputs, targets, optimizer, loss_fn, epochs=1):
        """
        Train the model using LLM feedback to adjust the learning process.
        
        Args:
            inputs (torch.Tensor): Input tensor for the model
            targets (torch.Tensor): Target tensor for the model
            optimizer (torch.optim.Optimizer): Optimizer for the model
            loss_fn (callable): Loss function
            epochs (int): Number of epochs to train
            
        Returns:
            dict: Training results
        """
        return self.llm_interface.train_with_llm_feedback(
            model=self,
            inputs=inputs,
            targets=targets,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=epochs
        )
    
    def reset_states(self):
        """Reset all internal states."""
        self.hidden_states = None
        self.persistent_memories = None
        self.neurotransmitter_levels = None
        self.input_metadata = None
        self.controller.reset_state()
        self.neuromodulator.reset()
        self.input_processor.reset_adaptation()
        
    def get_neurotransmitter_levels(self):
        """Get the current neurotransmitter levels."""
        return self.neurotransmitter_levels
    
    def get_input_metadata(self):
        """Get the metadata from input processing."""
        return self.input_metadata
    
    def get_dynamic_connections(self):
        """Get the dynamic connection strengths from the controller."""
        return self.controller.get_dynamic_connections()
    
    def get_synaptic_weights(self):
        """Get the synaptic weights from the neuromodulator system."""
        return self.neuromodulator.get_synaptic_weights()
    
    def update_from_reward(self, reward):
        """
        Update the model based on reward signals.
        
        This method allows for explicit reward-based updates outside of the
        forward pass, enabling reinforcement learning-like behavior.
        
        Args:
            reward (torch.Tensor): Reward signal
            
        Returns:
            bool: True if update was successful
        """
        # Update controller
        controller_updated = self.controller.update_from_reward(reward)
        
        # Update neuromodulator system
        if self.hidden_states and self.hidden_states[-1]:
            last_hidden = self.hidden_states[-1][-1]  # Last layer, last time step
            _, _ = self.neuromodulator(
                torch.zeros(last_hidden.size(0), self.input_size, device=last_hidden.device),
                last_hidden,
                reward
            )
        
        return controller_updated
