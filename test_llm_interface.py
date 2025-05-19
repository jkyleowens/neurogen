"""
Tests for the LLM Interface module.
"""

import os
import sys
import unittest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.llm_interface import (
    LLMInterface, BaseLLMProvider, OpenAIProvider, 
    HuggingFaceProvider, AnthropicProvider
)


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self):
        self.connect_called = False
        self.send_prompt_called = False
        self.send_prompt_streaming_called = False
        self.get_embedding_called = False
    
    def connect(self):
        self.connect_called = True
        return True
    
    def send_prompt(self, prompt, **kwargs):
        self.send_prompt_called = True
        return f"Response to: {prompt}"
    
    def send_prompt_streaming(self, prompt, callback=None, **kwargs):
        self.send_prompt_streaming_called = True
        response = f"Streaming response to: {prompt}"
        if callback:
            for char in response:
                callback(char)
        return response
    
    def get_embedding(self, text):
        self.get_embedding_called = True
        return np.random.randn(768)


class TestLLMInterface(unittest.TestCase):
    """Test cases for the LLM Interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_provider = MockLLMProvider()
        
        # Create a patched LLMInterface that uses our mock provider
        with patch('src.utils.llm_interface.LLMInterface._init_provider') as mock_init:
            mock_init.return_value = self.mock_provider
            self.llm_interface = LLMInterface(
                api_endpoint="https://mock-api.example.com",
                model_name="mock-model",
                provider="mock"
            )
    
    def test_initialization(self):
        """Test that the LLM interface initializes correctly."""
        self.assertEqual(self.llm_interface.api_endpoint, "https://mock-api.example.com")
        self.assertEqual(self.llm_interface.model_name, "mock-model")
        self.assertEqual(self.llm_interface.provider_name, "mock")
        self.assertTrue(self.mock_provider.connect_called)
    
    def test_get_response(self):
        """Test getting a response from the LLM."""
        prompt = "Test prompt"
        response = self.llm_interface.get_response(prompt)
        self.assertTrue(self.mock_provider.send_prompt_called)
        self.assertEqual(response, f"Response to: {prompt}")
    
    def test_get_response_streaming(self):
        """Test getting a streaming response from the LLM."""
        prompt = "Test streaming prompt"
        chunks = []
        
        def collect_chunk(chunk):
            chunks.append(chunk)
        
        response = self.llm_interface.get_response(
            prompt, streaming=True, callback=collect_chunk
        )
        
        self.assertTrue(self.mock_provider.send_prompt_streaming_called)
        self.assertEqual(response, f"Streaming response to: {prompt}")
        self.assertEqual(''.join(chunks), response)
    
    def test_response_to_model_input(self):
        """Test converting LLM response to model input."""
        response = "Test response"
        model_input = self.llm_interface.response_to_model_input(response, sequence_length=10)
        
        self.assertTrue(self.mock_provider.get_embedding_called)
        self.assertIsInstance(model_input, torch.Tensor)
        self.assertEqual(model_input.shape, (1, 10, 768))
    
    def test_model_output_to_prompt(self):
        """Test converting model output to LLM prompt."""
        model_output = torch.randn(1, 5, 10)
        prompt = self.llm_interface.model_output_to_prompt(model_output)
        
        self.assertIsInstance(prompt, str)
        self.assertIn("neural network produced", prompt.lower())
        self.assertIn("shape: (5, 10)", prompt)
    
    def test_evaluate_output(self):
        """Test evaluating model output with LLM."""
        # Mock the get_response method to return a valid JSON response
        self.llm_interface.get_response = MagicMock(return_value="""
        {
            "score": 7.5,
            "feedback": "Good output with some minor issues.",
            "strengths": ["Clear patterns", "Consistent values"],
            "weaknesses": ["Some outliers"],
            "suggestions": ["Normalize the data"]
        }
        """)
        
        output_prompt = "Model output description"
        original_prompt = "Original test prompt"
        evaluation = self.llm_interface.evaluate_output(output_prompt, original_prompt)
        
        self.assertIsInstance(evaluation, dict)
        self.assertEqual(evaluation["score"], 7.5)
        self.assertIn("Good output", evaluation["feedback"])
        self.assertEqual(len(evaluation["strengths"]), 2)
        self.assertEqual(len(evaluation["weaknesses"]), 1)
        self.assertEqual(len(evaluation["suggestions"]), 1)


class TestProviders(unittest.TestCase):
    """Test cases for the LLM providers."""
    
    def test_openai_provider_initialization(self):
        """Test that the OpenAI provider initializes correctly."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIProvider()
            self.assertEqual(provider.api_key, 'test-key')
            self.assertEqual(provider.model_name, 'gpt-4')
    
    def test_huggingface_provider_initialization(self):
        """Test that the Hugging Face provider initializes correctly."""
        with patch.dict('os.environ', {'HF_API_TOKEN': 'test-token'}):
            provider = HuggingFaceProvider()
            self.assertEqual(provider.api_key, 'test-token')
            self.assertIn('mistral', provider.model_name.lower())
    
    def test_anthropic_provider_initialization(self):
        """Test that the Anthropic provider initializes correctly."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            provider = AnthropicProvider()
            self.assertEqual(provider.api_key, 'test-key')
            self.assertIn('claude', provider.model_name.lower())


if __name__ == '__main__':
    unittest.main()
