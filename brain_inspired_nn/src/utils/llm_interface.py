"""
LLM Interface Module

This module provides an interface for interacting with LLM API endpoints,
allowing the brain-inspired neural network to leverage LLM capabilities
for training and validation.
"""

import os
import json
import time
import logging
import requests
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    This class defines the interface that all LLM provider implementations must follow.
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection with the LLM API.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        pass
    
    @abstractmethod
    def send_prompt(self, prompt: str, **kwargs) -> str:
        """
        Send a prompt to the LLM and get a response.
        
        Args:
            prompt (str): The prompt to send to the LLM
            **kwargs: Additional parameters specific to the provider
            
        Returns:
            str: The LLM's response
        """
        pass
    
    @abstractmethod
    def send_prompt_streaming(self, prompt: str, callback=None, **kwargs) -> str:
        """
        Send a prompt to the LLM and get a streaming response.
        
        Args:
            prompt (str): The prompt to send to the LLM
            callback: Optional callback function to process streaming chunks
            **kwargs: Additional parameters specific to the provider
            
        Returns:
            str: The complete LLM's response after streaming
        """
        pass
    
    @abstractmethod
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get an embedding vector for the given text.
        
        Args:
            text (str): The text to embed
            
        Returns:
            np.ndarray: The embedding vector
        """
        pass


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI API provider implementation.
    """
    
    def __init__(self, api_key: str = None, model_name: str = "gpt-4", 
                 max_tokens: int = 1024, temperature: float = 0.7,
                 embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the OpenAI provider.
        
        Args:
            api_key (str, optional): OpenAI API key. If None, will try to get from environment
            model_name (str): Name of the model to use
            max_tokens (int): Maximum number of tokens in the response
            temperature (float): Temperature parameter for response generation
            embedding_model (str): Model to use for embeddings
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.embedding_model = embedding_model
        self.base_url = "https://api.openai.com/v1"
        self.chat_endpoint = f"{self.base_url}/chat/completions"
        self.embedding_endpoint = f"{self.base_url}/embeddings"
        
        # Try to import OpenAI library if available
        try:
            import openai
            self.openai = openai
            self.openai.api_key = self.api_key
            self._use_library = True
            logger.info("Using OpenAI Python library")
        except ImportError:
            self._use_library = False
            logger.info("OpenAI Python library not found, using direct API calls")
    
    def connect(self) -> bool:
        """
        Test connection to the OpenAI API.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # Simple test request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5
            }
            response = requests.post(
                self.chat_endpoint,
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI API: {e}")
            return False
    
    def send_prompt(self, prompt: str, **kwargs) -> str:
        """
        Send a prompt to the OpenAI API and get a response.
        
        Args:
            prompt (str): The prompt to send
            **kwargs: Additional parameters to override defaults
            
        Returns:
            str: The model's response
        """
        try:
            model = kwargs.get("model", self.model_name)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            temperature = kwargs.get("temperature", self.temperature)
            
            if self._use_library:
                # Use the OpenAI Python library
                response = self.openai.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            else:
                # Use direct API calls
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                response = requests.post(
                    self.chat_endpoint,
                    headers=headers,
                    json=data
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {e}")
            return f"Error: {str(e)}"
    
    def send_prompt_streaming(self, prompt: str, callback=None, **kwargs) -> str:
        """
        Send a prompt to the OpenAI API and get a streaming response.
        
        Args:
            prompt (str): The prompt to send
            callback: Optional callback function to process streaming chunks
            **kwargs: Additional parameters to override defaults
            
        Returns:
            str: The complete response after streaming
        """
        try:
            model = kwargs.get("model", self.model_name)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            temperature = kwargs.get("temperature", self.temperature)
            
            full_response = ""
            
            if self._use_library:
                # Use the OpenAI Python library
                stream = self.openai.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        if callback:
                            callback(content)
            else:
                # Use direct API calls
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True
                }
                
                response = requests.post(
                    self.chat_endpoint,
                    headers=headers,
                    json=data,
                    stream=True
                )
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: ') and line != 'data: [DONE]':
                            data = json.loads(line[6:])
                            if 'choices' in data and data['choices']:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content = delta['content']
                                    full_response += content
                                    if callback:
                                        callback(content)
            
            return full_response
        except Exception as e:
            logger.error(f"Error in OpenAI streaming API call: {e}")
            return f"Error: {str(e)}"
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get an embedding vector for the given text using OpenAI's embedding model.
        
        Args:
            text (str): The text to embed
            
        Returns:
            np.ndarray: The embedding vector
        """
        try:
            if self._use_library:
                # Use the OpenAI Python library
                response = self.openai.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                return np.array(response.data[0].embedding)
            else:
                # Use direct API calls
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": self.embedding_model,
                    "input": text
                }
                response = requests.post(
                    self.embedding_endpoint,
                    headers=headers,
                    json=data
                )
                response.raise_for_status()
                result = response.json()
                return np.array(result["data"][0]["embedding"])
        except Exception as e:
            logger.error(f"Error getting embedding from OpenAI: {e}")
            # Return a zero vector as fallback
            return np.zeros(1536)  # Default dimension for OpenAI embeddings


class HuggingFaceProvider(BaseLLMProvider):
    """
    Hugging Face Inference API provider implementation.
    """
    
    def __init__(self, api_key: str = None, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", 
                 max_tokens: int = 1024, temperature: float = 0.7,
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the Hugging Face provider.
        
        Args:
            api_key (str, optional): Hugging Face API token. If None, will try to get from environment
            model_name (str): Name or ID of the model to use
            max_tokens (int): Maximum number of tokens in the response
            temperature (float): Temperature parameter for response generation
            embedding_model (str): Model to use for embeddings
        """
        self.api_key = api_key or os.environ.get("HF_API_TOKEN")
        if not self.api_key:
            logger.warning("No Hugging Face API token provided. Set HF_API_TOKEN environment variable.")
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.embedding_model = embedding_model
        
        # Base URL for Hugging Face Inference API
        self.base_url = "https://api-inference.huggingface.co/models"
        
        # Try to import transformers if available for local embedding
        try:
            import transformers
            from transformers import AutoTokenizer, AutoModel
            self.transformers = transformers
            self._has_transformers = True
            logger.info("Using transformers library for local embeddings")
        except ImportError:
            self._has_transformers = False
            logger.info("Transformers library not found, using API for embeddings")
        
        # Initialize local embedding model if transformers is available
        self._local_embedding_model = None
        self._local_tokenizer = None
    
    def _init_local_embedding_model(self):
        """Initialize the local embedding model if not already done."""
        if self._has_transformers and self._local_embedding_model is None:
            try:
                self._local_tokenizer = self.transformers.AutoTokenizer.from_pretrained(self.embedding_model)
                self._local_embedding_model = self.transformers.AutoModel.from_pretrained(self.embedding_model)
                logger.info(f"Loaded local embedding model: {self.embedding_model}")
            except Exception as e:
                logger.error(f"Failed to load local embedding model: {e}")
                self._has_transformers = False
    
    def connect(self) -> bool:
        """
        Test connection to the Hugging Face Inference API.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # Simple test request
            headers = {"Authorization": f"Bearer {self.api_key}"}
            data = {"inputs": "Hello"}
            response = requests.post(
                f"{self.base_url}/{self.model_name}",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Hugging Face API: {e}")
            return False
    
    def send_prompt(self, prompt: str, **kwargs) -> str:
        """
        Send a prompt to the Hugging Face Inference API and get a response.
        
        Args:
            prompt (str): The prompt to send
            **kwargs: Additional parameters to override defaults
            
        Returns:
            str: The model's response
        """
        try:
            model = kwargs.get("model", self.model_name)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            temperature = kwargs.get("temperature", self.temperature)
            
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            # Format the payload based on the model type
            if "mistral" in model.lower() or "llama" in model.lower():
                # Format for instruction-tuned models
                payload = {
                    "inputs": f"<s>[INST] {prompt} [/INST]",
                    "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                        "return_full_text": False
                    }
                }
            else:
                # Generic format
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                        "return_full_text": False
                    }
                }
            
            # Add wait_for_model header to ensure model is loaded
            headers["X-Wait-For-Model"] = "true"
            
            response = requests.post(
                f"{self.base_url}/{model}",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    return result[0]["generated_text"]
                else:
                    return str(result[0])
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"]
            else:
                return str(result)
        except Exception as e:
            logger.error(f"Error in Hugging Face API call: {e}")
            return f"Error: {str(e)}"
    
    def send_prompt_streaming(self, prompt: str, callback=None, **kwargs) -> str:
        """
        Send a prompt to the Hugging Face Inference API and get a streaming response.
        Note: Not all Hugging Face models support streaming.
        
        Args:
            prompt (str): The prompt to send
            callback: Optional callback function to process streaming chunks
            **kwargs: Additional parameters to override defaults
            
        Returns:
            str: The complete response after streaming
        """
        try:
            model = kwargs.get("model", self.model_name)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            temperature = kwargs.get("temperature", self.temperature)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "X-Wait-For-Model": "true"
            }
            
            # Format the payload based on the model type
            if "mistral" in model.lower() or "llama" in model.lower():
                payload = {
                    "inputs": f"<s>[INST] {prompt} [/INST]",
                    "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                        "return_full_text": False,
                        "stream": True
                    }
                }
            else:
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                        "return_full_text": False,
                        "stream": True
                    }
                }
            
            response = requests.post(
                f"{self.base_url}/{model}",
                headers=headers,
                json=payload,
                stream=True
            )
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = json.loads(line[6:])
                        if isinstance(data, list) and len(data) > 0:
                            if "token" in data[0]:
                                token = data[0]["token"]["text"]
                                full_response += token
                                if callback:
                                    callback(token)
                            elif "generated_text" in data[0]:
                                text = data[0]["generated_text"]
                                if text not in full_response:
                                    new_text = text[len(full_response):]
                                    full_response = text
                                    if callback:
                                        callback(new_text)
            
            return full_response
        except Exception as e:
            logger.error(f"Error in Hugging Face streaming API call: {e}")
            return f"Error: {str(e)}"
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get an embedding vector for the given text using a Hugging Face model.
        
        Args:
            text (str): The text to embed
            
        Returns:
            np.ndarray: The embedding vector
        """
        # Try to use local model first if available
        if self._has_transformers:
            try:
                self._init_local_embedding_model()
                
                # Use the local model for embedding
                inputs = self._local_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self._local_embedding_model(**inputs)
                
                # Use mean pooling to get a single vector
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embedding = (sum_embeddings / sum_mask).squeeze().numpy()
                return embedding
            except Exception as e:
                logger.error(f"Error using local embedding model: {e}")
                logger.info("Falling back to API for embeddings")
        
        # Fall back to API
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {"inputs": text}
            
            response = requests.post(
                f"{self.base_url}/{self.embedding_model}",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    # Some models return a list of lists
                    return np.array(result[0])
                else:
                    # Some models return a list of floats
                    return np.array(result)
            else:
                return np.array(result)
        except Exception as e:
            logger.error(f"Error getting embedding from Hugging Face API: {e}")
            # Return a zero vector as fallback
            return np.zeros(768)  # Default dimension for many HF embeddings


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude API provider implementation.
    """
    
    def __init__(self, api_key: str = None, model_name: str = "claude-3-sonnet-20240229", 
                 max_tokens: int = 1024, temperature: float = 0.7):
        """
        Initialize the Anthropic provider.
        
        Args:
            api_key (str, optional): Anthropic API key. If None, will try to get from environment
            model_name (str): Name of the model to use
            max_tokens (int): Maximum number of tokens in the response
            temperature (float): Temperature parameter for response generation
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("No Anthropic API key provided. Set ANTHROPIC_API_KEY environment variable.")
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_url = "https://api.anthropic.com/v1"
        self.messages_endpoint = f"{self.base_url}/messages"
        
        # Try to import Anthropic library if available
        try:
            import anthropic
            self.anthropic = anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self._use_library = True
            logger.info("Using Anthropic Python library")
        except ImportError:
            self._use_library = False
            logger.info("Anthropic Python library not found, using direct API calls")
    
    def connect(self) -> bool:
        """
        Test connection to the Anthropic API.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # Simple test request
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            data = {
                "model": self.model_name,
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "Hello"}]
            }
            response = requests.post(
                self.messages_endpoint,
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Anthropic API: {e}")
            return False
    
    def send_prompt(self, prompt: str, **kwargs) -> str:
        """
        Send a prompt to the Anthropic API and get a response.
        
        Args:
            prompt (str): The prompt to send
            **kwargs: Additional parameters to override defaults
            
        Returns:
            str: The model's response
        """
        try:
            model = kwargs.get("model", self.model_name)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            temperature = kwargs.get("temperature", self.temperature)
            
            if self._use_library:
                # Use the Anthropic Python library
                response = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            else:
                # Use direct API calls
                headers = {
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
                data = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}]
                }
                response = requests.post(
                    self.messages_endpoint,
                    headers=headers,
                    json=data
                )
                response.raise_for_status()
                result = response.json()
                return result["content"][0]["text"]
        except Exception as e:
            logger.error(f"Error in Anthropic API call: {e}")
            return f"Error: {str(e)}"
    
    def send_prompt_streaming(self, prompt: str, callback=None, **kwargs) -> str:
        """
        Send a prompt to the Anthropic API and get a streaming response.
        
        Args:
            prompt (str): The prompt to send
            callback: Optional callback function to process streaming chunks
            **kwargs: Additional parameters to override defaults
            
        Returns:
            str: The complete response after streaming
        """
        try:
            model = kwargs.get("model", self.model_name)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            temperature = kwargs.get("temperature", self.temperature)
            
            full_response = ""
            
            if self._use_library:
                # Use the Anthropic Python library
                with self.client.messages.stream(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                ) as stream:
                    for text in stream.text_stream:
                        full_response += text
                        if callback:
                            callback(text)
            else:
                # Use direct API calls
                headers = {
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
                data = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True
                }
                
                response = requests.post(
                    self.messages_endpoint,
                    headers=headers,
                    json=data,
                    stream=True
                )
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: ') and line != 'data: [DONE]':
                            data = json.loads(line[6:])
                            if data.get("type") == "content_block_delta":
                                delta = data.get("delta", {})
                                if "text" in delta:
                                    text = delta["text"]
                                    full_response += text
                                    if callback:
                                        callback(text)
            
            return full_response
        except Exception as e:
            logger.error(f"Error in Anthropic streaming API call: {e}")
            return f"Error: {str(e)}"
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get an embedding vector for the given text.
        
        Note: Anthropic doesn't provide a dedicated embedding API, so we'll use
        OpenAI's embedding API if available, or return a random embedding as fallback.
        
        Args:
            text (str): The text to embed
            
        Returns:
            np.ndarray: The embedding vector
        """
        logger.warning("Anthropic doesn't provide a dedicated embedding API. Using fallback.")
        
        # Try to use OpenAI if available
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            try:
                openai_provider = OpenAIProvider(api_key=openai_key)
                return openai_provider.get_embedding(text)
            except Exception as e:
                logger.error(f"Error using OpenAI for embeddings: {e}")
        
        # Fallback to random embedding
        embedding_dim = 1536  # Same as OpenAI's default
        return np.random.normal(0, 0.1, embedding_dim)


class LLMInterface:
    """
    Interface for interacting with LLM API endpoints.
    
    This class provides methods for sending prompts to an LLM API,
    processing responses, and converting between LLM text and model
    tensor representations.
    """
    
    def __init__(self, api_endpoint: str = None, model_name: str = "gpt-4", 
                 max_tokens: int = 1024, temperature: float = 0.7,
                 provider: str = "openai", api_key: str = None,
                 embedding_dim: int = 768):
        """
        Initialize the LLM Interface.
        
        Args:
            api_endpoint (str, optional): URL of the LLM API endpoint
            model_name (str): Name of the LLM model to use
            max_tokens (int): Maximum number of tokens in the response
            temperature (float): Temperature parameter for response generation
            provider (str): LLM provider to use ('openai', 'huggingface', 'anthropic')
            api_key (str, optional): API key for the provider
            embedding_dim (int): Dimension of embedding vectors
        """
        self.api_endpoint = api_endpoint
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.provider_name = provider.lower()
        self.api_key = api_key
        self.embedding_dim = embedding_dim
        
        # Initialize the appropriate provider
        self.provider = self._init_provider()
        
        # Test connection
        if not self.provider.connect():
            logger.warning(f"Failed to connect to {self.provider_name} API. Some functionality may be limited.")
    
    def _init_provider(self) -> BaseLLMProvider:
        """
        Initialize the appropriate LLM provider based on configuration.
        
        Returns:
            BaseLLMProvider: The initialized provider
        """
        if self.provider_name == "openai":
            return OpenAIProvider(
                api_key=self.api_key,
                model_name=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
        elif self.provider_name == "huggingface":
            return HuggingFaceProvider(
                api_key=self.api_key,
                model_name=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
        elif self.provider_name == "anthropic":
            return AnthropicProvider(
                api_key=self.api_key,
                model_name=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
        else:
            logger.warning(f"Unknown provider: {self.provider_name}. Falling back to OpenAI.")
            return OpenAIProvider(
                api_key=self.api_key,
                model_name=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
    
    def get_response(self, prompt: str, streaming: bool = False, callback=None, **kwargs) -> str:
        """
        Get a response from the LLM API for the given prompt.
        
        Args:
            prompt (str): The prompt to send to the LLM
            streaming (bool): Whether to use streaming response
            callback: Optional callback function for streaming responses
            **kwargs: Additional parameters to pass to the provider
            
        Returns:
            str: The LLM's response
        """
        if streaming:
            return self.provider.send_prompt_streaming(prompt, callback, **kwargs)
        else:
            return self.provider.send_prompt(prompt, **kwargs)
    
    def response_to_model_input(self, response: str, sequence_length: int = 64) -> torch.Tensor:
        """
        Convert an LLM response to a tensor input for the neural network model.
        
        This method uses the provider's embedding functionality to create a
        meaningful representation of the text.
        
        Args:
            response (str): The LLM response text
            sequence_length (int): The desired sequence length for the model input
            
        Returns:
            torch.Tensor: Tensor representation of the response
        """
        # Get embedding from provider
        embedding = self.provider.get_embedding(response)
        
        # Reshape to match sequence length
        if len(embedding.shape) == 1:
            # Single embedding vector, need to expand to sequence
            embedding_dim = embedding.shape[0]
            
            # Create a sequence by repeating and adding noise
            sequence = np.zeros((sequence_length, embedding_dim))
            for i in range(sequence_length):
                # Add some noise to make each position slightly different
                noise = np.random.normal(0, 0.01, embedding_dim)
                sequence[i] = embedding + noise
        else:
            # Already a sequence, pad or truncate
            orig_seq_len, embedding_dim = embedding.shape
            if orig_seq_len < sequence_length:
                # Pad
                padding = np.zeros((sequence_length - orig_seq_len, embedding_dim))
                sequence = np.vstack([embedding, padding])
            else:
                # Truncate
                sequence = embedding[:sequence_length]
        
        # Convert to tensor with batch dimension
        tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        
        return tensor
    
    def model_output_to_prompt(self, model_output: torch.Tensor) -> str:
        """
        Convert a model output tensor to a prompt for the LLM.
        
        This method creates a descriptive prompt based on the model's output
        that can be sent to the LLM for evaluation.
        
        Args:
            model_output (torch.Tensor): The model's output tensor
            
        Returns:
            str: A prompt derived from the model output
        """
        # Extract the output tensor (remove batch dimension)
        output = model_output.squeeze(0).cpu().numpy()
        
        # Create a descriptive prompt
        prompt = "The neural network produced the following output:\n\n"
        
        # Add statistical description
        prompt += f"Shape: {output.shape}\n"
        prompt += f"Mean: {np.mean(output):.4f}\n"
        prompt += f"Std Dev: {np.std(output):.4f}\n"
        prompt += f"Min: {np.min(output):.4f}\n"
        prompt += f"Max: {np.max(output):.4f}\n\n"
        
        # Add sample values
        prompt += "Sample values (first 5 rows):\n"
        for i in range(min(5, output.shape[0])):
            row = output[i]
            prompt += f"Row {i}: [{', '.join([f'{x:.4f}' for x in row[:5]])}...]\n"
        
        prompt += "\nPlease evaluate this output based on the following criteria:\n"
        prompt += "1. Pattern recognition: Are there discernible patterns in the output?\n"
        prompt += "2. Consistency: Is the output consistent with expected neural network behavior?\n"
        prompt += "3. Anomalies: Are there any anomalies or unexpected values?\n"
        
        return prompt
    
    def evaluate_output(self, output_prompt: str, original_prompt: str) -> Dict[str, Any]:
        """
        Use the LLM to evaluate the model's output against the original prompt.
        
        Args:
            output_prompt (str): Prompt derived from the model's output
            original_prompt (str): The original prompt sent to the LLM
            
        Returns:
            dict: Evaluation results including score and feedback
        """
        # Create an evaluation prompt
        eval_prompt = f"""
        Original Prompt: {original_prompt}
        
        Model Output: {output_prompt}
        
        Please evaluate the model's output based on the following criteria:
        1. Relevance to the original prompt
        2. Coherence and logical flow
        3. Accuracy of information
        4. Overall quality
        
        Provide a score from 0 to 10 and detailed feedback.
        Format your response as JSON with the following structure:
        {{
            "score": <score>,
            "feedback": "<detailed feedback>",
            "strengths": ["<strength1>", "<strength2>", ...],
            "weaknesses": ["<weakness1>", "<weakness2>", ...],
            "suggestions": ["<suggestion1>", "<suggestion2>", ...]
        }}
        """
        
        # Get LLM evaluation
        eval_response = self.get_response(eval_prompt)
        
        try:
            # Parse the JSON response
            evaluation = json.loads(eval_response)
            
            # Ensure the required fields are present
            if "score" not in evaluation:
                evaluation["score"] = 0
            if "feedback" not in evaluation:
                evaluation["feedback"] = "No feedback provided"
            if "strengths" not in evaluation:
                evaluation["strengths"] = []
            if "weaknesses" not in evaluation:
                evaluation["weaknesses"] = []
            if "suggestions" not in evaluation:
                evaluation["suggestions"] = []
            
            return evaluation
            
        except json.JSONDecodeError:
            # If the response is not valid JSON, extract score and feedback manually
            try:
                # Try to find a score in the text
                import re
                score_match = re.search(r'score[:\s]*(\d+(?:\.\d+)?)', eval_response, re.IGNORECASE)
                score = float(score_match.group(1)) if score_match else 0
                
                return {
                    "score": score,
                    "feedback": eval_response,
                    "strengths": [],
                    "weaknesses": [],
                    "suggestions": []
                }
            except Exception:
                # If all else fails, return a default evaluation
                return {
                    "score": 0,
                    "feedback": "Failed to parse evaluation response",
                    "strengths": [],
                    "weaknesses": [],
                    "suggestions": []
                }
    
    def train_with_llm_feedback(self, model: torch.nn.Module, inputs: torch.Tensor, 
                               targets: torch.Tensor, optimizer: torch.optim.Optimizer,
                               loss_fn: callable, epochs: int = 1) -> Dict[str, Any]:
        """
        Train the model using LLM feedback to adjust the loss function.
        
        Args:
            model (torch.nn.Module): The neural network model to train
            inputs (torch.Tensor): Input tensor for the model
            targets (torch.Tensor): Target tensor for the model
            optimizer (torch.optim.Optimizer): Optimizer for the model
            loss_fn (callable): Loss function
            epochs (int): Number of epochs to train
            
        Returns:
            dict: Training results including loss history and LLM feedback
        """
        model.train()
        loss_history = []
        feedback_history = []
        
        for epoch in range(epochs):
            # Forward pass
            outputs, predicted_rewards = model(inputs)
            
            # Calculate initial loss
            initial_loss = loss_fn(outputs, targets)
            
            # Get LLM feedback on the model's output
            output_prompt = self.model_output_to_prompt(outputs)
            input_prompt = f"Input tensor shape: {inputs.shape}, Target tensor shape: {targets.shape}"
            evaluation = self.evaluate_output(output_prompt, input_prompt)
            
            # Calculate LLM feedback factor (0.5 to 1.5 based on score)
            llm_score = evaluation["score"]
            feedback_factor = 1.5 - (llm_score / 10.0)  # Higher score = lower loss
            
            # Adjust loss based on LLM feedback
            adjusted_loss = initial_loss * feedback_factor
            
            # Backward pass and optimization
            optimizer.zero_grad()
            adjusted_loss.backward()
            optimizer.step()
            
            # Store history
            loss_history.append({
                "epoch": epoch,
                "initial_loss": initial_loss.item(),
                "adjusted_loss": adjusted_loss.item(),
                "llm_score": llm_score
            })
            
            feedback_history.append(evaluation)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{epochs}, Initial Loss: {initial_loss.item():.6f}, "
                       f"Adjusted Loss: {adjusted_loss.item():.6f}, LLM Score: {llm_score:.2f}")
        
        return {
            "loss_history": loss_history,
            "feedback_history": feedback_history
        }
    
    def validate_with_llm(self, model: torch.nn.Module, validation_prompts: List[str],
                         device: torch.device) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Validate the model using LLM integration.
        
        Args:
            model (torch.nn.Module): The neural network model to validate
            validation_prompts (List[str]): List of prompts for validation
            device (torch.device): Device to run the model on
            
        Returns:
            Tuple[float, List[Dict]]: Average score and detailed results
        """
        model.eval()
        results = []
        
        with torch.no_grad():
            for prompt in validation_prompts:
                # Get LLM response
                llm_response = self.get_response(prompt)
                
                # Convert LLM response to model input
                model_input = self.response_to_model_input(llm_response)
                model_input = model_input.to(device)
                
                # Get model prediction
                output, _ = model(model_input)
                
                # Convert model output to LLM prompt
                output_prompt = self.model_output_to_prompt(output)
                
                # Get LLM evaluation
                evaluation = self.evaluate_output(output_prompt, prompt)
                
                # Store results
                results.append({
                    'prompt': prompt,
                    'llm_response': llm_response,
                    'model_output': output.cpu().numpy(),
                    'evaluation': evaluation
                })
        
        # Calculate average score
        avg_score = np.mean([r['evaluation']['score'] for r in results])
        
        return avg_score, results
    
    def generate_training_data(self, prompts: List[str], sequence_length: int = 64,
                              output_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate training data using the LLM.
        
        Args:
            prompts (List[str]): List of prompts to generate data from
            sequence_length (int): Sequence length for inputs
            output_size (int): Size of the output vectors
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input and target tensors
        """
        inputs = []
        targets = []
        
        for prompt in prompts:
            # Get LLM response
            response = self.get_response(prompt)
            
            # Convert to model input
            model_input = self.response_to_model_input(response, sequence_length)
            inputs.append(model_input.squeeze(0).numpy())
            
            # Generate a target based on the response
            # This is a simplified approach - in a real scenario, you might
            # want to generate targets based on specific criteria
            target_prompt = f"Based on this response: '{response}', generate a concise summary."
            target_response = self.get_response(target_prompt)
            
            # Convert to target tensor
            target_embedding = self.provider.get_embedding(target_response)
            
            # Resize to match output_size
            if len(target_embedding) > output_size:
                target = target_embedding[:output_size]
            else:
                target = np.pad(target_embedding, (0, output_size - len(target_embedding)))
            
            targets.append(target)
        
        # Convert to tensors
        inputs_tensor = torch.tensor(np.array(inputs), dtype=torch.float32)
        targets_tensor = torch.tensor(np.array(targets), dtype=torch.float32)
        
        return inputs_tensor, targets_tensor
