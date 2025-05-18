# LLM Interface for Brain-Inspired Neural Network

## Overview

This component implements a comprehensive LLM (Large Language Model) API interface for training and validation of the brain-inspired neural network. The interface allows the neural network to connect with various LLM API endpoints, where the LLM automatically performs training and validation on the model.

## Key Features

- **Multi-Provider Support**: Works with OpenAI, Hugging Face, and Anthropic LLM APIs
- **Streaming Responses**: Supports streaming for real-time feedback
- **Training Integration**: Uses LLM feedback to adjust model training
- **Validation Framework**: Evaluates model outputs using LLM-based metrics
- **Tensor-Text Translation**: Converts between neural network representations and natural language
- **Robust Error Handling**: Gracefully handles API failures and rate limits

## Implementation Details

The LLM interface is implemented in `src/utils/llm_interface.py` and consists of:

1. **Base Provider Class**: An abstract base class defining the interface for all LLM providers
2. **Provider Implementations**: Concrete implementations for OpenAI, Hugging Face, and Anthropic
3. **Main Interface Class**: The `LLMInterface` class that provides a unified API for the rest of the system

## How to Use

### Configuration

Configure the LLM interface in `config/config.yaml`:

```yaml
llm:
  provider: "openai"  # Options: "openai", "huggingface", "anthropic"
  api_endpoint: ""  # Leave empty to use default endpoints
  model_name: "gpt-4"
  max_tokens: 1024
  temperature: 0.7
  embedding_dim: 768
  
  # Provider-specific settings
  openai:
    api_key: ""  # Set via environment variable OPENAI_API_KEY
    model_name: "gpt-4"
    embedding_model: "text-embedding-3-small"
  
  huggingface:
    api_key: ""  # Set via environment variable HF_API_TOKEN
    model_name: "mistralai/Mistral-7B-Instruct-v0.2"
    embedding_model: "sentence-transformers/all-mpnet-base-v2"
  
  anthropic:
    api_key: ""  # Set via environment variable ANTHROPIC_API_KEY
    model_name: "claude-3-sonnet-20240229"
```

### Setting API Keys

Set your API keys as environment variables:

```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Hugging Face
export HF_API_TOKEN="your-huggingface-token"

# For Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### Basic Usage

```python
from src.utils.llm_interface import LLMInterface

# Initialize the interface
llm_interface = LLMInterface(
    provider="openai",
    model_name="gpt-4",
    max_tokens=1024,
    temperature=0.7
)

# Get a response
response = llm_interface.get_response("What is neuromodulation?")
print(response)

# Get a streaming response
def print_chunk(chunk):
    print(chunk, end="", flush=True)

llm_interface.get_response(
    "Explain neural plasticity.",
    streaming=True,
    callback=print_chunk
)
```

### Training Integration

```python
# Train with LLM feedback
results = llm_interface.train_with_llm_feedback(
    model, inputs, targets, optimizer, loss_fn, epochs=5
)

# Analyze results
for epoch, loss_info in enumerate(results["loss_history"]):
    print(f"Epoch {epoch+1}: Initial Loss = {loss_info['initial_loss']:.6f}, "
          f"Adjusted Loss = {loss_info['adjusted_loss']:.6f}, "
          f"LLM Score = {loss_info['llm_score']:.2f}")
```

### Validation

```python
# Define validation prompts
validation_prompts = [
    "Explain how neural networks process information similar to the human brain.",
    "Describe the role of neuromodulators in learning and memory."
]

# Run validation
avg_score, results = llm_interface.validate_with_llm(
    model, validation_prompts, device
)

print(f"Average validation score: {avg_score:.2f}")
```

## Example Script

An example script demonstrating the LLM interface is provided at `examples/llm_interface_example.py`. Run it to see the interface in action:

```bash
python examples/llm_interface_example.py
```

## Testing

Unit tests for the LLM interface are available in `tests/test_llm_interface.py`. Run them to verify the implementation:

```bash
python -m unittest tests/test_llm_interface.py
```

## Documentation

Detailed documentation is available in `docs/LLM_INTERFACE.md`, covering:

- Architecture and design principles
- API reference
- Configuration options
- Advanced usage scenarios
- Performance considerations
- Troubleshooting

## Integration with Training

The LLM interface is integrated with the main training script (`src/train.py`). To enable LLM-based training and validation, use the `--use-llm` flag:

```bash
python src/train.py --config config/config.yaml --use-llm
```

You can also specify the validation interval:

```bash
python src/train.py --config config/config.yaml --use-llm --llm-validation-interval 5
```

## Extending the Interface

To add support for a new LLM provider:

1. Create a new class that inherits from `BaseLLMProvider`
2. Implement the required methods: `connect()`, `send_prompt()`, `send_prompt_streaming()`, and `get_embedding()`
3. Add the new provider to the `_init_provider()` method in `LLMInterface`
4. Update the configuration file to include settings for the new provider
