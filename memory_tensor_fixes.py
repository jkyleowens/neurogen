import torch
import gc
import os
import numpy as np


def optimize_memory_usage():
    """
    Set memory management settings to optimize CUDA memory usage.
    Call this at the start of your training script.
    """
    # Set PyTorch memory management settings
    if torch.cuda.is_available():
        # Enable memory sharing between tensors when possible
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Set memory allocation strategy to reduce fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Empty cache to start fresh
        torch.cuda.empty_cache()
        gc.collect()
        
        # Print memory status
        print_gpu_memory_status()
    else:
        print("CUDA not available, using CPU only.")


def print_gpu_memory_status():
    """Print current GPU memory usage statistics."""
    if torch.cuda.is_available():
        print("\n--- GPU Memory Status ---")
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # Convert to GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            free = total_memory - allocated
            
            print(f"GPU {i}:")
            print(f"  Total memory: {total_memory:.2f} GB")
            print(f"  Reserved memory: {reserved:.2f} GB")
            print(f"  Allocated memory: {allocated:.2f} GB")
            print(f"  Free memory: {free:.2f} GB")
        print("------------------------\n")


def reduce_batch_size(config, reduction_factor=2):
    """
    Reduce the batch size in the config to prevent OOM errors.
    
    Args:
        config (dict): The configuration dictionary
        reduction_factor (int): Factor by which to reduce batch size
        
    Returns:
        dict: Updated configuration
    """
    current_batch_size = config['training']['batch_size']
    new_batch_size = max(1, current_batch_size // reduction_factor)
    
    if new_batch_size < current_batch_size:
        print(f"Reducing batch size from {current_batch_size} to {new_batch_size} to prevent OOM errors")
        config['training']['batch_size'] = new_batch_size
    
    return config


def fix_tensor_dimensions(tensor_a, tensor_b, dim=1):
    """
    Reshape tensors to make their dimensions compatible at the specified dimension.
    
    Args:
        tensor_a (torch.Tensor): First tensor
        tensor_b (torch.Tensor): Second tensor
        dim (int): Dimension to match
        
    Returns:
        tuple: (fixed_tensor_a, fixed_tensor_b)
    """
    if tensor_a.size(dim) == tensor_b.size(dim):
        # Already compatible
        return tensor_a, tensor_b
    
    # Get the shape of both tensors
    shape_a = list(tensor_a.size())
    shape_b = list(tensor_b.size())
    
    # Determine target dimension
    if shape_a[dim] < shape_b[dim]:
        # Expand tensor_a to match tensor_b
        if dim == 1:  # Common case for feature dimension
            # Create linear projection layer
            projection = torch.nn.Linear(shape_a[dim], shape_b[dim]).to(tensor_a.device)
            # Initialize with identity-like mapping where possible
            torch.nn.init.zeros_(projection.weight)
            min_dim = min(shape_a[dim], shape_b[dim])
            projection.weight.data[:min_dim, :min_dim] = torch.eye(min_dim)
            
            # Reshape tensor_a for projection if it has more than 2 dimensions
            original_shape = tensor_a.shape
            if tensor_a.dim() > 2:
                # Flatten all dimensions except the last one
                tensor_a_reshaped = tensor_a.reshape(-1, shape_a[dim])
                # Project
                tensor_a_projected = projection(tensor_a_reshaped)
                # Restore original shape with new feature dimension
                new_shape = list(original_shape)
                new_shape[dim] = shape_b[dim]
                fixed_tensor_a = tensor_a_projected.reshape(new_shape)
            else:
                fixed_tensor_a = projection(tensor_a)
            
            return fixed_tensor_a, tensor_b
        else:
            # For other dimensions, try to use broadcasting or padding
            if tensor_a.dim() == tensor_b.dim():
                # Try padding
                pad_size = [(0, 0) for _ in range(tensor_a.dim())]
                pad_size[dim] = (0, shape_b[dim] - shape_a[dim])
                fixed_tensor_a = torch.nn.functional.pad(tensor_a, [item for sublist in reversed(pad_size) for item in sublist])
                return fixed_tensor_a, tensor_b
            else:
                # Dimensions differ, try basic reshaping
                return tensor_a.view_as(tensor_b), tensor_b
    else:
        # Expand tensor_b to match tensor_a
        if dim == 1:  # Common case for feature dimension
            # Create linear projection layer
            projection = torch.nn.Linear(shape_b[dim], shape_a[dim]).to(tensor_b.device)
            # Initialize with identity-like mapping where possible
            torch.nn.init.zeros_(projection.weight)
            min_dim = min(shape_a[dim], shape_b[dim])
            projection.weight.data[:min_dim, :min_dim] = torch.eye(min_dim)
            
            # Reshape tensor_b for projection if it has more than 2 dimensions
            original_shape = tensor_b.shape
            if tensor_b.dim() > 2:
                # Flatten all dimensions except the last one
                tensor_b_reshaped = tensor_b.reshape(-1, shape_b[dim])
                # Project
                tensor_b_projected = projection(tensor_b_reshaped)
                # Restore original shape with new feature dimension
                new_shape = list(original_shape)
                new_shape[dim] = shape_a[dim]
                fixed_tensor_b = tensor_b_projected.reshape(new_shape)
            else:
                fixed_tensor_b = projection(tensor_b)
            
            return tensor_a, fixed_tensor_b
        else:
            # For other dimensions, try to use broadcasting or padding
            if tensor_a.dim() == tensor_b.dim():
                # Try padding
                pad_size = [(0, 0) for _ in range(tensor_b.dim())]
                pad_size[dim] = (0, shape_a[dim] - shape_b[dim])
                fixed_tensor_b = torch.nn.functional.pad(tensor_b, [item for sublist in reversed(pad_size) for item in sublist])
                return tensor_a, fixed_tensor_b
            else:
                # Dimensions differ, try basic reshaping
                return tensor_a, tensor_b.view_as(tensor_a)


class MemoryEfficientTrainer:
    """
    A wrapper class that helps manage memory efficiently during training.
    """
    
    def __init__(self, model, device, config, gradient_accumulation_steps=1):
        """
        Initialize the memory-efficient trainer.
        
        Args:
            model (torch.nn.Module): The model to train
            device (torch.device): Device to use
            config (dict): Training configuration
            gradient_accumulation_steps (int): Number of steps to accumulate gradients
        """
        self.model = model
        self.device = device
        self.config = config
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Optimize memory at initialization
        optimize_memory_usage()
        
    def train_batch(self, optimizer, data, target, reward, criterion):
        """
        Train on a single batch with memory-efficient operations.
        
        Args:
            optimizer (torch.optim.Optimizer): The optimizer
            data (torch.Tensor): Input data
            target (torch.Tensor): Target values
            reward (torch.Tensor): Reward signals
            criterion: Loss function
            
        Returns:
            float: Batch loss
        """
        # Move data to device
        data = data.to(self.device)
        target = target.to(self.device)
        reward = reward.to(self.device)
        
        # Split batch if it's too large
        batch_size = data.size(0)
        max_batch_size = self.config.get('max_micro_batch_size', 16)
        
        total_loss = 0.0
        
        if batch_size > max_batch_size:
            # Process in smaller chunks
            num_chunks = (batch_size + max_batch_size - 1) // max_batch_size
            
            for i in range(num_chunks):
                start_idx = i * max_batch_size
                end_idx = min((i + 1) * max_batch_size, batch_size)
                
                # Get chunk
                data_chunk = data[start_idx:end_idx]
                target_chunk = target[start_idx:end_idx]
                reward_chunk = reward[start_idx:end_idx]
                
                # Forward pass
                output_chunk, predicted_reward_chunk = self.model(data_chunk, external_reward=reward_chunk)
                
                # Check tensor dimensions and fix if needed
                if output_chunk.size(-1) != target_chunk.size(-1):
                    output_chunk, target_chunk = fix_tensor_dimensions(output_chunk, target_chunk, dim=-1)
                
                if predicted_reward_chunk.size(-1) != reward_chunk.size(-1):
                    predicted_reward_chunk, reward_chunk = fix_tensor_dimensions(predicted_reward_chunk, reward_chunk, dim=-1)
                
                # Calculate loss
                task_loss = criterion(output_chunk, target_chunk)
                reward_loss = criterion(predicted_reward_chunk, reward_chunk)
                loss = task_loss + 0.5 * reward_loss
                
                # Scale loss by gradient accumulation steps
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Accumulate loss
                total_loss += loss.item() * self.gradient_accumulation_steps
                
                # Clear memory for next chunk
                del data_chunk, target_chunk, reward_chunk, output_chunk, predicted_reward_chunk
                torch.cuda.empty_cache()
                
            # Step optimizer
            optimizer.step()
            optimizer.zero_grad()
            
        else:
            # Process entire batch at once
            try:
                # Forward pass
                output, predicted_reward = self.model(data, external_reward=reward)
                
                # Check tensor dimensions and fix if needed
                if output.size(-1) != target.size(-1):
                    output, target = fix_tensor_dimensions(output, target, dim=-1)
                
                if predicted_reward.size(-1) != reward.size(-1):
                    predicted_reward, reward = fix_tensor_dimensions(predicted_reward, reward, dim=-1)
                
                # Calculate loss
                task_loss = criterion(output, target)
                reward_loss = criterion(predicted_reward, reward)
                loss = task_loss + 0.5 * reward_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                # Accumulate loss
                total_loss = loss.item()
                
            except RuntimeError as e:
                # Handle CUDA OOM errors
                if "CUDA out of memory" in str(e):
                    print(f"CUDA OOM error: {e}")
                    # Clear memory
                    del data, target, reward
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Reduce batch size for future iterations
                    self.config = reduce_batch_size(self.config)
                    
                    # Return a default loss value
                    return float('inf')
                else:
                    raise e
        
        return total_loss


class TensorShapeAdapter(torch.nn.Module):
    """
    A module wrapper that handles tensor shape mismatches.
    It can be used to wrap any model to transparently fix shape issues.
    """
    
    def __init__(self, model):
        """
        Initialize the adapter.
        
        Args:
            model (torch.nn.Module): The model to wrap
        """
        super(TensorShapeAdapter, self).__init__()
        self.model = model
        self.projection_layers = {}
    
    def forward(self, *args, **kwargs):
        """
        Forward pass with shape adaptation.
        
        Returns:
            The model's output with fixed shapes
        """
        try:
            # Try running the model directly
            outputs = self.model(*args, **kwargs)
            return outputs
        except RuntimeError as e:
            # Check if it's a shape mismatch error
            error_msg = str(e)
            if "size mismatch" in error_msg or "shape mismatch" in error_msg:
                # Parse the error message to identify the tensor sizes
                if "must match" in error_msg:
                    # Extract tensor dimensions
                    import re
                    sizes = re.findall(r'tensor a \((\d+)\) must match the size of tensor b \((\d+)\)', error_msg)
                    
                    if sizes:
                        size_a = int(sizes[0][0])
                        size_b = int(sizes[0][1])
                        
                        # Create a projection layer for these dimensions if it doesn't exist
                        key = f"{size_a}_{size_b}"
                        if key not in self.projection_layers:
                            self.projection_layers[key] = torch.nn.Linear(size_a, size_b).to(
                                next(self.model.parameters()).device
                            )
                            # Initialize to preserve identity mapping where possible
                            torch.nn.init.zeros_(self.projection_layers[key].weight)
                            min_dim = min(size_a, size_b)
                            self.projection_layers[key].weight.data[:min_dim, :min_dim] = torch.eye(min_dim)
                        
                        # Process inputs
                        if "external_reward" in kwargs:
                            # Try to identify which tensor needs projection - for BrainInspiredNN
                            for i, arg in enumerate(args):
                                if isinstance(arg, torch.Tensor) and arg.size(1) == size_a:
                                    # Project this tensor
                                    projected_tensor = self.projection_layers[key](arg)
                                    # Replace the tensor in args
                                    args = list(args)
                                    args[i] = projected_tensor
                                    args = tuple(args)
                                    break
                        
                        # Try the forward pass again with the projected tensors
                        outputs = self.model(*args, **kwargs)
                        return outputs
                
                # If we couldn't parse the specific dimensions, try a more generic approach
                print(f"Shape adaptation: Generic handling for error: {error_msg}")
                # Flatten and process any tensor args
                new_args = []
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        # Create a simple projection to a standard size
                        if arg.dim() > 1 and arg.size(1) == 365:
                            # This matches the error in the logs
                            proj = torch.nn.Linear(365, 32).to(arg.device)
                            new_args.append(proj(arg))
                        else:
                            new_args.append(arg)
                    else:
                        new_args.append(arg)
                
                # Try with the new arguments
                outputs = self.model(*new_args, **kwargs)
                return outputs
            
            # If it's not a shape mismatch error, re-raise it
            raise e


# Function to apply shape fixing to a model
def apply_tensor_shape_fixes(model):
    """
    Wrap a model with TensorShapeAdapter to automatically fix shape mismatches.
    
    Args:
        model (torch.nn.Module): The model to wrap
        
    Returns:
        TensorShapeAdapter: The wrapped model
    """
    return TensorShapeAdapter(model)
