import os
import sys
import numpy as np
import torch
import json
from datetime import datetime
from tqdm import tqdm
import time
import gc
from torch import nn
from .utils.memory_tensor_fixes import (
    optimize_memory_usage, print_gpu_memory_status, 
    fix_tensor_dimensions, apply_tensor_shape_fixes,
    MemoryEfficientTrainer
)


class LLMTrainer:
    """
    Enhanced training module that uses OpenAI to improve the bio-inspired neural network.
    Fixed to handle GPU memory issues and tensor shape mismatches.
    """
    
    def __init__(self, config, openai_interface=None, device='cuda'):
        """
        Initialize the LLM-based trainer with memory optimizations.
        
        Args:
            config (dict): Model and training configuration
            openai_interface (OpenAIInterface, optional): Interface for OpenAI API
            device (str): Device for computation
        """
        self.config = config
        self.llm = openai_interface
        
        # Set up device with proper memory management
        if device == 'cuda' and torch.cuda.is_available():
            # Optimize memory usage
            optimize_memory_usage()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            print("Using CPU for training")
        
        # Needed for dynamic model creation
        try:
            from src.model import BrainInspiredNN
            self.BrainInspiredNN = BrainInspiredNN
        except ImportError:
            print("Warning: Could not import BrainInspiredNN from src.model")
            print("Trying to import from current directory...")
            try:
                from model import BrainInspiredNN
                self.BrainInspiredNN = BrainInspiredNN
            except ImportError:
                print("Error: Could not import BrainInspiredNN. Please check your project structure.")
                raise
        
        # Create base model
        self.model = None  # Will be created after data is loaded to ensure correct dimensions
        
        # Track training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'neuromodulation_params': [],
            'market_analyses': [],
            'parameter_adjustments': [],
            'memory_usage': [],
            'batch_size_changes': []
        }
        
        # Prepare optimization components (will be initialized after model creation)
        self.optimizer = None
        self.scheduler = None
        
        # Memory-efficient trainer
        self.efficient_trainer = None
        
        # Set gradient accumulation steps based on config or default
        self.gradient_accumulation_steps = config.get('training', {}).get('gradient_accumulation_steps', 1)
        
        # Max micro batch size to prevent OOM errors
        self.max_micro_batch_size = config.get('training', {}).get('max_micro_batch_size', 16)
        
        # Keep track of tensor shapes for troubleshooting
        self.tensor_shapes = {}
    
    def _create_model(self, input_shape):
        """
        Create a brain-inspired neural network model with dimensions matching the data.
        
        Args:
            input_shape (tuple): Shape of input data (batch_size, seq_length, feature_dim)
        
        Returns:
            BrainInspiredNN: The created model
        """
        # Extract model parameters from config, ensuring dimensions match data
        input_size = input_shape[2]  # Use actual feature dimension from data
        hidden_size = self.config['controller']['hidden_size']
        output_size = 1  # For stock prediction - single value output
        
        # Update config with actual input size
        self.config['input_size'] = input_size
        
        print(f"Creating model with input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}")
        
        # Get controller parameters
        persistent_memory_size = self.config['controller']['persistent_memory_size']
        num_layers = self.config['controller']['num_layers']
        dropout = self.config['controller']['dropout']
        
        # Get neuromodulator parameters
        dopamine_scale = self.config['neuromodulator']['dopamine_scale']
        serotonin_scale = self.config['neuromodulator']['serotonin_scale']
        norepinephrine_scale = self.config['neuromodulator']['norepinephrine_scale']
        acetylcholine_scale = self.config['neuromodulator']['acetylcholine_scale']
        reward_decay = self.config['neuromodulator']['reward_decay']
        
        # Create model with correct architecture
        try:
            model = self.BrainInspiredNN(
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
            
            # Wrap model with shape adapter
            model = apply_tensor_shape_fixes(model)
            
            return model
            
        except Exception as e:
            print(f"Error creating model: {e}")
            print("Attempting to create model using setup_model method...")
            
            # Fallback to setup_model if it exists
            try:
                model = self.BrainInspiredNN.setup_model(self.config, input_shape)
                
                # Wrap model with shape adapter
                model = apply_tensor_shape_fixes(model)
                
                return model
                
            except Exception as e2:
                print(f"Error using setup_model: {e2}")
                raise ValueError("Could not create model with either method")
    
    def _create_optimizer(self):
        """Create optimizer for model training."""
        if self.model is None:
            raise ValueError("Model must be created before optimizer")
            
        optimizer_name = self.config['training']['optimizer'].lower()
        lr = self.config['training']['learning_rate']
        
        # Set up optimizer
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.optimizer is None:
            raise ValueError("Optimizer must be created before scheduler")
            
        scheduler_name = self.config['training'].get('scheduler', 'none').lower()
        
        if scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config['training']['num_epochs']
            )
        elif scheduler_name == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        else:
            scheduler = None
        
        return scheduler
    
    def prepare_data(self):
        """
        Prepare data for model training with LLM-enhanced preprocessing.
        
        Returns:
            tuple: (train_dataloader, val_dataloader, test_dataloader, data_info)
        """
        import yfinance as yf
        
        # Load stock data
        ticker = self.config['data']['ticker_symbol']
        start_date = self.config['data']['start_date']
        end_date = self.config['data']['end_date']
        
        print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            
            if stock_data.empty:
                raise ValueError(f"No data found for ticker {ticker} in the specified date range")
            
            print(f"Downloaded {len(stock_data)} days of stock data")
            
            # Process data
            data_info = self.BrainInspiredNN.preprocess_data(stock_data, self.config)
            
            # Get dataloaders
            train_dataloader = data_info['dataloaders']['train']
            val_dataloader = data_info['dataloaders']['val']
            test_dataloader = data_info['dataloaders']['test']
            
            # Now that we have data, create the model with correct input dimensions
            # First, get a sample from the dataloader to get input dimensions
            for data_batch, _, _ in train_dataloader:
                input_shape = data_batch.shape
                break
            
            print(f"Creating model with input shape: {input_shape}")
            self.model = self._create_model(input_shape)
            self.model.to(self.device)
            
            # Create memory-efficient trainer
            self.efficient_trainer = MemoryEfficientTrainer(
                self.model, self.device, self.config, 
                gradient_accumulation_steps=self.gradient_accumulation_steps
            )
            
            # Now create optimizer and scheduler
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler()
            
            # Use OpenAI to analyze feature importance if enabled
            if self.config.get('use_llm_feature_analysis', True) and self.llm is not None:
                try:
                    feature_columns = data_info['feature_columns']
                    price_data = stock_data['Close'].values if 'Close' in stock_data.columns else stock_data.iloc[:, 0].values
                    
                    # Create a simplified technical indicators dict
                    technical_indicators = {}
                    for col in stock_data.columns:
                        if col != 'Close' and col in feature_columns:
                            technical_indicators[col] = stock_data[col].values
                    
                    # Get feature importance analysis
                    feature_analysis = self.llm.assess_feature_importance(
                        feature_columns, price_data, technical_indicators
                    )
                    
                    print("\nLLM Feature Importance Analysis:")
                    if 'feature_rankings' in feature_analysis:
                        print("Top features:")
                        for i, feature in enumerate(feature_analysis['feature_rankings'][:5]):
                            score = feature_analysis.get('importance_scores', {}).get(feature, '-')
                            print(f"  {i+1}. {feature} (Score: {score})")
                    
                    # Store feature analysis in data_info
                    data_info['feature_analysis'] = feature_analysis
                    
                except Exception as e:
                    print(f"Warning: Feature importance analysis failed: {e}")
            
            return train_dataloader, val_dataloader, test_dataloader, data_info
        
        except Exception as e:
            print(f"Error preparing data: {e}")
            raise
    
    def update_neuromodulation_params(self, epoch, val_loss, price_data):
        """
        Update neuromodulation parameters based on market analysis and model performance.
        
        Args:
            epoch (int): Current training epoch
            val_loss (float): Validation loss
            price_data (np.ndarray): Recent price data
            
        Returns:
            bool: Whether parameters were updated
        """
        # Only update every N epochs (reduce API calls)
        update_frequency = self.config['training'].get('neuromod_update_freq', 5)
        if epoch % update_frequency != 0:
            return False
        
        # Skip if LLM interface not available
        if self.llm is None:
            return False
        
        # Get current neuromodulation parameters
        try:
            current_params = {
                'dopamine_scale': float(self.model.model.neuromodulator.dopamine_scale),
                'serotonin_scale': float(self.model.model.neuromodulator.serotonin_scale),
                'norepinephrine_scale': float(self.model.model.neuromodulator.norepinephrine_scale),
                'acetylcholine_scale': float(self.model.model.neuromodulator.acetylcholine_scale),
                'reward_decay': float(self.model.model.neuromodulator.reward_decay)
            }
        except Exception as e:
            print(f"Warning: Could not get current neuromodulation parameters: {e}")
            try:
                # Try direct access if the model is not wrapped
                current_params = {
                    'dopamine_scale': float(self.model.neuromodulator.dopamine_scale),
                    'serotonin_scale': float(self.model.neuromodulator.serotonin_scale),
                    'norepinephrine_scale': float(self.model.neuromodulator.norepinephrine_scale),
                    'acetylcholine_scale': float(self.model.neuromodulator.acetylcholine_scale),
                    'reward_decay': float(self.model.neuromodulator.reward_decay)
                }
            except:
                print("Could not access neuromodulation parameters. Skipping update.")
                return False
        
        # Get recent performance
        recent_performance = {
            'val_loss': float(val_loss),
            'loss_trend': 'improving' if len(self.history['val_loss']) > 1 and val_loss < self.history['val_loss'][-1] else 'worsening',
            'epoch': epoch
        }
        
        try:
            # Skip if LLM integration is disabled
            if not self.config.get('use_llm_neuromod', True):
                print("Skipping LLM neuromodulation parameter update (disabled in config)")
                return False
                
            # Analyze market patterns
            market_analysis = self.llm.analyze_market_patterns(price_data)
            self.history['market_analyses'].append(market_analysis)
            
            # Get parameter suggestions
            suggestions = self.llm.suggest_neuromodulation_params(
                market_analysis, recent_performance, current_params
            )
            
            # Log suggestions
            self.history['parameter_adjustments'].append({
                'epoch': epoch,
                'suggestions': suggestions
            })
            
            # Apply adjustments if they exist
            if 'parameter_adjustments' in suggestions:
                adjustments = suggestions['parameter_adjustments']
                
                print("\nUpdating neuromodulation parameters based on LLM suggestions:")
                for param, value in adjustments.items():
                    try:
                        # Try with wrapped model first
                        if param == 'dopamine_scale':
                            self.model.model.neuromodulator.dopamine_scale = nn.Parameter(
                                torch.tensor([value], device=self.device)
                            )
                            print(f"  Dopamine scale: {current_params['dopamine_scale']:.4f} -> {value:.4f}")
                        
                        elif param == 'serotonin_scale':
                            self.model.model.neuromodulator.serotonin_scale = nn.Parameter(
                                torch.tensor([value], device=self.device)
                            )
                            print(f"  Serotonin scale: {current_params['serotonin_scale']:.4f} -> {value:.4f}")
                        
                        elif param == 'norepinephrine_scale':
                            self.model.model.neuromodulator.norepinephrine_scale = nn.Parameter(
                                torch.tensor([value], device=self.device)
                            )
                            print(f"  Norepinephrine scale: {current_params['norepinephrine_scale']:.4f} -> {value:.4f}")
                        
                        elif param == 'acetylcholine_scale':
                            self.model.model.neuromodulator.acetylcholine_scale = nn.Parameter(
                                torch.tensor([value], device=self.device)
                            )
                            print(f"  Acetylcholine scale: {current_params['acetylcholine_scale']:.4f} -> {value:.4f}")
                        
                        elif param == 'reward_decay':
                            self.model.model.neuromodulator.reward_decay = nn.Parameter(
                                torch.tensor([value], device=self.device)
                            )
                            print(f"  Reward decay: {current_params['reward_decay']:.4f} -> {value:.4f}")
                    except AttributeError:
                        # Try direct access if the previous didn't work
                        try:
                            if param == 'dopamine_scale':
                                self.model.neuromodulator.dopamine_scale = nn.Parameter(
                                    torch.tensor([value], device=self.device)
                                )
                                print(f"  Dopamine scale: {current_params['dopamine_scale']:.4f} -> {value:.4f}")
                            
                            elif param == 'serotonin_scale':
                                self.model.neuromodulator.serotonin_scale = nn.Parameter(
                                    torch.tensor([value], device=self.device)
                                )
                                print(f"  Serotonin scale: {current_params['serotonin_scale']:.4f} -> {value:.4f}")
                            
                            elif param == 'norepinephrine_scale':
                                self.model.neuromodulator.norepinephrine_scale = nn.Parameter(
                                    torch.tensor([value], device=self.device)
                                )
                                print(f"  Norepinephrine scale: {current_params['norepinephrine_scale']:.4f} -> {value:.4f}")
                            
                            elif param == 'acetylcholine_scale':
                                self.model.neuromodulator.acetylcholine_scale = nn.Parameter(
                                    torch.tensor([value], device=self.device)
                                )
                                print(f"  Acetylcholine scale: {current_params['acetylcholine_scale']:.4f} -> {value:.4f}")
                            
                            elif param == 'reward_decay':
                                self.model.neuromodulator.reward_decay = nn.Parameter(
                                    torch.tensor([value], device=self.device)
                                )
                                print(f"  Reward decay: {current_params['reward_decay']:.4f} -> {value:.4f}")
                        except Exception as e:
                            print(f"  Failed to update {param}: {e}")
                
                # Get updated parameters to log
                try:
                    updated_params = {
                        'dopamine_scale': float(self.model.model.neuromodulator.dopamine_scale),
                        'serotonin_scale': float(self.model.model.neuromodulator.serotonin_scale),
                        'norepinephrine_scale': float(self.model.model.neuromodulator.norepinephrine_scale),
                        'acetylcholine_scale': float(self.model.model.neuromodulator.acetylcholine_scale),
                        'reward_decay': float(self.model.model.neuromodulator.reward_decay)
                    }
                except AttributeError:
                    try:
                        updated_params = {
                            'dopamine_scale': float(self.model.neuromodulator.dopamine_scale),
                            'serotonin_scale': float(self.model.neuromodulator.serotonin_scale),
                            'norepinephrine_scale': float(self.model.neuromodulator.norepinephrine_scale),
                            'acetylcholine_scale': float(self.model.neuromodulator.acetylcholine_scale),
                            'reward_decay': float(self.model.neuromodulator.reward_decay)
                        }
                    except Exception as e:
                        print(f"Could not get updated parameters: {e}")
                        updated_params = adjustments
                
                self.history['neuromodulation_params'].append({
                    'epoch': epoch,
                    'params': updated_params,
                    'market_regime': market_analysis.get('market_regime', 'unknown')
                })
                
                return True
                
        except Exception as e:
            print(f"Warning: Failed to update neuromodulation parameters: {e}")
        
        return False
    
    def train_epoch(self, dataloader, epoch):
        """
        Train the model for one epoch with memory-efficient processing.
        
        Args:
            dataloader (DataLoader): Training data loader
            epoch (int): Current epoch number
            
        Returns:
            float: Average training loss
        """
        total_loss = 0.0
        criterion = torch.nn.MSELoss()
        
        # Toggle training mode
        self.model.train()
        
        # Log memory usage before training
        print_gpu_memory_status()
        
        # Track current batch size for monitoring
        current_batch_size = self.config['training']['batch_size']
        
        batch_count = 0
        for batch_idx, (data, target, reward) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            try:
                # Use memory-efficient trainer
                batch_loss = self.efficient_trainer.train_batch(
                    self.optimizer, data, target, reward, criterion
                )
                
                # Accumulate loss
                if batch_loss != float('inf'):
                    total_loss += batch_loss
                    batch_count += 1
                
                # Update learning rate scheduler if it's step-based
                if self.scheduler is not None and isinstance(self.scheduler, 
                        (torch.optim.lr_scheduler.StepLR, torch.optim.lr_scheduler.MultiStepLR)):
                    self.scheduler.step()
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"\nCUDA out of memory error in batch {batch_idx}")
                    # Clear memory
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Reduce batch size for next iterations
                    prev_batch_size = self.config['training']['batch_size']
                    self.config['training']['batch_size'] = max(1, prev_batch_size // 2)
                    print(f"Reducing batch size from {prev_batch_size} to {self.config['training']['batch_size']}")
                    
                    # Log batch size change
                    self.history['batch_size_changes'].append({
                        'epoch': epoch,
                        'batch': batch_idx,
                        'old_size': prev_batch_size,
                        'new_size': self.config['training']['batch_size']
                    })
                    
                    # Skip this batch
                    continue
                else:
                    # Extract tensor shapes from the error message for debugging
                    import re
                    shape_matches = re.findall(r'tensor a \((\d+)\) must match the size of tensor b \((\d+)\)', str(e))
                    if shape_matches:
                        shape_a, shape_b = shape_matches[0]
                        self.tensor_shapes[f"epoch{epoch}_batch{batch_idx}"] = {
                            "shape_a": int(shape_a),
                            "shape_b": int(shape_b),
                            "error": str(e)
                        }
                        print(f"\nTensor shape mismatch: {shape_a} vs {shape_b}")
                    
                    # Log the error and continue
                    print(f"\nError in batch {batch_idx}: {e}")
                    continue
            
            # Log every 10 batches
            if batch_idx % 10 == 0:
                print_gpu_memory_status()
        
        # Return average loss
        return total_loss / batch_count if batch_count > 0 else float('inf')
    
    def validate(self, dataloader):
        """
        Validate the model on the validation set with tensor shape fixes.
        
        Args:
            dataloader (DataLoader): Validation data loader
            
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        criterion = torch.nn.MSELoss()
        all_targets = []
        all_outputs = []
        
        # Track batch count for proper averaging
        batch_count = 0
        
        with torch.no_grad():
            for data, target, reward in tqdm(dataloader, desc="Validation"):
                try:
                    # Move data to device
                    data = data.to(self.device)
                    target = target.to(self.device)
                    reward = reward.to(self.device)
                    
                    # Forward pass
                    output, predicted_reward = self.model(data, external_reward=reward)
                    
                    # Check output and target shapes
                    if output.size(-1) != target.size(-1):
                        output, target = fix_tensor_dimensions(output, target, dim=-1)
                    
                    # Ensure predicted_reward and reward have compatible shapes
                    if predicted_reward.size(-1) != reward.size(-1):
                        predicted_reward, reward = fix_tensor_dimensions(predicted_reward, reward, dim=-1)
                    
                    # Calculate loss
                    task_loss = criterion(output, target)
                    reward_loss = criterion(predicted_reward, reward)
                    loss = task_loss + 0.5 * reward_loss
                    
                    # Store predictions and targets
                    all_outputs.append(output.cpu().numpy())
                    all_targets.append(target.cpu().numpy())
                    
                    # Accumulate loss
                    total_loss += loss.item()
                    batch_count += 1
                
                except RuntimeError as e:
                    # Extract tensor shapes from the error message for debugging
                    import re
                    shape_matches = re.findall(r'tensor a \((\d+)\) must match the size of tensor b \((\d+)\)', str(e))
                    if shape_matches:
                        shape_a, shape_b = shape_matches[0]
                        self.tensor_shapes[f"validation"] = {
                            "shape_a": int(shape_a),
                            "shape_b": int(shape_b),
                            "error": str(e)
                        }
                        print(f"\nTensor shape mismatch in validation: {shape_a} vs {shape_b}")
                    
                    print(f"Error during validation: {e}")
                    continue
        
        # Return average loss and predictions
        try:
            return (total_loss / batch_count if batch_count > 0 else float('inf'), 
                   np.vstack(all_outputs) if all_outputs else np.array([]), 
                   np.vstack(all_targets) if all_targets else np.array([]))
        except Exception as e:
            print(f"Error computing validation results: {e}")
            return float('inf'), np.array([]), np.array([])
    
    def train(self, num_epochs=None, checkpoint_dir=None):
        """
        Train the model with OpenAI-enhanced learning.
        
        Args:
            num_epochs (int, optional): Number of epochs to train for
            checkpoint_dir (str, optional): Directory to save checkpoints
            
        Returns:
            dict: Training history
        """
        # Set number of epochs
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']
        
        # Create checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(self.config['general']['checkpoint_dir'], 
                                        f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save tensor shapes file for debugging
        tensor_shapes_file = os.path.join(checkpoint_dir, "tensor_shapes.json")
        
        # Save configuration
        config_file = os.path.join(checkpoint_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Prepare data
        train_dataloader, val_dataloader, test_dataloader, data_info = self.prepare_data()
        
        # Extract price data for market analysis
        price_data = np.array([])
        for _, target, _ in train_dataloader:
            if len(price_data) == 0:
                price_data = target.numpy().flatten()
            else:
                price_data = np.concatenate([price_data, target.numpy().flatten()])
        
        # Initial market analysis
        if self.config.get('use_llm_market_analysis', True) and self.llm is not None:
            try:
                initial_analysis = self.llm.analyze_market_patterns(price_data)
                print("\nInitial Market Analysis:")
                print(f"Market Regime: {initial_analysis.get('market_regime', 'Unknown')}")
                print(f"Technical Patterns: {', '.join(initial_analysis.get('technical_patterns', ['None']))}")
                print(f"Directional Bias: {initial_analysis.get('directional_bias', 'Neutral')}")
                
                self.history['market_analyses'].append(initial_analysis)
            except Exception as e:
                print(f"Warning: Initial market analysis failed: {e}")
        
        # Training loop
        best_val_loss = float('inf')
        patience = self.config['training'].get('patience', 10)
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            try:
                # Train one epoch
                train_loss = self.train_epoch(train_dataloader, epoch)
                
                # Validate
                val_loss, val_outputs, val_targets = self.validate(val_dataloader)
                
                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                
                # Log memory usage
                if torch.cuda.is_available():
                    memory_stats = {
                        'epoch': epoch,
                        'allocated': torch.cuda.memory_allocated(0) / 1024**2,  # MB
                        'reserved': torch.cuda.memory_reserved(0) / 1024**2,    # MB
                        'max_allocated': torch.cuda.max_memory_allocated(0) / 1024**2  # MB
                    }
                    self.history['memory_usage'].append(memory_stats)
                
                # Update neuromodulation parameters
                self.update_neuromodulation_params(epoch, val_loss, price_data)
                
                # Update learning rate with epoch-based schedulers
                if self.scheduler is not None and isinstance(self.scheduler, 
                        (torch.optim.lr_scheduler.CosineAnnealingLR, 
                         torch.optim.lr_scheduler.ExponentialLR)):
                    self.scheduler.step()
                
                # Print progress
                print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                # Check for improvement
                if val_loss < best_val_loss:
                    improvement = (best_val_loss - val_loss) / best_val_loss * 100
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save checkpoint if improved
                    checkpoint_path = os.path.join(checkpoint_dir, f"model_best_epoch_{epoch}.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'config': self.config,
                        'data_info': data_info,
                        'history': self.history,
                        'tensor_shapes': self.tensor_shapes
                    }, checkpoint_path)
                    
                    print(f"Saved best model checkpoint to {checkpoint_path} (improved by {improvement:.2f}%)")
                else:
                    patience_counter += 1
                    print(f"No improvement. Patience: {patience_counter}/{patience}")
                    
                    if patience_counter >= patience:
                        print(f"Early stopping triggered after {epoch} epochs")
                        break
                
                # Periodic checkpoint
                if epoch % 10 == 0 or epoch == num_epochs:
                    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'config': self.config,
                        'data_info': data_info,
                        'history': self.history,
                        'tensor_shapes': self.tensor_shapes
                    }, checkpoint_path)
                    
                    print(f"Saved checkpoint to {checkpoint_path}")
                    
                    # Save tensor shapes for debugging
                    with open(tensor_shapes_file, 'w') as f:
                        json.dump(self.tensor_shapes, f, indent=2)
                    
            except Exception as e:
                print(f"Error during epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                
                # Try to save current state
                emergency_path = os.path.join(checkpoint_dir, f"emergency_checkpoint_epoch_{epoch}.pt")
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'history': self.history,
                        'tensor_shapes': self.tensor_shapes
                    }, emergency_path)
                    print(f"Saved emergency checkpoint to {emergency_path}")
                    
                    # Save tensor shapes for debugging
                    with open(tensor_shapes_file, 'w') as f:
                        json.dump(self.tensor_shapes, f, indent=2)
                except Exception as e2:
                    print(f"Failed to save emergency checkpoint: {e2}")
                
                # Give time for any background processes to complete
                time.sleep(5)
                
                # Clear memory
                torch.cuda.empty_cache()
                gc.collect()
                continue
        
        # Save final results
        try:
            # Evaluate on test set
            test_loss, test_outputs, test_targets = self.validate(test_dataloader)
            print(f"\nTest Loss: {test_loss:.6f}")
            
            # Save final model
            final_path = os.path.join(checkpoint_dir, "model_final.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'test_loss': test_loss,
                'config': self.config,
                'data_info': data_info,
                'history': self.history,
                'tensor_shapes': self.tensor_shapes
            }, final_path)
            
            print(f"Saved final model to {final_path}")
            
            # Save tensor shapes for debugging
            with open(tensor_shapes_file, 'w') as f:
                json.dump(self.tensor_shapes, f, indent=2)
                
        except Exception as e:
            print(f"Error during final testing: {e}")
            traceback.print_exc()
            
            # Try to save final model anyway
            final_path = os.path.join(checkpoint_dir, "model_final_partial.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'history': self.history,
                'tensor_shapes': self.tensor_shapes
            }, final_path)
            
            print(f"Saved partial final model to {final_path}")
            
            # Save tensor shapes for debugging
            with open(tensor_shapes_file, 'w') as f:
                json.dump(self.tensor_shapes, f, indent=2)
        
        return self.history
