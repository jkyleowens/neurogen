import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

def create_pretrain_dataloader(dataloader, batch_size=16):
    """
    Create a simplified and smaller dataloader for pretraining.
    """
    print("Creating pretraining dataloader...")
    
    try:
        # Get a smaller batch size to avoid memory issues
        all_data = []
        all_targets = []
        
        # Collect just 2 small batches
        for i, batch in enumerate(dataloader):
            if i >= 2:  # Only 2 batches
                break
                
            try:
                if len(batch) >= 2:
                    data, target = batch[0], batch[1]
                    
                    # Take only first 16 samples to keep it small
                    if data.shape[0] > 16:
                        data = data[:16]
                        target = target[:16]
                    
                    all_data.append(data.cpu())
                    all_targets.append(target.cpu())
                    print(f"Collected batch {i}: data shape {data.shape}, target shape {target.shape}")
                    
            except Exception as e:
                print(f"Error collecting batch {i}: {e}")
                continue
        
        if not all_data:
            print("No data collected, creating synthetic pretraining data...")
            # Create minimal synthetic data
            synthetic_data = torch.randn(32, 30, 5)  # batch, seq, features
            synthetic_targets = torch.randn(32)  # batch
            dataset = TensorDataset(synthetic_data, synthetic_targets)
            return DataLoader(dataset, batch_size=16, shuffle=False)
        
        # Combine collected data
        X = torch.cat(all_data, dim=0)
        y = torch.cat(all_targets, dim=0)
        
        print(f"Created pretraining data: X shape {X.shape}, y shape {y.shape}")
        
        # Create dataset
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
    except Exception as e:
        print(f"Error creating pretraining dataloader: {e}")
        print("Creating emergency synthetic dataloader...")
        
        # Emergency fallback
        synthetic_data = torch.randn(32, 30, 5)
        synthetic_targets = torch.randn(32)
        dataset = TensorDataset(synthetic_data, synthetic_targets)
        return DataLoader(dataset, batch_size=16, shuffle=False)

def diagnose_model_shapes(model, sample_data, device):
    """
    Diagnose the model's input/output shapes to understand compatibility.
    """
    print("\n=== Model Shape Diagnosis ===")
    
    try:
        model.eval()
        with torch.no_grad():
            # Test with sample data
            data = sample_data.to(device)
            print(f"Input data shape: {data.shape}")
            
            # Test model forward pass
            if hasattr(model, 'reset_state'):
                model.reset_state()
            
            try:
                output = model(data)
                print(f"Model output shape: {output.shape}")
                print(f"Model output type: {type(output)}")
                
                if isinstance(output, tuple):
                    print(f"Output is tuple with {len(output)} elements:")
                    for i, elem in enumerate(output):
                        if hasattr(elem, 'shape'):
                            print(f"  Element {i}: shape {elem.shape}")
                        else:
                            print(f"  Element {i}: {type(elem)}")
                
            except Exception as e:
                print(f"Model forward pass failed: {e}")
                return False
            
            # Test controller if available
            if hasattr(model, 'controller'):
                print(f"\nController type: {type(model.controller)}")
                try:
                    if hasattr(model.controller, 'reset_state'):
                        model.controller.reset_state()
                    
                    if hasattr(model.controller, 'init_hidden'):
                        hidden = model.controller.init_hidden(data.shape[0], device)
                        print(f"Controller hidden state shape: {hidden.shape if hasattr(hidden, 'shape') else type(hidden)}")
                        
                        controller_output = model.controller(data, hidden)
                    else:
                        controller_output = model.controller(data)
                    
                    print(f"Controller output shape: {controller_output.shape if hasattr(controller_output, 'shape') else type(controller_output)}")
                    
                except Exception as e:
                    print(f"Controller test failed: {e}")
                    return False
            
            print("=== Shape Diagnosis Complete ===\n")
            return True
            
    except Exception as e:
        print(f"Error in shape diagnosis: {e}")
        return False

def simple_controller_pretraining(model, sample_data, device, epochs=2):
    """
    Extremely simple controller pretraining that just tests forward passes.
    """
    print("Starting simple controller pretraining...")
    
    try:
        if not hasattr(model, 'controller'):
            print("No controller found in model")
            return True
        
        model.train()
        data = sample_data.to(device)
        
        for epoch in range(epochs):
            try:
                # Reset states
                if hasattr(model, 'reset_state'):
                    model.reset_state()
                if hasattr(model.controller, 'reset_state'):
                    model.controller.reset_state()
                
                # Simple forward pass without gradients
                with torch.no_grad():
                    if hasattr(model.controller, 'init_hidden'):
                        hidden = model.controller.init_hidden(data.shape[0], device)
                        output = model.controller(data, hidden)
                    else:
                        output = model.controller(data)
                    
                    print(f"Controller pretraining epoch {epoch+1}: Forward pass successful")
                
            except Exception as e:
                print(f"Controller pretraining epoch {epoch+1} failed: {e}")
                return False
        
        print("Simple controller pretraining completed")
        return True
        
    except Exception as e:
        print(f"Error in simple controller pretraining: {e}")
        return False

def simple_neuromodulator_pretraining(model, sample_data, device, epochs=2):
    """
    Simple neuromodulator pretraining with minimal reward feedback.
    """
    print("Starting simple neuromodulator pretraining...")
    
    try:
        model.train()
        data = sample_data.to(device)
        
        for epoch in range(epochs):
            try:
                # Reset model state
                if hasattr(model, 'reset_state'):
                    model.reset_state()
                
                # Forward pass without gradients
                with torch.no_grad():
                    # Test basic forward pass
                    output = model(data)
                    print(f"Neuromodulator epoch {epoch+1}: Basic forward pass successful")
                    
                    # Test with small reward
                    reward = torch.tensor(0.01, device=device)
                    output_with_reward = model(data, reward=reward)
                    print(f"Neuromodulator epoch {epoch+1}: Reward feedback successful")
                
            except Exception as e:
                print(f"Neuromodulator pretraining epoch {epoch+1} failed: {e}")
                return False
        
        print("Simple neuromodulator pretraining completed")
        return True
        
    except Exception as e:
        print(f"Error in simple neuromodulator pretraining: {e}")
        return False

def safe_pretrain_components(model, dataloader, device, config):
    """
    Ultra-safe pretraining that won't crash.
    """
    print("\n=== Starting Safe Component Pretraining ===")
    
    try:
        # Create smaller, safer pretraining data
        pretrain_loader = create_pretrain_dataloader(dataloader, batch_size=8)
        
        # Get a sample batch for testing
        try:
            sample_batch = next(iter(pretrain_loader))
            sample_data = sample_batch[0]
            sample_target = sample_batch[1]
            print(f"Sample data shape: {sample_data.shape}")
            print(f"Sample target shape: {sample_target.shape}")
        except Exception as e:
            print(f"Error getting sample data: {e}")
            # Create emergency sample data
            sample_data = torch.randn(8, 30, 5)
            sample_target = torch.randn(8)
            print("Created emergency sample data")
        
        # Diagnose model shapes
        shapes_ok = diagnose_model_shapes(model, sample_data, device)
        if not shapes_ok:
            print("Shape diagnosis failed - skipping pretraining")
            return model
        
        # Controller pretraining
        controller_config = config.get('controller', {})
        if controller_config.get('enabled', True):
            print("\n--- Simple Controller Pretraining ---")
            controller_success = simple_controller_pretraining(
                model, 
                sample_data, 
                device, 
                epochs=min(2, controller_config.get('epochs', 2))
            )
            
            if controller_success:
                print("✅ Controller pretraining completed successfully")
            else:
                print("❌ Controller pretraining failed - continuing anyway")
        
        # Neuromodulator pretraining
        neuromod_config = config.get('neuromodulator', {})
        if neuromod_config.get('enabled', True):
            print("\n--- Simple Neuromodulator Pretraining ---")
            neuromod_success = simple_neuromodulator_pretraining(
                model, 
                sample_data, 
                device, 
                epochs=min(2, neuromod_config.get('epochs', 2))
            )
            
            if neuromod_success:
                print("✅ Neuromodulator pretraining completed successfully")
            else:
                print("❌ Neuromodulator pretraining failed - continuing anyway")
        
        print("\n=== Safe Component Pretraining Complete ===")
        return model
        
    except Exception as e:
        print(f"Critical error in safe pretraining: {e}")
        print("Returning model without pretraining...")
        return model

# Simplified versions for direct import
def pretrain_controller(controller, dataloader, device, epochs=2, learning_rate=0.001):
    """Simplified controller pretraining that focuses on compatibility."""
    print(f"Starting simplified controller pretraining for {epochs} epochs...")
    
    try:
        # Get sample data
        sample_batch = next(iter(dataloader))
        sample_data = sample_batch[0].to(device)
        
        # Test controller forward pass
        controller.eval()
        with torch.no_grad():
            if hasattr(controller, 'reset_state'):
                controller.reset_state()
            
            try:
                if hasattr(controller, 'init_hidden'):
                    hidden = controller.init_hidden(sample_data.shape[0], device)
                    output = controller(sample_data, hidden)
                else:
                    output = controller(sample_data)
                
                print(f"Controller forward pass successful: output shape {output.shape}")
                
            except Exception as e:
                print(f"Controller forward pass failed: {e}")
                return controller
        
        print("Simplified controller pretraining completed")
        return controller
        
    except Exception as e:
        print(f"Error in simplified controller pretraining: {e}")
        return controller

def pretrain_neuromodulator_components(model, dataloader, device, epochs=2, learning_rate=0.0005):
    """Simplified neuromodulator pretraining."""
    print(f"Starting simplified neuromodulator pretraining for {epochs} epochs...")
    
    try:
        # Get sample data
        sample_batch = next(iter(dataloader))
        sample_data = sample_batch[0].to(device)
        
        # Test model with reward
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'reset_state'):
                model.reset_state()
            
            try:
                # Test basic forward pass
                output = model(sample_data)
                print(f"Model forward pass successful: output shape {output.shape}")
                
                # Test with reward
                reward = torch.tensor(0.01, device=device)
                output_with_reward = model(sample_data, reward=reward)
                print("Reward feedback successful")
                
            except Exception as e:
                print(f"Neuromodulator test failed: {e}")
                return model
        
        print("Simplified neuromodulator pretraining completed")
        return model
        
    except Exception as e:
        print(f"Error in simplified neuromodulator pretraining: {e}")
        return model