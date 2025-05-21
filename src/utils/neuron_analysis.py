import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_neuron_health(model):
    """
    Analyze neuron health and activity patterns.
    
    Args:
        model: BioGRU model instance
        
    Returns:
        dict: Health statistics
    """
    health_stats = {'layer_health': [], 'active_ratios': [], 'weight_norms': []}
    
    # Skip if model doesn't have layers attribute
    if not hasattr(model, 'layers'):
        return health_stats
    
    for layer_idx, layer in enumerate(model.layers):
        # Skip non-BiologicalGRUCell layers
        if not hasattr(layer, 'neuron_mask'):
            continue
            
        layer_health = []
        active_neurons = 0
        weight_norms = []
        
        for i in range(layer.hidden_size):
            if layer.neuron_mask[i] > 0:
                active_neurons += 1
                # Average health across all components
                neuron_health = (
                    layer.update_gate_neurons[i].health.item() +
                    layer.reset_gate_neurons[i].health.item() +
                    layer.candidate_neurons[i].health.item()
                ) / 3
                layer_health.append(neuron_health)
                
                # Weight norm
                for neuron in [layer.update_gate_neurons[i], layer.reset_gate_neurons[i], layer.candidate_neurons[i]]:
                    weight_norms.append(torch.norm(neuron.weights).item())
        
        health_stats['layer_health'].append(np.mean(layer_health) if layer_health else 0)
        health_stats['active_ratios'].append(active_neurons / max(1, layer.hidden_size))
        health_stats['weight_norms'].append(np.mean(weight_norms) if weight_norms else 0)
    
    return health_stats

def visualize_neuron_health(model, save_dir='neuron_viz'):
    """
    Create visualizations of neuron health and activity.
    
    Args:
        model: BioGRU model
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get health statistics
    health_stats = analyze_neuron_health(model)
    
    # Plot layer health
    if health_stats['layer_health']:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(health_stats['layer_health'])), health_stats['layer_health'])
        plt.xlabel('Layer')
        plt.ylabel('Average Neuron Health')
        plt.title('Layer-wise Neuron Health')
        plt.xticks(range(len(health_stats['layer_health'])))
        plt.ylim(0, 1)
        plt.savefig(os.path.join(save_dir, 'layer_health.png'))
        plt.close()
    
    # Plot active neuron ratios
    if health_stats['active_ratios']:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(health_stats['active_ratios'])), health_stats['active_ratios'])
        plt.xlabel('Layer')
        plt.ylabel('Ratio of Active Neurons')
        plt.title('Layer-wise Active Neuron Ratio')
        plt.xticks(range(len(health_stats['active_ratios'])))
        plt.ylim(0, 1)
        plt.savefig(os.path.join(save_dir, 'active_ratios.png'))
        plt.close()
    
    # Plot weight norms
    if health_stats['weight_norms']:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(health_stats['weight_norms'])), health_stats['weight_norms'])
        plt.xlabel('Layer')
        plt.ylabel('Average Weight Norm')
        plt.title('Layer-wise Weight Magnitudes')
        plt.xticks(range(len(health_stats['weight_norms'])))
        plt.savefig(os.path.join(save_dir, 'weight_norms.png'))
        plt.close()
    
    # Visualize pathway activity if available
    if hasattr(model, 'pathway_activity'):
        plt.figure(figsize=(12, 8))
        for i in range(model.pathway_activity.shape[0]):
            activity = model.pathway_activity[i].cpu().numpy()
            plt.subplot(model.pathway_activity.shape[0], 1, i+1)
            plt.bar(range(len(activity)), activity)
            plt.ylabel(f'Layer {i}')
            plt.title(f'Neuron Activity in Layer {i}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'pathway_activity.png'))
        plt.close()
    
    # Visualize neuromodulator levels if available
    if hasattr(model, 'neuromodulator_levels') and isinstance(model.neuromodulator_levels, dict):
        plt.figure(figsize=(10, 6))
        names = []
        values = []
        for name, value in model.neuromodulator_levels.items():
            names.append(name)
            if isinstance(value, torch.Tensor):
                values.append(value.item())
            else:
                values.append(value)
        
        plt.bar(names, values)
        plt.ylabel('Level')
        plt.title('Neuromodulator Levels')
        plt.ylim(0, 1)
        plt.savefig(os.path.join(save_dir, 'neuromodulator_levels.png'))
        plt.close()
    
    return health_stats