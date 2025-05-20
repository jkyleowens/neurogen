import os
import matplotlib.pyplot as plt
import numpy as np
import json

def generate_performance_report(train_loss, val_loss, test_loss, metrics=None, output_dir="docs/"):
    """
    Generate a comprehensive performance report including per-epoch stats, metrics, and plots.

    Args:
        train_loss: Training loss (can be a single value or list)
        val_loss: Validation loss (can be a single value or list)
        test_loss: Final testing loss
        metrics (dict, optional): Additional metrics (e.g., accuracy, MSE)
        output_dir (str): Directory to save the report and plots
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Create a summary file
    summary_path = os.path.join(output_dir, "performance_report.md")
    with open(summary_path, "w") as f:
        f.write("# Performance Report\n\n")
        
        # Per-epoch table if lists were provided
        if isinstance(train_loss, (list, tuple)) and isinstance(val_loss, (list, tuple)):
            f.write("## Training & Validation Loss per Epoch\n\n")
            f.write("| Epoch | Train Loss | Val Loss |")
            
            # Add accuracy columns if available in metrics
            if metrics and 'train_accuracy' in metrics and 'val_accuracy' in metrics:
                f.write(" Train Accuracy | Val Accuracy |")
            
            f.write("\n|:-----:|:----------:|:--------:|")
            
            # Add appropriate number of columns for accuracy
            if metrics and 'train_accuracy' in metrics and 'val_accuracy' in metrics:
                f.write(":-------------:|:-------------:|")
            
            f.write("\n")
            
            # Determine number of epochs
            num_epochs = len(train_loss)
            
            # Write per-epoch metrics
            for i in range(num_epochs):
                line = f"| {i+1} | {train_loss[i]:.6f} | {val_loss[i]:.6f} |"
                
                # Add accuracy values if available
                if metrics and 'train_accuracy' in metrics and 'val_accuracy' in metrics:
                    if i < len(metrics['train_accuracy']) and i < len(metrics['val_accuracy']):
                        line += f" {metrics['train_accuracy'][i]:.2f}% | {metrics['val_accuracy'][i]:.2f}% |"
                
                f.write(line + "\n")
            
            f.write("\n")
        else:
            # Simple summary for single values
            f.write("## Training & Validation Summary\n\n")
            f.write(f"**Training Loss:** {train_loss:.6f}\n\n")
            f.write(f"**Validation Loss:** {val_loss:.6f}\n\n")
            
            # Add accuracy if available
            if metrics:
                if 'train_accuracy' in metrics and not isinstance(metrics['train_accuracy'], (list, tuple)):
                    f.write(f"**Training Accuracy:** {metrics['train_accuracy']:.2f}%\n\n")
                if 'val_accuracy' in metrics and not isinstance(metrics['val_accuracy'], (list, tuple)):
                    f.write(f"**Validation Accuracy:** {metrics['val_accuracy']:.2f}%\n\n")
        
        # Test results
        f.write("## Test Performance\n\n")
        f.write(f"**Test Loss:** {test_loss:.6f}\n\n")
        
        # Include test accuracy prominently if available
        if metrics and 'accuracy' in metrics:
            f.write(f"**Test Accuracy:** {metrics['accuracy']:.2f}%\n\n")
        
        # Include additional metrics
        if metrics:
            f.write("## Additional Metrics\n\n")
            for name, value in metrics.items():
                # Skip lists, already reported metrics, or certain special metrics
                skip_metrics = ['train_accuracy', 'val_accuracy', 'accuracy', 'neuron_health', 
                                'active_neuron_percentage', 'neuromodulator_levels']
                if not isinstance(value, (list, tuple)) and name not in skip_metrics:
                    try:
                        f.write(f"- **{name.upper()}:** {value:.6f}")
                        # Add % for accuracy metrics
                        if 'accuracy' in name.lower():
                            f.write("%")
                        f.write("\n")
                    except (ValueError, TypeError):
                        # Handle non-float metrics
                        f.write(f"- **{name.upper()}:** {value}\n")
        
        # Neural health section if available
        if metrics and ('neuron_health' in metrics or 'active_neuron_percentage' in metrics):
            f.write("\n## Neural Network Health\n\n")
            
            if 'active_neuron_percentage' in metrics:
                f.write(f"**Active Neurons:** {metrics['active_neuron_percentage']}%\n\n")
                
            if 'neuron_health' in metrics:
                f.write(f"**Overall Health Score:** {metrics['neuron_health']:.4f}\n\n")
            
            # Add neuromodulator levels if available
            if 'neuromodulator_levels' in metrics:
                f.write("### Neuromodulator Levels\n\n")
                for name, level in metrics['neuromodulator_levels'].items():
                    f.write(f"- **{name.capitalize()}:** {level:.4f}\n")

    # Generate plots
    # Plot loss curves if we have lists
    if isinstance(train_loss, (list, tuple)) and isinstance(val_loss, (list, tuple)):
        epochs = range(1, len(train_loss) + 1)
        
        # Loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, 'b-', label='Training Loss')
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'loss_curves.png'))
        plt.close()
        
        # Accuracy plot if available
        if metrics and 'train_accuracy' in metrics and 'val_accuracy' in metrics:
            if isinstance(metrics['train_accuracy'], (list, tuple)) and isinstance(metrics['val_accuracy'], (list, tuple)):
                plt.figure(figsize=(10, 6))
                plt.plot(epochs[:len(metrics['train_accuracy'])], metrics['train_accuracy'], 'g-', label='Training Accuracy')
                plt.plot(epochs[:len(metrics['val_accuracy'])], metrics['val_accuracy'], 'm-', label='Validation Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy (%)')
                plt.title('Training and Validation Accuracy')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(plots_dir, 'accuracy_curves.png'))
                plt.close()
    
    # Bar chart for test metrics
    if metrics:
        # Select numeric non-list metrics for display
        display_metrics = {}
        for key, value in metrics.items():
            if not isinstance(value, (list, tuple, dict)) and key not in ['neuromodulator_levels', 'error']:
                try:
                    float_val = float(value)
                    display_metrics[key] = float_val
                except (ValueError, TypeError):
                    continue
        
        if display_metrics:
            plt.figure(figsize=(12, 6))
            
            # Sort by metric name
            keys = sorted(display_metrics.keys())
            values = [display_metrics[k] for k in keys]
            
            # Create bar colors
            colors = ['green' if 'accuracy' in k.lower() else 'red' if ('loss' in k.lower() or 'error' in k.lower()) else 'blue' for k in keys]
            
            # Create bars
            bars = plt.bar(keys, values, color=colors)
            
            # Prettify labels
            pretty_labels = [k.replace('_', ' ').title() for k in keys]
            plt.xticks(range(len(keys)), pretty_labels, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                label_text = f"{height:.2f}"
                if height < 1.0:
                    label_text = f"{height:.4f}"
                
                # Add % for accuracy metrics
                if 'accuracy' in bar.get_label().lower():
                    label_text += "%"
                
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.02 * max(values),
                        label_text, ha='center', va='bottom', rotation=0)
            
            plt.title('Test Performance Metrics')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'test_metrics.png'))
            plt.close()
    
    # Save metrics as JSON for potential future use
    metrics_path = os.path.join(output_dir, "metrics.json")
    
    # Prepare JSON-serializable metrics
    json_metrics = {}
    
    # Handle train/val losses
    if isinstance(train_loss, (list, tuple)):
        json_metrics['train_loss'] = [float(x) for x in train_loss]
    else:
        json_metrics['train_loss'] = float(train_loss)
        
    if isinstance(val_loss, (list, tuple)):
        json_metrics['val_loss'] = [float(x) for x in val_loss]
    else:
        json_metrics['val_loss'] = float(val_loss)
    
    json_metrics['test_loss'] = float(test_loss)
    
    # Add other metrics from the metrics dict
    if metrics:
        for key, value in metrics.items():
            # Convert NumPy arrays to lists
            if isinstance(value, np.ndarray):
                json_metrics[key] = value.tolist()
            # Handle non-serializable types
            elif isinstance(value, (list, tuple)):
                try:
                    json_metrics[key] = [float(x) if isinstance(x, (int, float, np.number)) else str(x) for x in value]
                except (TypeError, ValueError):
                    json_metrics[key] = str(value)
            elif isinstance(value, dict):
                try:
                    # Convert dict values to JSON-serializable
                    json_metrics[key] = {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                                        for k, v in value.items()}
                except (TypeError, ValueError):
                    json_metrics[key] = str(value)
            elif isinstance(value, (int, float, np.number)):
                json_metrics[key] = float(value)
            else:
                json_metrics[key] = str(value)
    
    # Save to file
    try:
        with open(metrics_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save metrics to JSON: {e}")
    
    print(f"Performance report saved to {summary_path}")
    print(f"Performance plots saved to {plots_dir}")