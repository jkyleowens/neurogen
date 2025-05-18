"""
Utility functions for the improved GRU trading model.

This module contains helper functions for model evaluation, performance analysis,
and result saving to avoid circular dependencies between improved_main.py and
improved_neurogru.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import traceback
import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Try to import cupy, fall back to numpy if not available
try:
    import cupy as cp
    use_cupy = True
except ImportError:
    import numpy as np
    cp = np  # Use numpy as cp
    use_cupy = False
    print("CuPy not available, using NumPy instead.")

def evaluate_model(model, W_out, b_out, X_test, y_test):
    """
    Evaluate the improved model with detailed performance analysis
    
    Parameters:
    model: ImprovedNeuroGRU model
    W_out: Output weights
    b_out: Output bias
    X_test: Test input sequences
    y_test: Test target labels
    
    Returns:
    y_true: True labels
    y_pred: Predicted labels
    confidences: Prediction confidences
    """
    try:
        # Update output weights if needed
        if model.hidden_dim != W_out.shape[1]:
            print(f"Updating output weights to match hidden dimension: {model.hidden_dim}")
            W_out = cp.random.randn(3, model.hidden_dim) * 0.1 / cp.sqrt(model.hidden_dim)
            b_out = cp.zeros(3)
        
        print(f"Evaluating on {len(X_test)} test samples...")
        
        # Put network in evaluation mode (stable inference)
        model.set_learning_phase('convergence')
        
        # Temporarily disable growth for stable inference
        original_growth_state = model.enable_growth
        model.enable_growth = False
        
        # Tracking variables
        y_true = []
        y_pred = []
        confidences = []
        
        # Batch testing for efficiency
        batch_size = 32
        
        for batch_start in range(0, len(X_test), batch_size):
            batch_end = min(batch_start + batch_size, len(X_test))
            print(f"  Processing test samples {batch_start} to {batch_end-1}")
            
            for i in range(batch_start, batch_end):
                try:
                    # Process sequence
                    seq = X_test[i]
                    
                    # Reset state for each sequence
                    model.reset_state()
                    
                    # Process full sequence
                    for t in range(len(seq)):
                        x_t = seq[t]
                        # Use inference mode
                        h = model.forward(x_t, inference_mode=True)
                    
                    # Ensure output layer compatibility
                    if h.shape[0] != W_out.shape[1]:
                        W_out = cp.random.randn(3, h.shape[0]) * 0.1 / cp.sqrt(h.shape[0])
                        b_out = cp.zeros(3)
                    
                    # Generate prediction
                    logits = W_out @ h + b_out
                    
                    # Implement manual softmax since that was an issue
                    exp_logits = cp.exp(logits - cp.max(logits))
                    probs = exp_logits / cp.sum(exp_logits)
                    
                    # Record results
                    pred = int(cp.argmax(probs))
                    label = int(cp.argmax(y_test[i]))
                    confidence = float(cp.max(probs))
                    
                    y_pred.append(pred)
                    y_true.append(label)
                    confidences.append(confidence)
                    
                except Exception as e:
                    print(f"Error evaluating sample {i}: {str(e)}")
                    continue
        
        # Restore original growth state
        model.enable_growth = original_growth_state
        
        # Calculate performance metrics
        if len(y_true) > 0:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            confidences = np.array(confidences)
            
            # Core metrics
            accuracy = np.mean(y_true == y_pred)
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Print results
            print("\n====== Evaluation Results ======")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision (macro): {precision:.4f}")
            print(f"Recall (macro): {recall:.4f}")
            print(f"F1 Score (macro): {f1:.4f}")
            
            # Create evaluation visualization
            try:
                class_names = ['Buy', 'Sell', 'Hold']
                plt.figure(figsize=(10, 5))
                
                plt.subplot(1, 2, 1)
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title('Confusion Matrix')
                plt.colorbar()
                tick_marks = np.arange(len(class_names))
                plt.xticks(tick_marks, class_names, rotation=45)
                plt.yticks(tick_marks, class_names)
                
                # Add text annotations to confusion matrix
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(j, i, format(cm[i, j], 'd'),
                                horizontalalignment="center",
                                color="white" if cm[i, j] > cm.max() / 2 else "black")
                
                plt.subplot(1, 2, 2)
                plt.hist(confidences, bins=10)
                plt.title('Prediction Confidence')
                plt.xlabel('Confidence')
                plt.ylabel('Count')
                
                plt.tight_layout()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plt.savefig(f'evaluation_results_{timestamp}.png')
                print(f"Evaluation visualization saved to evaluation_results_{timestamp}.png")
            except Exception as e:
                print(f"Error creating evaluation visualization: {str(e)}")
            
            # Create a results dictionary
            results = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'confusion_matrix': cm.tolist(),
                'class_names': class_names
            }
            
            return results
        
        else:
            print("No valid predictions could be made")
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
    except Exception as e:
        print(f"Critical error during evaluation: {e}")
        traceback.print_exc()
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}

def analyze_model_performance(model, W_out, b_out, X_test, y_test, original_df):
    """
    Analyze model performance with financial metrics
    
    Parameters:
    model: ImprovedNeuroGRU model
    W_out: Output weights
    b_out: Output bias
    X_test: Test input sequences
    y_test: Test target labels
    original_df: Original price dataframe
    
    Returns:
    Dictionary of performance metrics
    """
    try:
        print("Analyzing model performance...")
        
        # Get network statistics
        network_stats = model.get_network_stats()
        
        # Calculate basic trading metrics
        # This is a simplified version - in a real implementation, you would
        # calculate more sophisticated financial metrics
        
        # Create a simple trading simulation
        returns = simulate_trading(model, W_out, b_out, X_test, y_test, original_df)
        
        # Compile performance metrics
        performance_metrics = {
            'network_stats': network_stats,
            'financial_metrics': {
                'simulated_return': returns,
                'sharpe_ratio': calculate_sharpe_ratio(returns)
            }
        }
        
        # Print summary
        print(f"Network size: {network_stats['hidden_dim']} neurons")
        print(f"E/I ratio: {network_stats['ei_ratio']:.2f}")
        print(f"Simulated return: {returns:.2f}%")
        
        return performance_metrics
        
    except Exception as e:
        print(f"Error analyzing model performance: {e}")
        traceback.print_exc()
        return {}

def simulate_trading(model, W_out, b_out, X_test, y_test, original_df, initial_capital=10000):
    """
    Simulate trading based on model predictions
    
    This is a simplified trading simulation for demonstration purposes.
    """
    try:
        # Extract price data
        if len(original_df) < len(X_test):
            print("Warning: Original dataframe is shorter than test set")
            return 0.0
            
        # Use the last portion of the dataframe that corresponds to the test set
        prices = original_df['Close'].values[-len(X_test):]
        
        # Initialize portfolio
        cash = initial_capital
        shares = 0
        positions = []
        
        # Simulate trading
        for i in range(len(X_test)):
            # Get model prediction
            seq = X_test[i]
            model.reset_state()
            
            for t in range(len(seq)):
                h = model.forward(seq[t], inference_mode=True)
                
            # Generate prediction
            logits = W_out @ h + b_out
            probs = model.softmax(logits)
            pred = cp.argmax(probs)
            
            # Execute trade based on prediction
            current_price = prices[i]
            
            if pred == 0:  # Buy
                if cash > 0:
                    # Buy as many shares as possible
                    new_shares = cash // current_price
                    shares += new_shares
                    cash -= new_shares * current_price
            elif pred == 1:  # Sell
                if shares > 0:
                    # Sell all shares
                    cash += shares * current_price
                    shares = 0
            # For hold (pred == 2), do nothing
            
            # Track portfolio value
            portfolio_value = cash + shares * current_price
            positions.append(portfolio_value)
        
        # Calculate return
        if len(positions) > 0:
            final_return = (positions[-1] / initial_capital - 1) * 100
        else:
            final_return = 0.0
            
        return final_return
        
    except Exception as e:
        print(f"Error in trading simulation: {e}")
        return 0.0

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio from returns"""
    if returns == 0:
        return 0.0
        
    # Convert percentage return to decimal
    returns_decimal = returns / 100
    
    # Simplified Sharpe calculation
    excess_return = returns_decimal - risk_free_rate
    
    # Assume some volatility if we don't have a time series
    # In a real implementation, you would calculate this from the return series
    volatility = abs(returns_decimal) * 0.5
    
    if volatility == 0:
        return 0.0
        
    sharpe = excess_return / volatility
    
    return sharpe

def save_results(model, W_out, b_out, training_metrics, test_results, performance_metrics, args):
    """
    Save model results and metrics to files
    
    Parameters:
    model: ImprovedNeuroGRU model
    W_out: Output weights
    b_out: Output bias
    training_metrics: Dictionary of training metrics
    test_results: Dictionary of test results
    performance_metrics: Dictionary of performance metrics
    args: Command line arguments
    """
    try:
        # Create results directory if it doesn't exist
        results_dir = "model_results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics to JSON
        metrics = {
            'training': training_metrics,
            'test': test_results,
            'performance': performance_metrics,
            'parameters': {
                'ticker': args.ticker,
                'start_date': args.start,
                'end_date': args.end,
                'sequence_length': args.seq_len,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'multi_horizon': args.multi_horizon
            }
        }
        
        # Convert numpy arrays to lists for JSON serialization
        metrics_json = json_serialize(metrics)
        
        # Save metrics
        metrics_file = os.path.join(results_dir, f"metrics_{args.ticker}_{timestamp}.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics_json, f, indent=2)
            
        print(f"Results saved to {metrics_file}")
        
        # Save model summary
        summary_file = os.path.join(results_dir, f"model_summary_{args.ticker}_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write(f"ImprovedNeuroGRU Model Summary\n")
            f.write(f"===========================\n\n")
            f.write(f"Ticker: {args.ticker}\n")
            f.write(f"Date Range: {args.start} to {args.end}\n")
            f.write(f"Sequence Length: {args.seq_len}\n\n")
            
            # Write network statistics
            network_stats = model.get_network_stats()
            f.write(f"Network Statistics:\n")
            f.write(f"  Hidden Dimension: {network_stats['hidden_dim']}\n")
            f.write(f"  Excitatory Neurons: {network_stats['excitatory_count']}\n")
            f.write(f"  Inhibitory Neurons: {network_stats['inhibitory_count']}\n")
            f.write(f"  E/I Ratio: {network_stats['ei_ratio']:.2f}\n\n")
            
            # Write performance metrics
            f.write(f"Performance Metrics:\n")
            if 'accuracy' in test_results:
                f.write(f"  Accuracy: {test_results['accuracy']:.4f}\n")
                f.write(f"  Precision: {test_results['precision']:.4f}\n")
                f.write(f"  Recall: {test_results['recall']:.4f}\n")
                f.write(f"  F1 Score: {test_results['f1']:.4f}\n\n")
            
            # Write financial metrics
            if 'financial_metrics' in performance_metrics:
                fin_metrics = performance_metrics['financial_metrics']
                f.write(f"Financial Metrics:\n")
                f.write(f"  Simulated Return: {fin_metrics['simulated_return']:.2f}%\n")
                f.write(f"  Sharpe Ratio: {fin_metrics['sharpe_ratio']:.2f}\n")
            
        print(f"Model summary saved to {summary_file}")
        
        return True
        
    except Exception as e:
        print(f"Error saving results: {e}")
        traceback.print_exc()
        return False

def json_serialize(obj):
    """Convert numpy/cupy arrays to lists for JSON serialization"""
    if isinstance(obj, dict):
        return {k: json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_serialize(item) for item in obj]
    elif isinstance(obj, (np.ndarray, cp.ndarray)):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
        return obj.item()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)

