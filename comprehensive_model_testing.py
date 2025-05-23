#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive Model Testing and Analysis

This module provides extensive testing, analysis, and visualization
capabilities for the Brain-Inspired Neural Network.
"""

import os
import cupy as cp
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelTester:
    """Comprehensive model testing and analysis class."""
    
    def __init__(self, model, test_loader, device, config=None):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.config = config or {}
        
        # Storage for results
        self.predictions = []
        self.targets = []
        self.losses = []
        self.batch_times = []
        self.neuron_activities = []
        self.neuromodulator_levels = []
        
        # Metrics storage
        self.metrics = {}
        self.detailed_metrics = {}
        
    def run_comprehensive_test(self, save_dir='test_results'):
        """Run comprehensive testing and analysis."""
        print("🧠 Starting Comprehensive Model Testing...")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Basic Performance Testing
        print("📊 Running basic performance tests...")
        self.test_basic_performance()
        
        # 2. Detailed Analysis
        print("🔍 Performing detailed analysis...")
        self.analyze_predictions()
        
        # 3. Neuron Activity Analysis
        print("🧪 Analyzing neuron activity...")
        self.analyze_neuron_activity()
        
        # 4. Generate Visualizations
        print("📈 Creating visualizations...")
        self.create_all_visualizations(save_dir)
        
        # 5. Generate Report
        print("📝 Generating comprehensive report...")
        self.generate_report(save_dir)
        
        print(f"✅ Testing complete! Results saved to {save_dir}")
        return self.metrics
    
    def test_basic_performance(self):
        """Test basic model performance."""
        self.model.eval()
        criterion = nn.MSELoss()
        
        total_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(self.test_loader, desc="Testing")):
                try:
                    # Handle different batch formats
                    if len(batch_data) == 2:
                        data, target = batch_data
                    elif len(batch_data) == 3:
                        data, target, _ = batch_data
                    else:
                        continue
                    
                    # Move to device
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Reset model state
                    self.model.reset_state()
                    
                    # Time the forward pass
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    
                    if torch.cuda.is_available():
                        start_time.record()
                    
                    # Forward pass
                    output = self.model(data)
                    
                    if torch.cuda.is_available():
                        end_time.record()
                        torch.cuda.synchronize()
                        batch_time = start_time.elapsed_time(end_time)
                        self.batch_times.append(batch_time)
                    
                    # Handle output shapes
                    if output.dim() == 3:
                        output = output[:, -1, :]
                    if target.dim() == 1:
                        target = target.unsqueeze(1)
                    
                    # Ensure compatible shapes
                    if output.shape != target.shape:
                        min_features = min(output.shape[-1], target.shape[-1])
                        output = output[..., :min_features]
                        target = target[..., :min_features]
                    
                    # Calculate loss
                    loss = criterion(output, target)
                    
                    # Store results
                    self.predictions.append(output.cpu().numpy())
                    self.targets.append(target.cpu().numpy())
                    self.losses.append(loss.item())
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                    # Collect neuron activity if available
                    if hasattr(self.model, 'pathway_activity'):
                        self.neuron_activities.append(
                            self.model.pathway_activity.cpu().numpy()
                        )
                    
                    # Collect neuromodulator levels if available
                    if hasattr(self.model, 'neuromodulator_levels'):
                        self.neuromodulator_levels.append(
                            dict(self.model.neuromodulator_levels)
                        )
                
                except Exception as e:
                    print(f"Error in test batch {batch_idx}: {e}")
                    continue
        
        # Store basic metrics
        self.metrics['test_loss'] = total_loss / max(1, batch_count)
        self.metrics['num_batches'] = batch_count
        self.metrics['avg_batch_time'] = cp.mean(self.batch_times) if self.batch_times else 0
    
    def analyze_predictions(self):
        """Analyze prediction quality in detail."""
        if not self.predictions or not self.targets:
            print("No predictions to analyze")
            return
        
        # Convert to numpy arrays
        predictions = cp.vstack(self.predictions)
        targets = cp.vstack(self.targets)
        
        # Flatten for easier analysis
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Calculate comprehensive metrics
        mse = mean_squared_error(target_flat, pred_flat)
        rmse = cp.sqrt(mse)
        mae = mean_absolute_error(target_flat, pred_flat)
        
        # R² score
        try:
            r2 = r2_score(target_flat, pred_flat)
        except:
            r2 = 0.0
        
        # Mean Absolute Percentage Error
        mape = cp.mean(cp.abs((target_flat - pred_flat) / (target_flat + 1e-8))) * 100
        
        # Directional accuracy (for time series)
        if len(pred_flat) > 1:
            pred_diff = cp.diff(pred_flat)
            target_diff = cp.diff(target_flat)
            direction_accuracy = cp.mean((pred_diff > 0) == (target_diff > 0)) * 100
        else:
            direction_accuracy = 0.0
        
        # Error distribution analysis
        errors = pred_flat - target_flat
        error_std = cp.std(errors)
        error_mean = cp.mean(errors)
        
        # Quantile analysis
        error_quantiles = cp.percentile(cp.abs(errors), [25, 50, 75, 90, 95, 99])
        
        # Store detailed metrics
        self.detailed_metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape),
            'direction_accuracy': float(direction_accuracy),
            'error_mean': float(error_mean),
            'error_std': float(error_std),
            'error_quantiles': {
                'q25': float(error_quantiles[0]),
                'q50': float(error_quantiles[1]),
                'q75': float(error_quantiles[2]),
                'q90': float(error_quantiles[3]),
                'q95': float(error_quantiles[4]),
                'q99': float(error_quantiles[5])
            },
            'predictions_stats': {
                'mean': float(cp.mean(pred_flat)),
                'std': float(cp.std(pred_flat)),
                'min': float(cp.min(pred_flat)),
                'max': float(cp.max(pred_flat))
            },
            'targets_stats': {
                'mean': float(cp.mean(target_flat)),
                'std': float(cp.std(target_flat)),
                'min': float(cp.min(target_flat)),
                'max': float(cp.max(target_flat))
            }
        }
        
        # Update main metrics
        self.metrics.update(self.detailed_metrics)
    
    def analyze_neuron_activity(self):
        """Analyze neuron activity patterns."""
        if not self.neuron_activities:
            return
        
        # Convert to numpy array
        activities = cp.array(self.neuron_activities)
        
        # Calculate activity statistics
        mean_activity = cp.mean(activities, axis=0)
        std_activity = cp.std(activities, axis=0)
        max_activity = cp.max(activities, axis=0)
        
        # Calculate active neuron percentage
        active_threshold = 0.1
        active_neurons = cp.mean(activities > active_threshold, axis=0)
        
        self.metrics['neuron_analysis'] = {
            'mean_activity': mean_activity.tolist(),
            'std_activity': std_activity.tolist(),
            'max_activity': max_activity.tolist(),
            'active_neuron_percentage': active_neurons.tolist(),
            'overall_activity': float(cp.mean(mean_activity))
        }
    
    def create_all_visualizations(self, save_dir):
        """Create comprehensive visualizations."""
        # 1. Performance Overview
        self.plot_performance_overview(save_dir)
        
        # 2. Prediction Analysis
        self.plot_prediction_analysis(save_dir)
        
        # 3. Error Analysis
        self.plot_error_analysis(save_dir)
        
        # 4. Neuron Activity
        self.plot_neuron_activity(save_dir)
        
        # 5. Time Series Analysis
        self.plot_time_series_analysis(save_dir)
        
        # 6. Interactive Plotly Dashboard
        self.create_interactive_dashboard(save_dir)
    
    def plot_performance_overview(self, save_dir):
        """Create performance overview plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Overview', fontsize=16, fontweight='bold')
        
        # 1. Predictions vs Targets Scatter
        if self.predictions and self.targets:
            pred_flat = cp.vstack(self.predictions).flatten()
            target_flat = cp.vstack(self.targets).flatten()
            
            axes[0, 0].scatter(target_flat, pred_flat, alpha=0.6, s=20)
            axes[0, 0].plot([target_flat.min(), target_flat.max()], 
                           [target_flat.min(), target_flat.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('Actual Values')
            axes[0, 0].set_ylabel('Predicted Values')
            axes[0, 0].set_title(f'Predictions vs Actual (R² = {self.metrics.get("r2", 0):.3f})')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Loss Distribution
        if self.losses:
            axes[0, 1].hist(self.losses, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 1].axvline(cp.mean(self.losses), color='red', linestyle='--', 
                              label=f'Mean: {cp.mean(self.losses):.4f}')
            axes[0, 1].set_xlabel('Loss')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Loss Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Batch Processing Times
        if self.batch_times:
            axes[0, 2].plot(self.batch_times, color='green', alpha=0.7)
            axes[0, 2].set_xlabel('Batch Index')
            axes[0, 2].set_ylabel('Processing Time (ms)')
            axes[0, 2].set_title(f'Batch Processing Times (Avg: {cp.mean(self.batch_times):.2f}ms)')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Error Distribution
        if self.predictions and self.targets:
            errors = pred_flat - target_flat
            axes[1, 0].hist(errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].axvline(0, color='red', linestyle='--', label='Perfect Prediction')
            axes[1, 0].set_xlabel('Prediction Error')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title(f'Error Distribution (MAE: {self.metrics.get("mae", 0):.4f})')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Metrics Summary
        metrics_text = []
        for key, value in self.detailed_metrics.items():
            if isinstance(value, (int, float)) and key not in ['predictions_stats', 'targets_stats', 'error_quantiles']:
                metrics_text.append(f'{key.upper()}: {value:.4f}')
        
        axes[1, 1].text(0.1, 0.9, '\n'.join(metrics_text[:10]), transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Key Metrics Summary')
        
        # 6. Q-Q Plot for Error Distribution
        if self.predictions and self.targets:
            from scipy import stats
            errors = pred_flat - target_flat
            stats.probplot(errors, dist="norm", plot=axes[1, 2])
            axes[1, 2].set_title('Q-Q Plot: Error Normality')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'performance_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_prediction_analysis(self, save_dir):
        """Create detailed prediction analysis plots."""
        if not self.predictions or not self.targets:
            return
        
        pred_flat = cp.vstack(self.predictions).flatten()
        target_flat = cp.vstack(self.targets).flatten()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Detailed Prediction Analysis', fontsize=16, fontweight='bold')
        
        # 1. Time Series Comparison (first 200 points)
        n_points = min(200, len(pred_flat))
        axes[0, 0].plot(range(n_points), target_flat[:n_points], 'b-', label='Actual', linewidth=2)
        axes[0, 0].plot(range(n_points), pred_flat[:n_points], 'r--', label='Predicted', linewidth=2)
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Values')
        axes[0, 0].set_title('Time Series Comparison (First 200 Points)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residual Plot
        residuals = pred_flat - target_flat
        axes[0, 1].scatter(pred_flat, residuals, alpha=0.6, s=20)
        axes[0, 1].axhline(y=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Error by Magnitude
        target_abs = cp.abs(target_flat)
        error_abs = cp.abs(residuals)
        
        # Bin by target magnitude
        bins = cp.percentile(target_abs, cp.linspace(0, 100, 11))
        bin_centers = []
        bin_errors = []
        
        for i in range(len(bins)-1):
            mask = (target_abs >= bins[i]) & (target_abs < bins[i+1])
            if cp.sum(mask) > 0:
                bin_centers.append((bins[i] + bins[i+1]) / 2)
                bin_errors.append(cp.mean(error_abs[mask]))
        
        axes[1, 0].plot(bin_centers, bin_errors, 'o-', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Target Magnitude')
        axes[1, 0].set_ylabel('Mean Absolute Error')
        axes[1, 0].set_title('Error by Target Magnitude')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Cumulative Error Distribution
        sorted_errors = cp.sort(cp.abs(residuals))
        cumulative_prob = cp.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        
        axes[1, 1].plot(sorted_errors, cumulative_prob, linewidth=2)
        axes[1, 1].set_xlabel('Absolute Error')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Cumulative Error Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add percentile lines
        for p in [50, 90, 95, 99]:
            error_p = cp.percentile(cp.abs(residuals), p)
            axes[1, 1].axvline(error_p, color='red', linestyle='--', alpha=0.7, 
                              label=f'{p}th percentile')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'prediction_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_error_analysis(self, save_dir):
        """Create detailed error analysis plots."""
        if not self.predictions or not self.targets:
            return
        
        pred_flat = cp.vstack(self.predictions).flatten()
        target_flat = cp.vstack(self.targets).flatten()
        errors = pred_flat - target_flat
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Error Analysis', fontsize=16, fontweight='bold')
        
        # 1. Error Distribution with Statistics
        axes[0, 0].hist(errors, bins=50, alpha=0.7, color='lightcoral', edgecolor='black', density=True)
        
        # Overlay normal distribution
        mu, sigma = cp.mean(errors), cp.std(errors)
        x = cp.linspace(errors.min(), errors.max(), 100)
        axes[0, 0].plot(x, (1/(sigma * cp.sqrt(2 * cp.pi))) * cp.exp(-0.5 * ((x - mu) / sigma) ** 2), 
                       'b-', linewidth=2, label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
        
        axes[0, 0].axvline(0, color='red', linestyle='--', label='Zero Error')
        axes[0, 0].set_xlabel('Prediction Error')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Error Distribution vs Normal')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Absolute Error vs Target Value
        axes[0, 1].scatter(target_flat, cp.abs(errors), alpha=0.6, s=20, c='purple')
        axes[0, 1].set_xlabel('Target Value')
        axes[0, 1].set_ylabel('Absolute Error')
        axes[0, 1].set_title('Absolute Error vs Target Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Error Autocorrelation
        if len(errors) > 50:
            from statsmodels.tsa.stattools import acf
            lags = range(min(50, len(errors)//4))
            autocorr = acf(errors, nlags=len(lags)-1, fft=True)
            
            axes[0, 2].plot(lags, autocorr, 'o-', linewidth=2)
            axes[0, 2].axhline(y=0, color='red', linestyle='--')
            axes[0, 2].set_xlabel('Lag')
            axes[0, 2].set_ylabel('Autocorrelation')
            axes[0, 2].set_title('Error Autocorrelation')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Rolling Error Statistics
        window_size = max(10, len(errors) // 50)
        rolling_mae = pd.Series(cp.abs(errors)).rolling(window_size).mean()
        rolling_std = pd.Series(errors).rolling(window_size).std()
        
        axes[1, 0].plot(rolling_mae, label='Rolling MAE', linewidth=2)
        axes[1, 0].plot(rolling_std, label='Rolling Std', linewidth=2)
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Error Metric')
        axes[1, 0].set_title(f'Rolling Error Statistics (Window={window_size})')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Error Percentiles Box Plot
        error_percentiles = [cp.percentile(cp.abs(errors), p) for p in range(0, 101, 10)]
        axes[1, 1].boxplot([cp.abs(errors)], labels=['Absolute Errors'])
        axes[1, 1].set_ylabel('Absolute Error')
        axes[1, 1].set_title('Error Distribution Box Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Error Heatmap (if predictions are 2D)
        if len(cp.vstack(self.predictions).shape) > 1 and cp.vstack(self.predictions).shape[1] > 1:
            error_matrix = cp.abs(cp.vstack(self.predictions) - cp.vstack(self.targets))
            im = axes[1, 2].imshow(error_matrix[:min(50, error_matrix.shape[0])].T, 
                                  aspect='auto', cmap='viridis')
            axes[1, 2].set_xlabel('Time Steps')
            axes[1, 2].set_ylabel('Feature Dimension')
            axes[1, 2].set_title('Error Heatmap (First 50 Steps)')
            plt.colorbar(im, ax=axes[1, 2])
        else:
            # Alternative: Error vs prediction confidence
            axes[1, 2].text(0.5, 0.5, 'Error Heatmap\n(Not applicable for 1D output)', 
                           ha='center', va='center', transform=axes[1, 2].transAxes,
                           fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
            axes[1, 2].set_xlim(0, 1)
            axes[1, 2].set_ylim(0, 1)
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'error_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_neuron_activity(self, save_dir):
        """Plot neuron activity analysis."""
        if not self.neuron_activities:
            return
        
        activities = cp.array(self.neuron_activities)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Neuron Activity Analysis', fontsize=16, fontweight='bold')
        
        # 1. Average Activity per Layer
        mean_activity = cp.mean(activities, axis=0)
        for layer_idx in range(mean_activity.shape[0]):
            axes[0, 0].plot(mean_activity[layer_idx], label=f'Layer {layer_idx}', linewidth=2)
        
        axes[0, 0].set_xlabel('Neuron Index')
        axes[0, 0].set_ylabel('Average Activity')
        axes[0, 0].set_title('Average Neuron Activity by Layer')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Activity Distribution
        all_activities = activities.flatten()
        axes[0, 1].hist(all_activities, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].axvline(cp.mean(all_activities), color='red', linestyle='--', 
                          label=f'Mean: {cp.mean(all_activities):.3f}')
        axes[0, 1].set_xlabel('Activity Level')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Activity Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Activity Heatmap
        im = axes[1, 0].imshow(mean_activity, aspect='auto', cmap='viridis')
        axes[1, 0].set_xlabel('Neuron Index')
        axes[1, 0].set_ylabel('Layer')
        axes[1, 0].set_title('Activity Heatmap (Average)')
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Active Neuron Percentage
        active_threshold = 0.1
        active_percentage = cp.mean(activities > active_threshold, axis=0) * 100
        
        for layer_idx in range(active_percentage.shape[0]):
            axes[1, 1].bar(range(len(active_percentage[layer_idx])), active_percentage[layer_idx], 
                          alpha=0.7, label=f'Layer {layer_idx}')
        
        axes[1, 1].set_xlabel('Neuron Index')
        axes[1, 1].set_ylabel('Active Percentage (%)')
        axes[1, 1].set_title(f'Active Neurons (Threshold > {active_threshold})')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'neuron_activity.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_time_series_analysis(self, save_dir):
        """Create time series specific analysis."""
        if not self.predictions or not self.targets:
            return
        
        pred_flat = cp.vstack(self.predictions).flatten()
        target_flat = cp.vstack(self.targets).flatten()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Time Series Analysis', fontsize=16, fontweight='bold')
        
        # 1. Directional Accuracy Analysis
        if len(pred_flat) > 1:
            pred_diff = cp.diff(pred_flat)
            target_diff = cp.diff(target_flat)
            
            # Calculate rolling directional accuracy
            window = max(10, len(pred_diff) // 20)
            rolling_dir_acc = []
            
            for i in range(window, len(pred_diff)):
                window_pred = pred_diff[i-window:i]
                window_target = target_diff[i-window:i]
                acc = cp.mean((window_pred > 0) == (window_target > 0)) * 100
                rolling_dir_acc.append(acc)
            
            axes[0, 0].plot(range(window, len(pred_diff)), rolling_dir_acc, linewidth=2)
            axes[0, 0].axhline(y=50, color='red', linestyle='--', label='Random Baseline')
            axes[0, 0].set_xlabel('Time Steps')
            axes[0, 0].set_ylabel('Directional Accuracy (%)')
            axes[0, 0].set_title(f'Rolling Directional Accuracy (Window={window})')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Trend Analysis
        def calculate_trend(data, window=20):
            trends = []
            for i in range(window, len(data)):
                slope = cp.polyfit(range(window), data[i-window:i], 1)[0]
                trends.append(slope)
            return cp.array(trends)
        
        if len(pred_flat) > 20:
            pred_trends = calculate_trend(pred_flat)
            target_trends = calculate_trend(target_flat)
            
            axes[0, 1].scatter(target_trends, pred_trends, alpha=0.6, s=20)
            axes[0, 1].plot([target_trends.min(), target_trends.max()], 
                           [target_trends.min(), target_trends.max()], 'r--', lw=2)
            axes[0, 1].set_xlabel('Actual Trend')
            axes[0, 1].set_ylabel('Predicted Trend')
            axes[0, 1].set_title('Trend Prediction Accuracy')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Volatility Analysis
        def calculate_volatility(data, window=20):
            volatilities = []
            for i in range(window, len(data)):
                vol = cp.std(data[i-window:i])
                volatilities.append(vol)
            return cp.array(volatilities)
        
        if len(pred_flat) > 20:
            pred_vol = calculate_volatility(pred_flat)
            target_vol = calculate_volatility(target_flat)
            
            axes[1, 0].plot(target_vol, label='Actual Volatility', linewidth=2)
            axes[1, 0].plot(pred_vol, label='Predicted Volatility', linewidth=2, alpha=0.8)
            axes[1, 0].set_xlabel('Time Steps')
            axes[1, 0].set_ylabel('Volatility')
            axes[1, 0].set_title('Volatility Comparison')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Prediction Confidence Analysis
        pred_confidence = 1 / (1 + cp.abs(pred_flat - target_flat))  # Simple confidence measure
        
        # Bin predictions by confidence
        conf_bins = cp.percentile(pred_confidence, cp.linspace(0, 100, 11))
        bin_accuracies = []
        bin_centers = []
        
        for i in range(len(conf_bins)-1):
            mask = (pred_confidence >= conf_bins[i]) & (pred_confidence < conf_bins[i+1])
            if cp.sum(mask) > 0:
                bin_centers.append((conf_bins[i] + conf_bins[i+1]) / 2)
                bin_accuracy = 1 - cp.mean(cp.abs(pred_flat[mask] - target_flat[mask]) / 
                                         (cp.abs(target_flat[mask]) + 1e-8))
                bin_accuracies.append(max(0, bin_accuracy))
        
        axes[1, 1].plot(bin_centers, bin_accuracies, 'o-', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('Prediction Confidence')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Accuracy by Confidence Level')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'time_series_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_interactive_dashboard(self, save_dir):
        """Create interactive Plotly dashboard."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.express as px
            
            if not self.predictions or not self.targets:
                return
            
            pred_flat = cp.vstack(self.predictions).flatten()
            target_flat = cp.vstack(self.targets).flatten()
            errors = pred_flat - target_flat
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=[
                    'Predictions vs Actual (Interactive)',
                    'Error Distribution',
                    'Time Series Comparison',
                    'Residual Plot',
                    'Loss Over Batches',
                    'Performance Metrics'
                ],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 1. Predictions vs Actual Scatter
            fig.add_trace(
                go.Scatter(
                    x=target_flat[:1000],  # Limit points for performance
                    y=pred_flat[:1000],
                    mode='markers',
                    name='Predictions',
                    marker=dict(size=4, opacity=0.6),
                    hovertemplate='Actual: %{x}<br>Predicted: %{y}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Perfect prediction line
            min_val, max_val = min(target_flat.min(), pred_flat.min()), max(target_flat.max(), pred_flat.max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=1
            )
            
            # 2. Error Distribution
            fig.add_trace(
                go.Histogram(
                    x=errors,
                    nbinsx=50,
                    name='Error Distribution',
                    opacity=0.7
                ),
                row=1, col=2
            )
            
            # 3. Time Series Comparison
            n_points = min(200, len(pred_flat))
            fig.add_trace(
                go.Scatter(
                    x=list(range(n_points)),
                    y=target_flat[:n_points],
                    mode='lines',
                    name='Actual',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(n_points)),
                    y=pred_flat[:n_points],
                    mode='lines',
                    name='Predicted',
                    line=dict(color='red', width=2, dash='dash')
                ),
                row=2, col=1
            )
            
            # 4. Residual Plot
            fig.add_trace(
                go.Scatter(
                    x=pred_flat,
                    y=errors,
                    mode='markers',
                    name='Residuals',
                    marker=dict(size=3, opacity=0.6)
                ),
                row=2, col=2
            )
            
            # Zero line for residuals
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
            
            # 5. Loss Over Batches
            if self.losses:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(self.losses))),
                        y=self.losses,
                        mode='lines+markers',
                        name='Batch Loss',
                        line=dict(color='green')
                    ),
                    row=3, col=1
                )
            
            # 6. Performance Metrics Table
            metrics_data = []
            for key, value in self.detailed_metrics.items():
                if isinstance(value, (int, float)) and key not in ['predictions_stats', 'targets_stats', 'error_quantiles']:
                    metrics_data.append([key.upper(), f"{value:.6f}"])
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Metric', 'Value'],
                               fill_color='paleturquoise',
                               align='left'),
                    cells=dict(values=list(zip(*metrics_data)),
                              fill_color='lavender',
                              align='left')
                ),
                row=3, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=1200,
                title_text="Interactive Model Performance Dashboard",
                title_x=0.5,
                showlegend=True
            )
            
            # Save interactive HTML
            fig.write_html(os.path.join(save_dir, 'interactive_dashboard.html'))
            
        except ImportError:
            print("Plotly not available, skipping interactive dashboard")
        except Exception as e:
            print(f"Error creating interactive dashboard: {e}")
    
    def generate_report(self, save_dir):
        """Generate comprehensive HTML report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Brain-Inspired Neural Network - Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 20px; }}
                h2 {{ color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; }}
                h3 {{ color: #7f8c8d; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric-card {{ background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
                .status-good {{ color: #27ae60; }}
                .status-warning {{ color: #f39c12; }}
                .status-bad {{ color: #e74c3c; }}
                .summary {{ background: #d5e8d4; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }}
                .image-card {{ text-align: center; }}
                .image-card img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #3498db; color: white; }}
                .footer {{ text-align: center; margin-top: 40px; color: #7f8c8d; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🧠 Brain-Inspired Neural Network Test Report</h1>
                
                <div class="summary">
                    <h2>📊 Executive Summary</h2>
                    <p><strong>Test Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Model Type:</strong> Brain-Inspired Neural Network with Neuromodulation</p>
                    <p><strong>Test Samples:</strong> {len(cp.vstack(self.predictions)) if self.predictions else 0}</p>
                    <p><strong>Overall Performance:</strong> 
                        <span class="{'status-good' if self.metrics.get('r2', 0) > 0.7 else 'status-warning' if self.metrics.get('r2', 0) > 0.4 else 'status-bad'}">
                            {'Excellent' if self.metrics.get('r2', 0) > 0.7 else 'Good' if self.metrics.get('r2', 0) > 0.4 else 'Needs Improvement'}
                        </span>
                    </p>
                </div>
                
                <h2>📈 Key Performance Metrics</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value status-{'good' if self.metrics.get('r2', 0) > 0.7 else 'warning' if self.metrics.get('r2', 0) > 0.4 else 'bad'}">{self.metrics.get('r2', 0):.3f}</div>
                        <div class="metric-label">R² Score</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{self.metrics.get('rmse', 0):.4f}</div>
                        <div class="metric-label">RMSE</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{self.metrics.get('mae', 0):.4f}</div>
                        <div class="metric-label">MAE</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{self.metrics.get('mape', 0):.2f}%</div>
                        <div class="metric-label">MAPE</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{self.metrics.get('direction_accuracy', 0):.1f}%</div>
                        <div class="metric-label">Directional Accuracy</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{self.metrics.get('avg_batch_time', 0):.2f}ms</div>
                        <div class="metric-label">Avg Processing Time</div>
                    </div>
                </div>
                
                <h2>📋 Detailed Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
        """
        
        # Add detailed metrics to table
        metric_interpretations = {
            'mse': 'Lower is better. Mean squared prediction error.',
            'rmse': 'Lower is better. Root mean squared error in original units.',
            'mae': 'Lower is better. Mean absolute prediction error.',
            'r2': 'Higher is better (max 1.0). Proportion of variance explained.',
            'mape': 'Lower is better. Mean absolute percentage error.',
            'direction_accuracy': 'Higher is better. Accuracy of trend direction.',
            'error_mean': 'Should be close to 0. Indicates prediction bias.',
            'error_std': 'Lower is better. Consistency of predictions.'
        }
        
        for key, value in self.detailed_metrics.items():
            if isinstance(value, (int, float)) and key in metric_interpretations:
                html_content += f"""
                    <tr>
                        <td><strong>{key.upper()}</strong></td>
                        <td>{value:.6f}</td>
                        <td>{metric_interpretations[key]}</td>
                    </tr>
                """
        
        html_content += """
                </table>
                
                <h2>🎯 Error Analysis</h2>
                <div style="background: #fff3cd; padding: 15px; border-radius: 8px; margin: 20px 0;">
        """
        
        if 'error_quantiles' in self.detailed_metrics:
            error_q = self.detailed_metrics['error_quantiles']
            html_content += f"""
                <h3>Error Distribution Quantiles</h3>
                <ul>
                    <li><strong>25th percentile:</strong> {error_q['q25']:.4f}</li>
                    <li><strong>50th percentile (median):</strong> {error_q['q50']:.4f}</li>
                    <li><strong>75th percentile:</strong> {error_q['q75']:.4f}</li>
                    <li><strong>90th percentile:</strong> {error_q['q90']:.4f}</li>
                    <li><strong>95th percentile:</strong> {error_q['q95']:.4f}</li>
                    <li><strong>99th percentile:</strong> {error_q['q99']:.4f}</li>
                </ul>
            """
        
        html_content += """
                </div>
                
                <h2>🖼️ Visualizations</h2>
                <div class="image-grid">
                    <div class="image-card">
                        <h3>Performance Overview</h3>
                        <img src="performance_overview.png" alt="Performance Overview">
                    </div>
                    <div class="image-card">
                        <h3>Prediction Analysis</h3>
                        <img src="prediction_analysis.png" alt="Prediction Analysis">
                    </div>
                    <div class="image-card">
                        <h3>Error Analysis</h3>
                        <img src="error_analysis.png" alt="Error Analysis">
                    </div>
        """
        
        if self.neuron_activities:
            html_content += """
                    <div class="image-card">
                        <h3>Neuron Activity</h3>
                        <img src="neuron_activity.png" alt="Neuron Activity">
                    </div>
            """
        
        html_content += """
                    <div class="image-card">
                        <h3>Time Series Analysis</h3>
                        <img src="time_series_analysis.png" alt="Time Series Analysis">
                    </div>
                </div>
                
                <h2>🔧 Recommendations</h2>
                <div style="background: #e8f5e8; padding: 20px; border-radius: 8px;">
        """
        
        # Generate recommendations based on metrics
        recommendations = []
        
        if self.metrics.get('r2', 0) < 0.5:
            recommendations.append("📈 Low R² score suggests poor model fit. Consider increasing model complexity or improving data quality.")
        
        if self.metrics.get('error_mean', 0) > 0.01:
            recommendations.append("⚖️ High error mean indicates prediction bias. Model tends to over/under-predict consistently.")
        
        if self.metrics.get('direction_accuracy', 0) < 60:
            recommendations.append("📊 Low directional accuracy. Model struggles with trend prediction. Consider trend-focused loss functions.")
        
        if self.metrics.get('mape', 0) > 20:
            recommendations.append("🎯 High MAPE suggests poor relative accuracy. Consider relative error-based training.")
        
        if len(recommendations) == 0:
            recommendations.append("✅ Model performance is satisfactory across all key metrics!")
        
        for rec in recommendations:
            html_content += f"<p>{rec}</p>"
        
        html_content += """
                </div>
                
                <h2>🧪 Model Architecture Notes</h2>
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px;">
        """
        
        if hasattr(self.model, 'config'):
            config = self.model.config
            html_content += f"""
                <p><strong>Input Size:</strong> {config.get('model', {}).get('input_size', 'N/A')}</p>
                <p><strong>Hidden Size:</strong> {config.get('model', {}).get('hidden_size', 'N/A')}</p>
                <p><strong>Output Size:</strong> {config.get('model', {}).get('output_size', 'N/A')}</p>
                <p><strong>Uses BioGRU:</strong> {config.get('model', {}).get('use_bio_gru', False)}</p>
                <p><strong>Learning Mode:</strong> {config.get('training', {}).get('learning_mode', 'N/A')}</p>
            """
        
        html_content += """
                </div>
                
                <div class="footer">
                    <p>Report generated by Brain-Inspired Neural Network Testing Suite</p>
                    <p>For interactive analysis, open interactive_dashboard.html</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(os.path.join(save_dir, 'test_report.html'), 'w') as f:
            f.write(html_content)
        
        # Save metrics as JSON
        with open(os.path.join(save_dir, 'test_metrics.json'), 'w') as f:
            json.dump({
                'basic_metrics': self.metrics,
                'detailed_metrics': self.detailed_metrics,
                'test_timestamp': datetime.now().isoformat(),
                'model_config': getattr(self.model, 'config', {})
            }, f, indent=2)
        
        print(f"📋 Comprehensive report saved to {os.path.join(save_dir, 'test_report.html')}")


def test_model_comprehensive(model, test_loader, device, config=None, save_dir='test_results'):
    """
    Convenience function to run comprehensive model testing.
    
    Args:
        model: Trained model to test
        test_loader: Test data loader
        device: Computation device
        config: Model configuration (optional)
        save_dir: Directory to save results
        
    Returns:
        dict: Test metrics
    """
    tester = ModelTester(model, test_loader, device, config)
    return tester.run_comprehensive_test(save_dir)


# Usage example for integration with main.py
def add_to_main_py():

    # After training is complete, run comprehensive testing
    print("\\n🧠 Running comprehensive model testing...")
    
    try:
        # Load best model for testing
        best_checkpoint_path = os.path.join('models/checkpoints', 'best_model.pt')
        if os.path.exists(best_checkpoint_path):
            best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(best_checkpoint['model_state_dict'])
            print("Loaded best model for testing")
        
        # Run comprehensive testing
        test_metrics = test_model_comprehensive(
            model=model,
            test_loader=test_loader,
            device=device,
            config=config,
            save_dir='comprehensive_test_results'
        )
        
        print("\\n📊 Test Results Summary:")
        for key, value in test_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
        
        print("\\n📁 Detailed results and visualizations saved to 'comprehensive_test_results/'")
        print("📋 Open 'comprehensive_test_results/test_report.html' for the full report")
        
    except Exception as e:
        print(f"Error during comprehensive testing: {e}")
    pass