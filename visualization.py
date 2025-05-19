"""
Visualization Tools for Brain-Inspired Neural Networks

This module provides specialized visualization functions for monitoring and analyzing
the behavior of neuromorphic computing systems, with particular focus on:

1. Neural activity patterns (neuromodulatory dynamics, persistent memory states)
2. Learning processes (training curves, weight adaptation)
3. Financial decision metrics (trading performance, risk-adjusted returns)

The visualizations are designed to highlight the relationships between biological
neural principles and their application to financial time series analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta

# Configure visualization aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Create custom neuroscience-inspired color palettes
DOPAMINE_CMAP = LinearSegmentedColormap.from_list(
    'dopamine', ['#F5F7F7', '#89CFF0', '#0047AB', '#00008B'], N=256)
SEROTONIN_CMAP = LinearSegmentedColormap.from_list(
    'serotonin', ['#F5F7F7', '#FFD580', '#FFA500', '#FF4500'], N=256)
NOREPINEPHRINE_CMAP = LinearSegmentedColormap.from_list(
    'norepinephrine', ['#F5F7F7', '#98FB98', '#32CD32', '#006400'], N=256)
ACETYLCHOLINE_CMAP = LinearSegmentedColormap.from_list(
    'acetylcholine', ['#F5F7F7', '#DDA0DD', '#BA55D3', '#800080'], N=256)

# Financial visualization palette
FINANCIAL_PALETTE = {
    'price': '#1f77b4',          # Blue
    'prediction': '#ff7f0e',     # Orange
    'equity': '#2ca02c',         # Green
    'profit': '#2ca02c',         # Green
    'loss': '#d62728',           # Red
    'buy': '#9467bd',            # Purple
    'sell': '#8c564b',           # Brown
    'hold': '#7f7f7f',           # Gray
    'volume': '#17becf',         # Cyan
    'volatility': '#bcbd22',     # Olive
}


def plot_training_curves(train_losses, val_losses, save_path=None, show=False, metrics=None):
    """
    Visualize training and validation loss curves with optional additional metrics.
    
    This visualization helps track learning progress and identify potential issues
    like overfitting or training instability in the neuromorphic architecture.
    
    Args:
        train_losses (list): Training loss values per epoch
        val_losses (list): Validation loss values per epoch
        save_path (str, optional): Path to save the figure
        show (bool): Whether to display the figure interactively
        metrics (dict, optional): Additional metrics to plot (e.g., direction accuracy)
    
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1] if metrics else [1])
    
    # Loss curves plot
    ax1 = fig.add_subplot(gs[0])
    epochs = np.arange(1, len(train_losses) + 1)
    
    # Plot training and validation losses with gradient shading for depth
    ax1.plot(epochs, train_losses, label='Training Loss', color='#3498db', linewidth=2)
    ax1.fill_between(epochs, train_losses, alpha=0.1, color='#3498db')
    
    ax1.plot(epochs, val_losses, label='Validation Loss', color='#e74c3c', linewidth=2)
    ax1.fill_between(epochs, val_losses, alpha=0.1, color='#e74c3c')
    
    # Find the best validation loss epoch
    best_epoch = np.argmin(val_losses) + 1
    best_loss = min(val_losses)
    ax1.scatter(best_epoch, best_loss, c='#e74c3c', s=100, 
               label=f'Best Validation Loss: {best_loss:.4f} (Epoch {best_epoch})', 
               zorder=5, edgecolor='white')
    
    # Add neuromorphic learning annotation
    if len(train_losses) >= 3:
        # Calculate early vs late phase learning rates
        early_rate = (train_losses[2] - train_losses[0]) / 2
        late_idx = len(train_losses) // 2
        late_rate = (train_losses[-1] - train_losses[late_idx]) / (len(train_losses) - late_idx)
        
        if abs(early_rate) > abs(late_rate) * 2:  # If early learning is much faster
            ax1.annotate('Rapid Early Learning\n(Hippocampal Phase)', 
                       xy=(3, train_losses[2]), 
                       xytext=(5, train_losses[2] - (max(train_losses) - min(train_losses))*0.2),
                       arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
            
            ax1.annotate('Slower Consolidation\n(Neocortical Phase)', 
                       xy=(len(train_losses)-5, train_losses[-5]), 
                       xytext=(len(train_losses)-15, train_losses[-5] - (max(train_losses) - min(train_losses))*0.15),
                       arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss Curves', fontsize=16, pad=20)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Format y-axis to show more precision
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    
    # Plot additional metrics if provided
    if metrics:
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        
        for name, values in metrics.items():
            if len(values) == len(epochs):  # Make sure the metric aligns with epochs
                ax2.plot(epochs, values, label=name, linewidth=2)
        
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Metric Value')
        ax2.set_title('Additional Metrics', fontsize=14)
        ax2.legend(loc='lower right')
        ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def visualize_neuromodulators(neurotransmitter_levels, save_path=None, show=False, time_steps=None):
    """
    Visualize the dynamics of neuromodulatory systems in the neural architecture.
    
    This visualization reveals how the model's internal neuromodulators respond
    to different market conditions, similar to how brain neuromodulatory systems
    adapt to changing environmental stimuli.
    
    Args:
        neurotransmitter_levels (dict): Dictionary containing neurotransmitter levels
            with keys like 'dopamine', 'serotonin', 'norepinephrine', 'acetylcholine'
        save_path (str, optional): Path to save the figure
        show (bool): Whether to display the figure interactively
        time_steps (list, optional): Time points corresponding to neurotransmitter samples
    
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Extract neurotransmitter data
    neurotransmitters = {}
    
    # Handle both single time point and time series data
    if isinstance(next(iter(neurotransmitter_levels.values())), (list, np.ndarray)):
        # Time series data
        for key, values in neurotransmitter_levels.items():
            if isinstance(values[0], (torch.Tensor, np.ndarray)):
                neurotransmitters[key] = [v.item() if hasattr(v, 'item') else v for v in values]
            else:
                neurotransmitters[key] = values
    else:
        # Single time point - convert to list for consistent handling
        for key, value in neurotransmitter_levels.items():
            if hasattr(value, 'item'):
                neurotransmitters[key] = [value.item()]
            else:
                neurotransmitters[key] = [value]
    
    # Create figure with neuromorphic-inspired layout
    fig = plt.figure(figsize=(14, 10))
    
    # Define grid layout
    if len(list(neurotransmitters.values())[0]) > 1:  # Time series
        gs = GridSpec(3, 2, height_ratios=[1, 2, 1])
    else:  # Single time point
        gs = GridSpec(2, 2)
    
    # Color maps for different neurotransmitters
    cmap_dict = {
        'dopamine': DOPAMINE_CMAP,
        'serotonin': SEROTONIN_CMAP, 
        'norepinephrine': NOREPINEPHRINE_CMAP,
        'acetylcholine': ACETYLCHOLINE_CMAP
    }
    
    # Descriptive titles with neurobiological context
    title_dict = {
        'dopamine': 'Dopamine (Reward Prediction)',
        'serotonin': 'Serotonin (Risk Assessment)',
        'norepinephrine': 'Norepinephrine (Attention/Volatility)',
        'acetylcholine': 'Acetylcholine (Memory Formation)'
    }
    
    # Plot individual neurotransmitters
    for i, (nt_name, values) in enumerate(neurotransmitters.items()):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col])
        
        if len(values) > 1:  # Time series plot
            x = time_steps if time_steps is not None else np.arange(len(values))
            ax.plot(x, values, linewidth=2, color=cmap_dict.get(nt_name, 'blue')(0.8))
            ax.fill_between(x, 0, values, alpha=0.3, color=cmap_dict.get(nt_name, 'blue')(0.5))
            
            # Add rolling average for trend visualization
            window = min(len(values) // 5, 20)
            if window > 1:
                rolling_avg = pd.Series(values).rolling(window=window).mean()
                ax.plot(x, rolling_avg, linewidth=1.5, color='white', alpha=0.8, 
                       linestyle='--', label=f'{window}-point Moving Avg')
            
            # Identify key points (peaks, troughs)
            if len(values) > 10:
                peak_idx = np.argmax(values)
                trough_idx = np.argmin(values)
                
                ax.scatter(x[peak_idx], values[peak_idx], color='white', edgecolor='black', 
                          s=100, zorder=5, label=f'Peak: {values[peak_idx]:.3f}')
                ax.scatter(x[trough_idx], values[trough_idx], color='black', edgecolor='white', 
                          s=100, zorder=5, label=f'Trough: {values[trough_idx]:.3f}')
            
            ax.legend(loc='upper right', framealpha=0.7)
            
        else:  # Single value visualization as gauge
            # Create a simple gauge visualization
            value = values[0]
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            # Create gradient background
            for j in range(100):
                alpha = j/100
                ax.axvspan(j/100, (j+1)/100, color=cmap_dict.get(nt_name, 'blue')(alpha), alpha=0.8)
            
            # Add needle
            scaled_value = min(max(value, 0), 1)  # Ensure value is between 0 and 1
            ax.plot([0.5, scaled_value], [0, 0.5], color='black', linewidth=3)
            ax.scatter(scaled_value, 0.5, s=150, color='black', zorder=5)
            
            # Add value text
            ax.text(0.5, 0.7, f"{value:.3f}", horizontalalignment='center', 
                   fontsize=24, fontweight='bold')
            
            # Remove axes for cleaner look
            ax.axis('off')
        
        ax.set_title(title_dict.get(nt_name, nt_name.capitalize()), fontsize=14, pad=10)
        
        # If time series, add appropriate labels
        if len(values) > 1:
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Level')
            ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add interaction plot if time series data available
    if len(list(neurotransmitters.values())[0]) > 1 and len(neurotransmitters) >= 2:
        # Get two most interesting neurotransmitters (e.g., dopamine and norepinephrine)
        nt1 = 'dopamine' if 'dopamine' in neurotransmitters else list(neurotransmitters.keys())[0]
        nt2 = 'norepinephrine' if 'norepinephrine' in neurotransmitters else list(neurotransmitters.keys())[1]
        
        # Create interaction plot
        ax_interact = fig.add_subplot(gs[2, :])
        x = time_steps if time_steps is not None else np.arange(len(neurotransmitters[nt1]))
        
        # Plot both neurotransmitters
        l1 = ax_interact.plot(x, neurotransmitters[nt1], label=title_dict.get(nt1, nt1.capitalize()),
                             color=cmap_dict.get(nt1, 'blue')(0.8), linewidth=2)
        
        # Create second y-axis for the second neurotransmitter
        ax2 = ax_interact.twinx()
        l2 = ax2.plot(x, neurotransmitters[nt2], label=title_dict.get(nt2, nt2.capitalize()),
                     color=cmap_dict.get(nt2, 'green')(0.8), linewidth=2, linestyle='--')
        
        # Find periods of correlation/anticorrelation
        corr = np.corrcoef(neurotransmitters[nt1], neurotransmitters[nt2])[0, 1]
        corr_text = f"Correlation: {corr:.3f}"
        
        # Annotate regions of strong correlation if detected
        window = min(len(x) // 3, 20)
        if window > 5:
            rolling_corr = pd.Series(neurotransmitters[nt1]).rolling(window).corr(
                pd.Series(neurotransmitters[nt2]))
            
            # Find strongest correlation and anti-correlation regions
            max_corr_idx = rolling_corr.dropna().abs().idxmax()
            if not pd.isna(max_corr_idx):
                corr_val = rolling_corr[max_corr_idx]
                ax_interact.axvspan(max_corr_idx - window//2, max_corr_idx + window//2, 
                                   alpha=0.2, color='gray')
                
                if abs(corr_val) > 0.5:  # Only annotate strong correlations
                    annotation = "Synchronized" if corr_val > 0 else "Opposing"
                    ax_interact.annotate(f"{annotation}\nActivity", 
                                       xy=(max_corr_idx, neurotransmitters[nt1][max_corr_idx]),
                                       xytext=(max_corr_idx, max(neurotransmitters[nt1]) * 0.8),
                                       arrowprops=dict(arrowstyle="->"),
                                       bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        
        # Add legends
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax_interact.legend(lines, labels, loc='upper left')
        
        # Add correlation info
        ax_interact.text(0.02, 0.05, corr_text, transform=ax_interact.transAxes, 
                        fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
        ax_interact.set_xlabel('Time Steps')
        ax_interact.set_ylabel(title_dict.get(nt1, nt1.capitalize()))
        ax2.set_ylabel(title_dict.get(nt2, nt2.capitalize()))
        ax_interact.set_title('Neuromodulator Interaction Analysis', fontsize=14)
        ax_interact.grid(True, linestyle='--', alpha=0.5)
    
    # Add overall title
    plt.suptitle('Neuromodulatory Dynamics in Brain-Inspired Neural Network', fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Neuromodulator visualization saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_trading_performance(true_prices, pred_prices, trade_history, equity_curve, 
                          initial_capital, save_path=None, show=False, dates=None):
    """
    Visualize trading performance with decision points and equity curve.
    
    This comprehensive visualization shows how the neural network's predictions
    translate to trading decisions and financial outcomes, helping assess both
    the prediction accuracy and trading strategy effectiveness.
    
    Args:
        true_prices (array-like): Actual price series
        pred_prices (array-like): Predicted price series
        trade_history (list): List of tuples (action, index, price)
        equity_curve (list): Equity value over time
        initial_capital (float): Initial trading capital
        save_path (str, optional): Path to save the figure
        show (bool): Whether to display the figure interactively
        dates (list, optional): List of dates corresponding to price points
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Prepare figure layout
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(3, 1, height_ratios=[2, 1, 1])
    
    # Generate x-axis values
    if dates is not None:
        x = dates
    else:
        x = np.arange(len(true_prices))
    
    # Price and prediction subplot
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x, true_prices, label='Actual Price', color=FINANCIAL_PALETTE['price'], linewidth=2)
    ax1.plot(x, pred_prices, label='Predicted Price', color=FINANCIAL_PALETTE['prediction'], linewidth=1.5, alpha=0.8)
    
    # Highlight areas where prediction direction matches actual direction
    for i in range(1, len(true_prices)):
        true_direction = true_prices[i] > true_prices[i-1]
        pred_direction = pred_prices[i] > pred_prices[i-1]
        if true_direction == pred_direction:
            color = 'green' if true_direction else 'lightgreen'
            ax1.axvspan(x[i-1], x[i], alpha=0.1, color=color)
        else:
            ax1.axvspan(x[i-1], x[i], alpha=0.1, color='mistyrose')
    
    # Extract and plot buy/sell signals
    buys = [(idx, price) for action, idx, price in trade_history if action == 'buy']
    sells = [(idx, price) for action, idx, price in trade_history if action == 'sell']
    
    buy_indices = [idx for idx, _ in buys]
    buy_prices = [price for _, price in buys]
    
    sell_indices = [idx for idx, _ in sells]
    sell_prices = [price for _, price in sells]
    
    # Plot buy/sell markers
    if buy_indices:
        buy_x = [x[i] for i in buy_indices]
        ax1.scatter(buy_x, buy_prices, color=FINANCIAL_PALETTE['buy'], s=100, marker='^', 
                   label='Buy', zorder=5, edgecolor='white')
    
    if sell_indices:
        sell_x = [x[i] for i in sell_indices]
        ax1.scatter(sell_x, sell_prices, color=FINANCIAL_PALETTE['sell'], s=100, marker='v', 
                   label='Sell', zorder=5, edgecolor='white')
    
    # Annotate profitable trades
    if len(buys) > 0 and len(sells) > 0:
        for i, ((buy_idx, buy_price), (sell_idx, sell_price)) in enumerate(zip(buys, sells)):
            if sell_idx > buy_idx:  # Ensure we're matching the correct pairs
                profit = sell_price - buy_price
                profit_pct = (profit / buy_price) * 100
                
                if abs(profit_pct) > 2:  # Only annotate significant trades
                    color = FINANCIAL_PALETTE['profit'] if profit > 0 else FINANCIAL_PALETTE['loss']
                    ax1.plot([x[buy_idx], x[sell_idx]], [buy_price, sell_price], 
                            color=color, linestyle='--', alpha=0.7)
                    
                    if i % 3 == 0:  # Annotate every third trade to avoid clutter
                        mid_idx = (buy_idx + sell_idx) // 2
                        ax1.annotate(f"{profit_pct:.1f}%", 
                                   xy=(x[mid_idx], (buy_price + sell_price) / 2),
                                   color=color, fontweight='bold',
                                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.7))
    
    # Calculate and display prediction accuracy
    correct_dirs = sum(1 for i in range(1, len(true_prices)) 
                       if (true_prices[i] > true_prices[i-1]) == (pred_prices[i] > pred_prices[i-1]))
    direction_accuracy = correct_dirs / (len(true_prices) - 1) * 100
    
    title = f"Price Prediction and Trading Decisions\nDirection Accuracy: {direction_accuracy:.2f}%"
    ax1.set_title(title, fontsize=16, pad=10)
    ax1.set_ylabel('Price')
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Add dynamic annotations based on prediction trends
    if len(true_prices) > 30:
        # Identify key prediction events
        max_err_idx = np.argmax(np.abs(true_prices - pred_prices))
        largest_swing_idx = np.argmax(np.abs(np.diff(true_prices)))
        
        # Annotate largest prediction error
        ax1.annotate("Largest\nPrediction Error", 
                   xy=(x[max_err_idx], true_prices[max_err_idx]),
                   xytext=(x[max_err_idx+5], true_prices[max_err_idx] * 1.05),
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        
        # Annotate largest price swing
        if largest_swing_idx < len(true_prices) - 1:
            ax1.annotate("Largest\nPrice Swing", 
                       xy=(x[largest_swing_idx], true_prices[largest_swing_idx]),
                       xytext=(x[largest_swing_idx-5], true_prices[largest_swing_idx] * 0.95),
                       arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2"),
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    
    # Equity curve subplot
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(x[:len(equity_curve)], equity_curve, label='Portfolio Value', 
            color=FINANCIAL_PALETTE['equity'], linewidth=2)
    
    # Fill area under equity curve
    ax2.fill_between(x[:len(equity_curve)], initial_capital, equity_curve, 
                    where=(np.array(equity_curve) >= initial_capital), 
                    color=FINANCIAL_PALETTE['profit'], alpha=0.3)
    ax2.fill_between(x[:len(equity_curve)], initial_capital, equity_curve, 
                    where=(np.array(equity_curve) < initial_capital), 
                    color=FINANCIAL_PALETTE['loss'], alpha=0.3)
    
    # Calculate and show returns
    final_return = (equity_curve[-1] / initial_capital - 1) * 100
    ax2.axhline(y=initial_capital, color='gray', linestyle='--', 
               label=f'Initial: ${initial_capital:,.2f}')
    
    # Add return annotation
    if final_return > 0:
        color = FINANCIAL_PALETTE['profit']
        marker = '▲'
    else:
        color = FINANCIAL_PALETTE['loss']
        marker = '▼'
    
    ax2.annotate(f"Return: {marker} {final_return:.2f}%  Final: ${equity_curve[-1]:,.2f}", 
               xy=(0.02, 0.85), xycoords='axes fraction',
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.7),
               fontsize=12, color=color)
    
    # Calculate drawdown
    drawdown = np.zeros_like(equity_curve)
    peak = equity_curve[0]
    for i in range(len(equity_curve)):
        if equity_curve[i] > peak:
            peak = equity_curve[i]
        drawdown[i] = (peak - equity_curve[i]) / peak * 100
    
    # Find max drawdown point
    max_dd_idx = np.argmax(drawdown)
    max_dd = drawdown[max_dd_idx]
    
    # Annotate max drawdown if significant
    if max_dd > 5:
        ax2.annotate(f"Max Drawdown: {max_dd:.2f}%", 
                   xy=(x[max_dd_idx], equity_curve[max_dd_idx]),
                   xytext=(x[max_dd_idx], equity_curve[max_dd_idx] * 0.9),
                   arrowprops=dict(arrowstyle="->"),
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=FINANCIAL_PALETTE['loss'], alpha=0.7))
    
    ax2.set_title('Portfolio Equity Curve', fontsize=14)
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left')
    
    # Drawdown subplot
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.fill_between(x[:len(drawdown)], 0, drawdown, color=FINANCIAL_PALETTE['loss'], alpha=0.5)
    ax3.plot(x[:len(drawdown)], drawdown, color=FINANCIAL_PALETTE['loss'], linewidth=1.5)
    
    # Annotate drawdown regions
    threshold = max(5, max_dd/2)  # Adaptive threshold based on max drawdown
    
    # Find contiguous regions of significant drawdown
    regions = []
    in_region = False
    start_idx = 0
    
    for i in range(len(drawdown)):
        if drawdown[i] > threshold and not in_region:
            in_region = True
            start_idx = i
        elif drawdown[i] <= threshold and in_region:
            in_region = False
            if i - start_idx > 5:  # Only consider regions that last more than 5 time steps
                regions.append((start_idx, i-1))
    
    # If still in a region at the end
    if in_region and len(drawdown) - start_idx > 5:
        regions.append((start_idx, len(drawdown)-1))
    
    # Annotate significant drawdown regions
    if regions:
        for i, (start, end) in enumerate(regions):
            if i <= 2:  # Limit to 3 annotations to avoid clutter
                mid = (start + end) // 2
                region_dd = drawdown[start:end+1].mean()
                ax3.annotate(f"{region_dd:.1f}% Avg", 
                           xy=(x[mid], drawdown[mid]),
                           xytext=(x[mid], drawdown[mid] * 0.7),
                           arrowprops=dict(arrowstyle="->"),
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    
    # Add severity zones
    ax3.axhline(y=10, color='orange', linestyle='--', alpha=0.7)
    ax3.axhline(y=20, color='red', linestyle='--', alpha=0.7)
    ax3.text(x[0], 10, "Moderate", verticalalignment='bottom', color='orange')
    ax3.text(x[0], 20, "Severe", verticalalignment='bottom', color='red')
    
    ax3.set_title('Drawdown Analysis', fontsize=14)
    ax3.set_xlabel('Time' if dates is None else 'Date')
    ax3.set_ylabel('Drawdown (%)')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.invert_yaxis()  # Invert so that drawdowns go down
    
    # Format x-axis if dates are provided
    if dates is not None:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add overall title
    plt.suptitle('Neural Trading System Performance Analysis', fontsize=18, y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trading performance visualization saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def visualize_neuron_activations(model, sample_data, device, save_path=None, show=False):
    """
    Visualize activation patterns in the brain-inspired neural network.
    
    This function creates a visualization of neuron activations in response to 
    financial data, revealing how different regions of the network specialize
    in detecting particular market patterns.
    
    Args:
        model: Brain-inspired neural network model
        sample_data: Sample input data tensor
        device: Device to run the model on
        save_path (str, optional): Path to save the figure
        show (bool): Whether to display the figure interactively
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Process sample data through the model
    model.eval()
    with torch.no_grad():
        # Get the first batch item if it's a batch
        if isinstance(sample_data, tuple):
            data = sample_data[0].to(device)
        else:
            data = sample_data.to(device)
        
        # Make sure we have a batch dimension
        if data.dim() == 2:
            data = data.unsqueeze(0)
        
        # Forward pass through model
        output, _ = model(data)
        
        # Get hidden states if available
        hidden_states = None
        if hasattr(model, 'hidden_states') and model.hidden_states:
            hidden_states = model.hidden_states
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    if hidden_states:
        # Extract the last time step's hidden states for all layers
        num_layers = len(hidden_states[0])
        gs = GridSpec(2, num_layers)
        
        # Plot activation heatmaps for each layer
        for layer in range(num_layers):
            # Get the hidden state for this layer (last time step)
            hidden = hidden_states[-1][layer].cpu().numpy()
            
            # Create heatmap
            ax = fig.add_subplot(gs[0, layer])
            im = ax.imshow(hidden, aspect='auto', cmap='viridis')
            
            ax.set_title(f'Layer {layer+1} Activations')
            ax.set_xlabel('Neuron Index')
            ax.set_ylabel('Batch Item')
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
            
            # Plot activation distribution
            ax = fig.add_subplot(gs[1, layer])
            sns.histplot(hidden.flatten(), bins=30, kde=True, ax=ax)
            
            ax.set_title(f'Layer {layer+1} Activation Distribution')
            ax.set_xlabel('Activation Value')
            ax.set_ylabel('Frequency')
            
        plt.suptitle('Neuronal Activation Patterns', fontsize=16)
        
    else:
        # If hidden states are not available, create a simpler visualization
        plt.text(0.5, 0.5, "Detailed activation visualization requires access to model's hidden states", 
                horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Neuron activation visualization saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_prediction_error_analysis(true_values, predictions, save_path=None, show=False):
    """
    Analyze prediction errors and their patterns in relation to market conditions.
    
    This visualization helps identify systematic biases or weaknesses in the 
    neural architecture's predictive capabilities under different market scenarios.
    
    Args:
        true_values (array-like): Actual values
        predictions (array-like): Predicted values
        save_path (str, optional): Path to save the figure
        show (bool): Whether to display the figure interactively
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Calculate errors
    errors = np.array(predictions) - np.array(true_values)
    
    # Create percentage errors for relative analysis
    pct_errors = errors / np.abs(np.array(true_values)) * 100
    
    # Calculate basic statistics
    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(pct_errors))
    bias = np.mean(errors)
    
    # Create figure
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(3, 2)
    
    # Scatter plot of predictions vs true values
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(true_values, predictions, alpha=0.5, color='blue')
    
    # Add perfect prediction line
    min_val = min(min(true_values), min(predictions))
    max_val = max(max(true_values), max(predictions))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    
    ax1.set_title('Predictions vs Actual Values')
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predictions')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Add stats annotation
    stats_text = f"MAE: {mae:.4f}\nMAPE: {mape:.2f}%\nBias: {bias:.4f}"
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Residual plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(true_values, errors, alpha=0.5, color='green')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    
    ax2.set_title('Residual Plot')
    ax2.set_xlabel('Actual Values')
    ax2.set_ylabel('Error (Prediction - Actual)')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Histogram of errors
    ax3 = fig.add_subplot(gs[1, 0])
    sns.histplot(errors, bins=30, kde=True, ax=ax3, color='purple')
    
    ax3.set_title('Error Distribution')
    ax3.set_xlabel('Error')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # QQ plot to check normality of errors
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate quantiles
    sorted_errors = np.sort(errors)
    n = len(sorted_errors)
    quantiles = np.arange(1, n + 1) / (n + 1)
    theoretical_quantiles = np.quantile(np.random.normal(0, np.std(errors), 1000), quantiles)
    
    ax4.scatter(theoretical_quantiles, sorted_errors, alpha=0.5, color='orange')
    ax4.plot([min(theoretical_quantiles), max(theoretical_quantiles)], 
            [min(theoretical_quantiles), max(theoretical_quantiles)], 'r--', alpha=0.7)
    
    ax4.set_title('Q-Q Plot (Error Normality Check)')
    ax4.set_xlabel('Theoretical Quantiles')
    ax4.set_ylabel('Sample Quantiles')
    ax4.grid(True, linestyle='--', alpha=0.5)
    
    # Error vs. prediction horizon
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(np.abs(errors), color='blue', alpha=0.7)
    
    # Add moving average for trend
    window = min(len(errors) // 10, 20)
    if window > 1:
        error_ma = pd.Series(np.abs(errors)).rolling(window=window).mean()
        ax5.plot(error_ma, color='red', linewidth=2, label=f'{window}-point Moving Avg')
    
    ax5.set_title('Absolute Error Over Time')
    ax5.set_xlabel('Time Index')
    ax5.set_ylabel('Absolute Error')
    ax5.legend()
    ax5.grid(True, linestyle='--', alpha=0.5)
    
    # Error patterns related to market conditions
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Calculate returns
    returns = np.diff(true_values) / true_values[:-1] * 100
    returns = np.append(0, returns)  # Add 0 at beginning to match length
    
    # Create scatter plot colored by return magnitude
    scatter = ax6.scatter(returns, np.abs(errors), c=np.abs(returns), 
                        cmap='viridis', alpha=0.7, edgecolor='k', linewidth=0.5)
    
    # Add color bar
    plt.colorbar(scatter, ax=ax6, label='Absolute Return (%)')
    
    # Add trend line
    if len(returns) > 2:
        z = np.polyfit(np.abs(returns), np.abs(errors), 1)
        p = np.poly1d(z)
        ax6.plot(np.sort(np.abs(returns)), p(np.sort(np.abs(returns))), 
                'r--', alpha=0.7, label=f'Trend: y = {z[0]:.4f}x + {z[1]:.4f}')
    
    ax6.set_title('Error vs Market Volatility')
    ax6.set_xlabel('Return (%)')
    ax6.set_ylabel('Absolute Error')
    ax6.legend()
    ax6.grid(True, linestyle='--', alpha=0.5)
    
    # Add overall title
    plt.suptitle('Prediction Error Analysis', fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction error analysis saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig