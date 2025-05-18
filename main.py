try:
    import cupy as cp
    use_cupy = True
    # Optional: print("improved_neurogru.py: Using CuPy.")
except ImportError:
    import numpy as cp
    use_cupy = False
    print("CuPy not available in improved_neurogru.py, using NumPy instead.")
    
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import traceback
import pandas as pd
import seaborn as sns
import gc

# Import the improved neural model
from improved_neurogru import ImprovedNeuroGRU

# Import utility functions to avoid circular dependencies
# We will also update evaluate_model from utils.py contextually here
# for the purpose of this optimization discussion, assuming it's called from main.
# from utils import evaluate_model, analyze_model_performance, save_results # Original import

# --- fetch_stock_data and create_sequences remain largely the same ---
# The key is that they output CuPy arrays if use_cupy is True.

def cupy_train_test_split(*arrays, test_size=0.2, random_state=None, shuffle=True):
    """Rudimentary train_test_split for Cupy arrays."""
    if not arrays:
        raise ValueError("At least one array required as input")
    
    length = arrays[0].shape[0]
    for arr in arrays:
        if arr.shape[0] != length:
            raise ValueError("All arrays must have the same shape[0]")

    if random_state is not None:
        cp.random.seed(random_state) # Set CuPy's random seed

    if shuffle:
        indices = cp.random.permutation(length)
    else:
        indices = cp.arange(length)

    n_test = int(cp.ceil(length * test_size))
    n_train = length - n_test

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    result = []
    for arr in arrays:
        result.append(arr[train_indices])
        result.append(arr[test_indices])
    return result

try:
    import cupy as cp
    use_cupy = True
    # print("Successfully imported CuPy. GPU acceleration enabled where applicable.")
except ImportError:
    import numpy as cp  # cp will be numpy
    use_cupy = False
    print("CuPy not available, using NumPy instead for array operations.")
import numpy as np # np is always numpy

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
# ... other imports from main.py ...
import traceback # Ensure traceback is imported

def fetch_stock_data(symbol='AAPL', start='2020-01-01', end='2023-01-01'):
    """Fetch and prepare stock data with enhanced preprocessing.
    Uses CuPy for the final array if use_cupy is True and conversion succeeds."""
    try:
        # Download basic price data
        df = yf.download(symbol, start=start, end=end, progress=False) # Added progress=False for cleaner output
        
        if df.empty:
            print(f"Warning: No data downloaded for {symbol} from {start} to {end}. Attempting wider range.")
            # Attempt a wider date range as a fallback
            df = yf.download(symbol, start='2019-01-01', end=end, progress=False)
            if df.empty:
                print(f"Error: Still no data for {symbol} after attempting wider range. Returning empty.")
                # Return empty structures or None, and handle this in the calling function
                # Assuming scaler would not be fit, original_df and df are empty
                return np.array([]), None, pd.DataFrame(), pd.DataFrame()


        original_df = df.copy()
        
        # Use basic price columns to avoid dimension issues
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Add engineered features for better prediction
        df['DailyReturn'] = df['Close'].pct_change()
        df['SMA5'] = df['Close'].rolling(window=5).mean()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Ensure DailyReturn is calculated before using for Volatility
        if 'DailyReturn' in df:
            df['Volatility5'] = df['DailyReturn'].rolling(window=5).std()
            df['Volatility20'] = df['DailyReturn'].rolling(window=20).std()
        else: # Should not happen if pct_change worked
            df['Volatility5'] = 0.0
            df['Volatility20'] = 0.0

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=14).mean() # Use 0.0 for where
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean() # Use 0.0 for where
        
        # Avoid division by zero in RSI calculation
        rs = gain / loss
        rs[loss == 0] = np.inf # Handle cases where loss is zero to avoid NaN, leads to RSI 100
        df['RSI'] = 100 - (100 / (1 + rs))
        df.loc[:, 'RSI'] = df['RSI'].fillna(50)

        middle_band = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
        df['BB_Middle'] = middle_band
        df['BB_Upper'] = middle_band + 2 * std_20
        df['BB_Lower'] = middle_band - 2 * std_20
        
        df['Momentum5'] = df['Close'] / df['Close'].shift(5) - 1
        df['Momentum10'] = df['Close'] / df['Close'].shift(10) - 1
        
        df['Volume_Change'] = df['Volume'].pct_change()
        volume_ma5 = df['Volume'].rolling(window=5).mean()
        df['Volume_MA5'] = volume_ma5
        df['Volume_Ratio'] = df['Volume'] / volume_ma5
        
        # Drop NaN values from feature engineering
        df_before_dropna = len(df)
        df = df.dropna()
        # print(f"Dropped {df_before_dropna - len(df)} rows due to NaNs from feature engineering.")
        
        if df.empty:
            print(f"Error: No data left for {symbol} after feature engineering and dropna. Original data had {len(original_df)} rows.")
            # Return empty structures or None
            return np.array([]), None, original_df, pd.DataFrame() # Return original_df for context
        
        # Normalize data
        scaler = MinMaxScaler()
        # Ensure df contains only numeric columns for scaler
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) < df.shape[1]:
            print(f"Warning: Non-numeric columns found in DataFrame passed to MinMaxScaler. Using only numeric columns: {numeric_cols.tolist()}")
        
        if not numeric_cols.empty:
            normalized_data_numpy = scaler.fit_transform(df[numeric_cols]) # This is a NumPy array
        else:
            print("Error: No numeric columns to scale. Returning empty array.")
            return np.array([]), None, original_df, df


        # The `use_cupy` variable here refers to the global one defined at the top of the file.
        # There should be NO `use_cupy = ...` assignments within this function.
        if use_cupy:
            try:
                # print(f"fetch_stock_data: Attempting to convert data for {symbol} to CuPy array.")
                final_data_array = cp.asarray(normalized_data_numpy)
                # print(f"fetch_stock_data: Successfully converted data for {symbol} to CuPy array.")
                return final_data_array, scaler, original_df, df[numeric_cols] # Return df with only scaled cols
            except Exception as e:
                print(f"Warning: CuPy conversion failed for {symbol}: {e}. Falling back to NumPy.")
                # np.array() ensures it's a NumPy array, useful if normalized_data_numpy was None or other type
                return np.array(normalized_data_numpy), scaler, original_df, df[numeric_cols]
        else:
            # print(f"fetch_stock_data: Using NumPy for {symbol} (CuPy not enabled or not available).")
            # normalized_data_numpy is already a NumPy array
            return normalized_data_numpy, scaler, original_df, df[numeric_cols]

    except Exception as e:
        print(f"Critical error in fetch_stock_data for {symbol}: {str(e)}")
        traceback.print_exc()
        # In case of a major error, return structures that won't break the calling code too badly,
        # or re-raise if the caller is expected to handle it.
        # Returning empty/None allows the main function to check and potentially skip this stock.
        return np.array([]), None, pd.DataFrame(), pd.DataFrame()

# ... (rest of your main.py, including the main function that calls fetch_stock_data)

def create_sequences(data, seq_len=30, price_col_idx=3, multi_horizon=False):
    """Create input sequences, outputting CuPy arrays if use_cupy is True."""
    try:
        X_list, y_list = [], [] # Build Python lists first
        
        # Data is already cp.ndarray or np.ndarray here
        buy_threshold = 0.005
        sell_threshold = -0.005
        horizons = [1, 3, 5] if multi_horizon else [1] # Not fully used yet, but for structure

        for i in range(len(data) - seq_len - max(horizons)):
            X_list.append(data[i:i + seq_len])
            
            current_price = data[i + seq_len - 1, price_col_idx]
            next_price = data[i + seq_len, price_col_idx]
            
            # Ensure current_price is not zero to avoid division by zero
            if float(current_price) == 0: # Use float() for CuPy/NumPy scalar comparison
                price_return = 0.0
            else:
                price_return = (next_price - current_price) / current_price
            
            if price_return > buy_threshold:
                y_list.append([1, 0, 0])  # Buy
            elif price_return < sell_threshold:
                y_list.append([0, 1, 0])  # Sell
            else:
                y_list.append([0, 0, 1])  # Hold
        
        # Convert lists of arrays/lists to a single CuPy/NumPy array at the end
        if use_cupy:
            try:
                X = cp.asarray(X_list, dtype=cp.float32) # Specify dtype for consistency
                y = cp.asarray(y_list, dtype=cp.int32)   # Target labels
            except Exception as e:
                print(f"Warning: CuPy conversion failed in create_sequences: {e}. Falling back to NumPy.")
                X = np.array(X_list, dtype=np.float32)
                y = np.array(y_list, dtype=np.int32)
        else:
            X = np.array(X_list, dtype=np.float32)
            y = np.array(y_list, dtype=np.int32)
            
        return X, y
    except Exception as e:
        print(f"Error creating sequences: {str(e)}")
        traceback.print_exc()
        raise

def train_improved_model(X_train, y_train, X_val, y_val, epochs=10, batch_size=16):
    """
    Enhanced training for the improved neuromorphic model with validation monitoring
    Optimized for GPU usage by keeping batches on GPU.
    """
    global use_cupy # Allow modification if CuPy ops fail consistently
    try:
        input_dim = X_train.shape[2]
        hidden_dim = 48
        
        print(f"Initializing ImprovedNeuroGRU with input_dim={input_dim}, hidden_dim={hidden_dim}, use_cupy={use_cupy}")
        model = ImprovedNeuroGRU(input_dim=input_dim, hidden_dim=hidden_dim)

        # Initialize output layer weights on the correct device (CuPy or NumPy)
        W_out = cp.random.randn(3, model.hidden_dim) * 0.1 / cp.sqrt(model.hidden_dim)
        b_out = cp.zeros(3)
        
        model.output_weights = W_out
        model.output_bias = b_out
        
        training_metrics = {
            'accuracies': [], 'losses': [], 'val_accuracies': [], 'val_losses': [],
            'neuron_counts': [model.hidden_dim],
            'class_accuracies': {'buy': [], 'sell': [], 'hold': []}
        }
        
        print(f"Training on dataset with {len(X_train)} samples, {X_train.shape[1]} timesteps, {X_train.shape[2]} features")
        
        learning_phases = [
            {'phase': 'exploration', 'epochs': 2, 'lr': 0.01},
            {'phase': 'exploitation', 'epochs': 4, 'lr': 0.007},
            {'phase': 'refinement', 'epochs': 3, 'lr': 0.003},
            {'phase': 'convergence', 'epochs': 1, 'lr': 0.001}
        ]
        current_epoch = 0

        for phase_config in learning_phases:
            phase = phase_config['phase']
            phase_epochs = phase_config['epochs']
            learning_rate = phase_config['lr']
            
            model.enable_structural_plasticity(phase != 'exploration')
            
            print(f"\n=== Starting {phase.upper()} phase ({phase_epochs} epochs) ===")
            print(f"Learning rate: {learning_rate}, Structural plasticity: {'enabled' if model.structural_plasticity_enabled else 'disabled'}")

            for epoch in range(phase_epochs):
                current_epoch += 1
                num_samples_train = X_train.shape[0]
                indices_np = np.arange(num_samples_train) # Use NumPy for shuffling CPU-side
                np.random.shuffle(indices_np)
                
                epoch_loss = 0.0
                correct = 0
                class_correct = [0, 0, 0] # Python list for accumulation
                class_total = [0, 0, 0]   # Python list

                for batch_start in range(0, num_samples_train, batch_size):
                    batch_end = min(batch_start + batch_size, num_samples_train)
                    batch_indices_np = indices_np[batch_start:batch_end]
                    
                    # OPTIMIZATION: Get batch data on GPU (or keep as NumPy if not using CuPy)
                    # X_train, y_train are already cp.ndarray if use_cupy
                    batch_X = X_train[batch_indices_np]
                    batch_y_one_hot = y_train[batch_indices_np]
                    
                    batch_size_actual = batch_X.shape[0]
                    if batch_size_actual == 0: continue

                    current_batch_loss_sum = 0.0 # Accumulate on CPU
                    current_batch_correct_count = 0 # Accumulate on CPU

                    # Process each sequence *within the batch* (batch_X and batch_y are on GPU)
                    for i in range(batch_size_actual):
                        seq = batch_X[i]             # This is a slice of a CuPy array, stays on GPU
                        target_one_hot = batch_y_one_hot[i] # Stays on GPU
                        
                        try:
                            model.reset_state()
                            h_final = model.process_sequence(seq, target_one_hot, inference_mode=False)
                           
                            # Ensure W_out and b_out match current model.hidden_dim
                            if h_final.shape[0] != W_out.shape[1]:
                                print(f"Training: Output layer dim mismatch. Model: {h_final.shape[0]}, W_out: {W_out.shape[1]}. Re-initializing W_out.")
                                W_out = model.cp.random.randn(3, h_final.shape[0]) * 0.1 / model.cp.sqrt(h_final.shape[0])
                                b_out = model.cp.zeros(3)
                                model.output_weights = W_out # Update model's copy too
                                model.output_bias = b_out

                            logits = W_out @ h_final + b_out
                            probs = model.softmax(logits) # model.softmax should handle cp array
                            
                            loss = -model.cp.sum(target_one_hot * model.cp.log(probs + 1e-10)) # All CuPy ops
                            current_batch_loss_sum += float(loss.item()) # Convert to float for accumulation

                            pred_idx = int(model.cp.argmax(probs).item())
                            label_idx = int(model.cp.argmax(target_one_hot).item())

                            if pred_idx == label_idx:
                                current_batch_correct_count += 1
                                class_correct[label_idx] += 1
                            class_total[label_idx] += 1
                            
                            # Gradients and updates (all CuPy ops)
                            d_logits = probs - target_one_hot
                            d_W_out = model.cp.outer(d_logits, h_final)
                            d_b_out = d_logits
                            
                            W_out -= learning_rate * d_W_out
                            b_out -= learning_rate * d_b_out
                            model.output_weights = W_out # Keep model's copy synced
                            model.output_bias = b_out

                        except Exception as e_sample:
                            print(f"Error processing sample {batch_indices_np[i]} in batch: {str(e_sample)}")
                            if use_cupy: cp.cuda.runtime.deviceSynchronize() # Helps get more precise CuPy errors
                            traceback.print_exc()
                            # Attempt to recover by switching to NumPy if CuPy errors persist
                            # This is a very aggressive recovery, might not be desired.
                            # if use_cupy and isinstance(e_sample, cp.cuda.CuPyError):
                            #     print("Persistent CuPy error, attempting to switch to NumPy for the rest of training.")
                            #     use_cupy = False
                            #     model.cp = np # Switch model's cp reference
                            #     # Need to convert all model weights and states to NumPy - very complex here.
                            #     # For simplicity, this advanced recovery is omitted.
                            continue
                    
                    correct += current_batch_correct_count
                    epoch_loss += current_batch_loss_sum
                    
                    batch_acc = current_batch_correct_count / batch_size_actual if batch_size_actual > 0 else 0
                    batch_avg_loss = current_batch_loss_sum / batch_size_actual if batch_size_actual > 0 else 0
                    model.update_performance(batch_avg_loss, batch_acc) # This fn should handle cp scalars

                    if (batch_start // batch_size) % 5 == 0: # Print every 5 batches
                        stats = model.get_network_stats()
                        print(f"  Batch {(batch_start // batch_size)+1}/{(num_samples_train + batch_size -1)//batch_size}: "
                              f"Loss={batch_avg_loss:.4f}, Acc={batch_acc:.4f}, Neurons={stats['hidden_dim']}")
                
                train_acc = correct / num_samples_train if num_samples_train > 0 else 0
                train_loss = epoch_loss / num_samples_train if num_samples_train > 0 else 0
                
                class_acc_epoch = [(class_correct[c] / class_total[c] if class_total[c] > 0 else 0) for c in range(3)]
                training_metrics['class_accuracies']['buy'].append(class_acc_epoch[0])
                training_metrics['class_accuracies']['sell'].append(class_acc_epoch[1])
                training_metrics['class_accuracies']['hold'].append(class_acc_epoch[2])
                
                # Validation
                val_correct = 0
                val_loss_sum = 0.0
                num_samples_val = X_val.shape[0]

                for val_batch_start in range(0, num_samples_val, batch_size): # Use same batch_size for val
                    val_batch_end = min(val_batch_start + batch_size, num_samples_val)
                    val_batch_X = X_val[val_batch_start:val_batch_end]
                    val_batch_y_one_hot = y_val[val_batch_start:val_batch_end]
                    
                    val_batch_size_actual = val_batch_X.shape[0]
                    if val_batch_size_actual == 0: continue

                    for i in range(val_batch_size_actual):
                        seq = val_batch_X[i]
                        target_one_hot = val_batch_y_one_hot[i]
                        try:
                            model.reset_state()
                            # In NeuroGRU, forward/process_sequence should handle inference_mode
                            h_final_val = model.process_sequence(seq, inference_mode=True) 

                            if h_final_val.shape[0] != W_out.shape[1]:
                                # This case means W_out from training is not compatible with current model state
                                # (e.g. after pruning/growth in validation which is unusual but possible if logic is complex)
                                # For validation, it's best to use the model's *current* internal output layer if it has one,
                                # or re-init a temporary one based on h_final_val.shape[0]
                                # Assuming W_out, b_out are the "master" ones from training loop:
                                print(f"Validation: Output layer dim mismatch. Model: {h_final_val.shape[0]}, W_out: {W_out.shape[1]}. Using temp W_out for this val pred.")
                                temp_W_out = model.cp.random.randn(3, h_final_val.shape[0]) * 0.1 / model.cp.sqrt(h_final_val.shape[0])
                                temp_b_out = model.cp.zeros(3)
                                logits = temp_W_out @ h_final_val + temp_b_out
                            else:
                                logits = W_out @ h_final_val + b_out

                            probs = model.softmax(logits)
                            val_loss_sum += float(-model.cp.sum(target_one_hot * model.cp.log(probs + 1e-10)).item())
                            
                            if int(model.cp.argmax(probs).item()) == int(model.cp.argmax(target_one_hot).item()):
                                val_correct +=1
                        except Exception as e_val_sample:
                            print(f"Error in validation sample: {str(e_val_sample)}")
                            if use_cupy: cp.cuda.runtime.deviceSynchronize()
                            traceback.print_exc()
                            continue
                
                val_acc = val_correct / num_samples_val if num_samples_val > 0 else 0
                val_avg_loss = val_loss_sum / num_samples_val if num_samples_val > 0 else 0
                
                training_metrics['accuracies'].append(train_acc)
                training_metrics['losses'].append(train_loss)
                training_metrics['val_accuracies'].append(val_acc)
                training_metrics['val_losses'].append(val_avg_loss)
                training_metrics['neuron_counts'].append(model.hidden_dim) # Track at end of epoch too

                stats = model.get_network_stats() # get_network_stats should convert to Python types for dict
                print(f"Epoch {current_epoch}/{sum(p['epochs'] for p in learning_phases)}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_avg_loss:.4f}, Acc={val_acc:.4f}")
                print(f"  Class Acc: Buy={class_acc_epoch[0]:.4f}, Sell={class_acc_epoch[1]:.4f}, Hold={class_acc_epoch[2]:.4f}")
                print(f"  Network: {stats['hidden_dim']} neurons (E:{stats['excitatory_count']}/I:{stats['inhibitory_count']}), Activity: {stats.get('activity_metrics',{}).get('mean_activity',0.0):.3f}")

                if len(training_metrics['val_losses']) > 3 and training_metrics['val_losses'][-1] > training_metrics['val_losses'][-3] and phase == 'refinement':
                    print("Early stopping triggered: Validation loss increasing.")
                    break 
                gc.collect()
                if use_cupy: cp.get_default_memory_pool().free_all_blocks()
            
            if phase == 'refinement' and len(training_metrics['val_losses']) > 3 and training_metrics['val_losses'][-1] > training_metrics['val_losses'][-3]:
                break

        # Training visualization (ensure data passed to matplotlib is NumPy)
        plt.figure(figsize=(20, 12))
        def to_numpy_if_cupy(arr_list):
            if use_cupy and len(arr_list) > 0 and isinstance(arr_list[0], cp.ndarray):
                return [cp.asnumpy(a) for a in arr_list]
            if len(arr_list) > 0 and isinstance(arr_list[0], np.ndarray): # Ensure list of np arrays, not cp.ndarray
                 return [a.item() if a.ndim == 0 else a for a in arr_list] # Convert 0-d arrays
            return arr_list # Assume list of Python numbers

        plt.subplot(2, 3, 1)
        plt.plot(to_numpy_if_cupy(training_metrics['accuracies']), label='Train')
        plt.plot(to_numpy_if_cupy(training_metrics['val_accuracies']), label='Validation')
        plt.title('Accuracy Evolution'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

        plt.subplot(2, 3, 2)
        plt.plot(to_numpy_if_cupy(training_metrics['losses']), label='Train')
        plt.plot(to_numpy_if_cupy(training_metrics['val_losses']), label='Validation')
        plt.title('Loss Evolution'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
        
        plt.subplot(2, 3, 3)
        plt.plot(to_numpy_if_cupy(training_metrics['neuron_counts']))
        plt.title('Network Size Evolution'); plt.xlabel('Epoch/Update Period'); plt.ylabel('Number of Neurons')

        plt.subplot(2, 3, 4)
        plt.plot(to_numpy_if_cupy(training_metrics['class_accuracies']['buy']), label='Buy')
        plt.plot(to_numpy_if_cupy(training_metrics['class_accuracies']['sell']), label='Sell')
        plt.plot(to_numpy_if_cupy(training_metrics['class_accuracies']['hold']), label='Hold')
        plt.title('Class-Specific Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
        
        # ... (rest of plotting similar, ensure .get_network_stats() returns python types or convert here) ...
        final_stats = model.get_network_stats() # This should return dict of Python types
        e_count = final_stats.get('excitatory_count', 0)
        i_count = final_stats.get('inhibitory_count', 0)
        plt.subplot(2,3,5)
        if e_count + i_count > 0:
             plt.bar([0, 1], [e_count, i_count], color=['#4CAF50', '#F44336'])
        plt.title('Final E/I Balance'); plt.xticks([0,1],['Excitatory','Inhibitory']); plt.ylabel('Neuron Count')

        plt.subplot(2,3,6)
        # ... (learning phase plot logic) ...
        phase_labels_flat = []
        # ... (ensure this logic correctly reflects actual run epochs if early stopping) ...
        total_epochs_run = len(training_metrics['accuracies'])
        current_total_epochs_viz = 0
        for p_config in learning_phases:
            phase_name = p_config['phase']
            defined_phase_epochs = p_config['epochs']
            epochs_in_this_phase_for_viz = 0
            if current_total_epochs_viz < total_epochs_run:
                epochs_in_this_phase_for_viz = min(defined_phase_epochs, total_epochs_run - current_total_epochs_viz)
            phase_labels_flat.extend([phase_name] * epochs_in_this_phase_for_viz)
            current_total_epochs_viz += epochs_in_this_phase_for_viz
            if current_total_epochs_viz >= total_epochs_run:
                break
        
        x_values = range(len(phase_labels_flat))
        colors = {'exploration': '#FF9800', 'exploitation': '#2196F3', 
                 'refinement': '#9C27B0', 'convergence': '#00BCD4'}
        if len(x_values) > 0:
            plt.bar(x_values, [1]*len(x_values), color=[colors.get(p_label, '#000000') for p_label in phase_labels_flat], width=1.0)
        plt.title('Learning Phase Progression'); plt.xlabel('Epoch'); plt.yticks([])
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[p_key], label=p_key.capitalize()) for p_key in colors.keys()]
        plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'improved_training_results_{timestamp}.png')
        print(f"Training visualization saved to improved_training_results_{timestamp}.png")
        plt.close() # Close plot to free memory
        
        return model, W_out, b_out, training_metrics
    except Exception as e:
        print(f"Critical error during training: {str(e)}")
        if use_cupy: cp.cuda.runtime.deviceSynchronize()
        traceback.print_exc()
        raise
    finally:
        if use_cupy: # Final cleanup
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()


# --- utils.py context (evaluate_model, analyze_model_performance, save_results) ---
# We'll update evaluate_model as an example of applying similar batching logic.

def evaluate_model(model, W_out, b_out, X_test, y_test, batch_size=32): # Added batch_size
    """Evaluate the improved model with detailed performance analysis (GPU optimized batching)."""
    global use_cupy
    cp_module = model.cp # Use the model's cp module
    try:
        if model.hidden_dim != W_out.shape[1]:
            print(f"Eval: Output layer dim mismatch. Model: {model.hidden_dim}, W_out: {W_out.shape[1]}. Re-init W_out.")
            W_out = cp_module.random.randn(3, model.hidden_dim) * 0.1 / cp_module.sqrt(model.hidden_dim)
            b_out = cp_module.zeros(3)
        
        print(f"Evaluating on {len(X_test)} test samples...")
        model.set_learning_phase('convergence') # Ensure inference mode
        original_structural_plasticity = model.structural_plasticity_enabled
        model.enable_structural_plasticity(False) # Disable for stable inference
        
        y_true_list, y_pred_list, confidences_list = [], [], []
        num_samples_test = X_test.shape[0]

        for batch_start in range(0, num_samples_test, batch_size):
            batch_end = min(batch_start + batch_size, num_samples_test)
            batch_X_test = X_test[batch_start:batch_end]
            batch_y_test_one_hot = y_test[batch_start:batch_end]
            
            batch_size_actual = batch_X_test.shape[0]
            if batch_size_actual == 0: continue
            
            # print(f"  Eval Batch: {batch_start // batch_size + 1} / {(num_samples_test + batch_size - 1) // batch_size}")

            for i in range(batch_size_actual):
                seq = batch_X_test[i]
                target_one_hot = batch_y_test_one_hot[i]
                try:
                    model.reset_state()
                    h_final_test = model.process_sequence(seq, inference_mode=True)
                    
                    current_W_out, current_b_out = W_out, b_out
                    if h_final_test.shape[0] != current_W_out.shape[1]:
                        # print(f"Eval sample: Output layer dim mismatch. Model: {h_final_test.shape[0]}, W_out: {current_W_out.shape[1]}. Using temp W_out.")
                        current_W_out = cp_module.random.randn(3, h_final_test.shape[0]) * 0.1 / cp_module.sqrt(h_final_test.shape[0])
                        current_b_out = cp_module.zeros(3)

                    logits = current_W_out @ h_final_test + current_b_out
                    probs = model.softmax(logits)
                    
                    pred_idx = int(cp_module.argmax(probs).item())
                    label_idx = int(cp_module.argmax(target_one_hot).item())
                    confidence = float(cp_module.max(probs).item())
                    
                    y_pred_list.append(pred_idx)
                    y_true_list.append(label_idx)
                    confidences_list.append(confidence)
                except Exception as e_eval_sample:
                    print(f"Error evaluating test sample: {str(e_eval_sample)}")
                    if use_cupy: cp.cuda.runtime.deviceSynchronize()
                    traceback.print_exc()
                    continue
        
        model.enable_structural_plasticity(original_structural_plasticity) # Restore
        
        results = {}
        if len(y_true_list) > 0:
            y_true_np = np.array(y_true_list) # For scikit-learn
            y_pred_np = np.array(y_pred_list)
            # confidences_np = np.array(confidences_list)

            accuracy = np.mean(y_true_np == y_pred_np)
            precision = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0)
            recall = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0)
            f1 = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
            cm = confusion_matrix(y_true_np, y_pred_np)
            
            print("\n====== Evaluation Results ======")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision (macro): {precision:.4f}")
            print(f"Recall (macro): {recall:.4f}")
            print(f"F1 Score (macro): {f1:.4f}")
            
            results = {
                'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
                'confusion_matrix': cm.tolist(), 'class_names': ['Buy', 'Sell', 'Hold']
            }
            # Plotting (ensure confidences_list is converted to NumPy for hist)
            # ... (plotting logic as before, using np.array(confidences_list) for histogram)
        else:
            print("No valid predictions in evaluation.")
            results = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'confusion_matrix': [], 'class_names':[]}

        return results
    except Exception as e:
        print(f"Critical error during evaluation: {e}")
        if use_cupy: cp.cuda.runtime.deviceSynchronize()
        traceback.print_exc()
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'confusion_matrix': [], 'class_names':[]}
    finally:
        if use_cupy:
             cp.get_default_memory_pool().free_all_blocks()

# analyze_model_performance and save_results would also need careful handling of cp/np arrays
# For analyze_model_performance -> simulate_trading, it will use the same batching logic as evaluate_model
# For save_results, ensure all cp.ndarrays are converted to lists or np.ndarrays then lists via json_serialize.

def analyze_model_performance(model, W_out, b_out, X_test, y_test, original_df):
    """Analyze model performance with financial metrics."""
    print("Analyzing model performance...")
    network_stats = model.get_network_stats() # Should return dict of Python types
    
    # Simplified trading simulation (assumes prices are available and align with X_test)
    # This simulation needs to be robust against index mismatches and use GPU if possible
    sim_returns = 0.0 # Placeholder
    sharpe = 0.0 # Placeholder
    
    # For a more accurate simulation, you'd align X_test samples with original_df dates/prices
    # And use the evaluate_model's prediction logic.
    # Due to complexity, full trading sim re-write is omitted here, but it would follow
    # the batching and GPU processing pattern of evaluate_model.

    print(f"Network size: {network_stats.get('hidden_dim',0)} neurons")
    print(f"E/I ratio: {network_stats.get('ei_ratio',0.0):.2f}")
    # print(f"Simulated return: {sim_returns:.2f}%") # Requires full simulation
    
    return {
        'network_stats': network_stats,
        'financial_metrics': {'simulated_return': sim_returns, 'sharpe_ratio': sharpe}
    }

def json_serialize(obj):
    """Convert numpy/cupy arrays to lists for JSON serialization"""
    if isinstance(obj, dict):
        return {k: json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_serialize(item) for item in obj]
    elif use_cupy and isinstance(obj, cp.ndarray):
        return cp.asnumpy(obj).tolist() # CuPy -> NumPy -> List
    elif isinstance(obj, np.ndarray):
        return obj.tolist() # NumPy -> List
    elif isinstance(obj, (np.generic, cp.generic)): # Handles NumPy/CuPy scalars
        return obj.item()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # print(f"Warning: Unhandled type in json_serialize: {type(obj)}")
        return str(obj) # Fallback

def save_results(model, W_out, b_out, training_metrics, test_results, performance_metrics, args):
    """Save model results and metrics to files, ensuring CuPy arrays are handled."""
    try:
        results_dir = "model_results"
        if not os.path.exists(results_dir): os.makedirs(results_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure all metrics are JSON serializable (handles CuPy arrays)
        metrics_to_save = {
            'training': json_serialize(training_metrics),
            'test': json_serialize(test_results),
            'performance': json_serialize(performance_metrics),
            'parameters': vars(args) # Convert argparse Namespace to dict
        }
        
        metrics_file = os.path.join(results_dir, f"metrics_{args.ticker}_{timestamp}.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        print(f"Results saved to {metrics_file}")

        # Model summary can use get_network_stats directly as it should return Python types
        summary_file = os.path.join(results_dir, f"model_summary_{args.ticker}_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write(f"ImprovedNeuroGRU Model Summary - Ticker: {args.ticker}\n")
            # ... (rest of summary writing, using .get() for safety) ...
            network_stats = performance_metrics.get('network_stats', {})
            f.write(f"  Hidden Dimension: {network_stats.get('hidden_dim', 'N/A')}\n")
            # ...
        print(f"Model summary saved to {summary_file}")
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        traceback.print_exc()
        return False

# Main execution function
def main(args):
    """Main execution function with improved workflow"""
    global use_cupy # Allow main to potentially change this based on early failures
    try:
        print(f"Starting improved stock prediction with {args.ticker} from {args.start} to {args.end}")
        
        data, scaler, original_df, processed_df = fetch_stock_data(
            symbol=args.ticker, start=args.start, end=args.end
        ) # fetch_stock_data now handles initial cp.asarray
        
        print(f"Data shape: {data.shape}, Type: {type(data)}")
        print(f"Features: {processed_df.columns.tolist()}")
        
        X, y = create_sequences(data, seq_len=args.seq_len, multi_horizon=args.multi_horizon)
        print(f"Created {len(X)} sequences with shape {X.shape}, Type X: {type(X)}, Type y: {type(y)}")
        
        X_train_full, X_test, y_train_full, y_test = cupy_train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        X_train, X_val, y_train, y_val = cupy_train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, shuffle=False) # sklearn's default is 0.25 for validation if test_size is float
                                                                                                                                # here test_size=0.2 means 20% of X_train_full becomes X_val.
        print(f"Using CuPy for train/test split.")
        print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}")
        
        model, W_out, b_out, training_metrics = train_improved_model(
            X_train, y_train, X_val, y_val, 
            epochs=args.epochs, 
            batch_size=args.batch_size
        )
        
        print("\nEvaluating model on test set...")
        test_results = evaluate_model(model, W_out, b_out, X_test, y_test, batch_size=args.batch_size)
        
        print("\nAnalyzing model performance...")
        performance_metrics = analyze_model_performance(model, W_out, b_out, X_test, y_test, original_df)
        
        save_results(model, W_out, b_out, training_metrics, test_results, performance_metrics, args)
        
        print("Improved stock prediction completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        if use_cupy: cp.cuda.runtime.deviceSynchronize()
        traceback.print_exc()
        return 1
    finally:
        if use_cupy: # Final final cleanup
            print("Final GPU memory cleanup.")
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Improved Stock Prediction with Neuromorphic Computing")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2023-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--seq_len", type=int, default=30, help="Sequence length for prediction")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs") # Defaulted to 10 for quicker tests
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size") # Defaulted to 32
    parser.add_argument("--multi_horizon", action="store_true", help="Enable multi-horizon prediction")
    
    args = parser.parse_args()
    exit(main(args))