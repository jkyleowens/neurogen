import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(true_values, predicted_values):
    """
    Calculate regression metrics.

    Args:
        true_values (list or np.array): Ground truth values.
        predicted_values (list or np.array): Predicted values by the model.

    Returns:
        dict: Dictionary with RMSE, MAE, and R2 scores.
    """
    true_values = np.array(true_values).flatten()
    predicted_values = np.array(predicted_values).flatten()

    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }