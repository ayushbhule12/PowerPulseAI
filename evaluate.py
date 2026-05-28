# src/evaluate.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> dict:
    """
    Compute MAE, RMSE, MAPE, R² and print a formatted summary.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    # MAPE — avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    results = {
        'Model': model_name,
        'MAE':   round(mae,  4),
        'RMSE':  round(rmse, 4),
        'MAPE':  round(mape, 2),
        'R2':    round(r2,   4)
    }

    print(f"\n{'='*45}")
    print(f"  {model_name} Evaluation")
    print(f"{'='*45}")
    print(f"  MAE  : {mae:.4f} kW")
    print(f"  RMSE : {rmse:.4f} kW")
    print(f"  MAPE : {mape:.2f}%")
    print(f"  R²   : {r2:.4f}")
    print(f"{'='*45}")

    return results


def compare_models(results_list: list) -> pd.DataFrame:
    """
    Takes a list of result dicts from evaluate() and builds a comparison table.
    """
    df = pd.DataFrame(results_list).set_index('Model')
    df = df.sort_values('RMSE')
    print("\n=== Model Comparison Table ===")
    print(df.to_string())
    df.to_csv("data/model_comparison.csv")
    return df