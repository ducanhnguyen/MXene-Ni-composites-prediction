# src/models/varx_model.py
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.metrics import r2_score, mean_absolute_error


def run_varx(
    train_df,
    test_df,
    state_vars,
    process_features,
    maxlags=1
):

    y_train = train_df[state_vars].astype(float)
    y_test = test_df[state_vars].astype(float)

    X_train = train_df[process_features].astype(float)
    X_test = test_df[process_features].astype(float)

    # =============================
    # Train VARX
    # =============================
    model = VARMAX(
        endog=y_train,
        exog=X_train,
        order=(maxlags, 0),
        trend="c"
    )

    res = model.fit(disp=False, maxiter=1000)

    # =============================
    # Forecast
    # =============================
    y_pred = res.forecast(
        steps=len(y_test),
        exog=X_test
    )

    # =============================
    # Metrics
    # =============================
    metrics = []
    for c in state_vars:
        y_true_c = y_test[c].to_numpy()
        y_pred_c = y_pred[c].to_numpy()

        mae = mean_absolute_error(y_true_c, y_pred_c)
        mse = np.mean((y_true_c - y_pred_c) ** 2)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_c, y_pred_c)

        mean_abs_true = np.mean(np.abs(y_true_c))
        mae_pct = (mae / mean_abs_true * 100) if mean_abs_true > 0 else np.nan

        metrics.append([c, r2, mae, mse, rmse, mae_pct])

    metrics_df = pd.DataFrame(
        metrics,
        columns=["State", "R2", "MAE", "MSE", "RMSE", "MAE_%"]
    )
    # =============================
    # Build viz_df (same format)
    # =============================
    viz_df = test_df.copy().reset_index(drop=True)
    for c in state_vars:
        viz_df[f"{c}_pred"] = y_pred[c].values

    return {
        "model": res,
        "metrics": metrics_df,
        "viz_df": viz_df,
        "viz_columns": state_vars,
        "method": "VARX"
    }
