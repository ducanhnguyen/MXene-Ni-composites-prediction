# src/XGB.py
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.models.xgb_model import train_xgb_multioutput


def run_xgb(
    train_df,
    test_df,
    input_features,
    state_vars,
    param_grid,
    random_state=42,
    cv=3
):
    # =============================
    # BUILD TRAIN / TEST
    # =============================
    X_train = train_df[input_features].to_numpy(np.float32)
    y_train = train_df[[f"d_{c}" for c in state_vars]].to_numpy(np.float32)

    X_test = test_df[input_features].to_numpy(np.float32)
    y_test = test_df[[f"d_{c}" for c in state_vars]].to_numpy(np.float32)

    # =============================
    # TRAIN
    # =============================
    grid = train_xgb_multioutput(
        X_train=X_train,
        y_train=y_train,
        param_grid=param_grid,
        random_state=random_state,
        cv=cv
    )
    model = grid.best_estimator_
    # =============================
    # PRINT BEST CONFIG
    # =============================
    print("\n[BEST PARAMS]")
    for k, v in grid.best_params_.items():
        print(f"  {k}: {v}")

    print(f"[BEST CV SCORE] {grid.best_score_:.6f}\n")

    # =============================
    # PREDICT Δ
    # =============================
    y_pred_delta = model.predict(X_test)

    # =============================
    # METRICS (Δ space)
    # =============================
    metrics = []
    for i, c in enumerate(state_vars):
        y_true = y_test[:, i]
        y_pred = y_pred_delta[:, i]

        mse = mean_squared_error(y_true, y_pred)

        y_true = y_test[:, i]

        value_range = np.max(y_true) - np.min(y_true)
        rel_mae_pct = (
            mean_absolute_error(y_true, y_pred) / value_range * 100
            if value_range > 0 else np.nan
        )


        metrics.append([
            c,
            r2_score(y_true, y_pred),
            mean_absolute_error(y_true, y_pred),
            mse,
            np.sqrt(mse),
            rel_mae_pct
        ])

    metrics_df = pd.DataFrame(
        metrics,
        columns=["State", "R2", "MAE", "MSE", "RMSE", "MAE_%"]
    )

    # =============================
    # RECONSTRUCTION (MODEL-SPECIFIC)
    # =============================
    viz_df = test_df.copy().reset_index(drop=True)
    for i, c in enumerate(state_vars):
        viz_df[f"{c}_pred"] = viz_df[f"{c}_lag1"] + y_pred_delta[:, i]

    viz_columns = state_vars

    return {
        "grid": grid,
        "metrics": metrics_df,
        "viz_df": viz_df,
        "viz_columns": viz_columns,
        "method": "XGBRegressor"
    }

