# src/models/lgbm_model.py
import numpy as np
import pandas as pd
import sklearn
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    make_scorer
)

from lightgbm import LGBMRegressor


def train_lgbm_multioutput(
    X_train,
    y_train,
    param_grid,
    random_state=42,
    cv=3,
    verbose=2
):
    """
    Train MultiOutput LightGBM with GridSearchCV
    """

    base_lgbm = LGBMRegressor(
        objective="regression",
        learning_rate=0.05,
        random_state=random_state,
        n_jobs=1,
        verbose=-1
    )

    model = MultiOutputRegressor(base_lgbm)

    scoring = {
        "R2": "r2",
        "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        refit="R2",
        cv=cv,
        verbose=verbose,
        n_jobs=1
    )

    grid.fit(X_train, y_train)
    return grid


def run_lgbm(
    train_df,
    test_df,
    input_features,
    state_vars,
    param_grid,
    random_state=42,
    cv=3
):
    """
    LightGBM
    """
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
    grid = train_lgbm_multioutput(
        X_train=X_train,
        y_train=y_train,
        param_grid=param_grid,
        random_state=random_state,
        cv=cv
    )

    model = grid.best_estimator_

    # =============================
    # PREDICT
    # =============================
    y_pred_delta = model.predict(X_test)

    # =============================
    # METRICS
    # =============================
    metrics = []
    for i, c in enumerate(state_vars):
        y_true_i = y_test[:, i]
        y_pred_i = y_pred_delta[:, i]

        mae = mean_absolute_error(y_true_i, y_pred_i)
        mse = np.mean((y_true_i - y_pred_i) ** 2)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_i, y_pred_i)

        mean_abs_true = np.mean(np.abs(y_true_i))
        mae_pct = (mae / mean_abs_true * 100) if mean_abs_true > 0 else np.nan

        metrics.append([c, r2, mae, mse, rmse, mae_pct])

    metrics_df = pd.DataFrame(
        metrics,
        columns=["State", "R2", "MAE", "MSE", "RMSE", "MAE_%"]
    )

    # =============================
    # RECONSTRUCTION
    # =============================
    viz_df = test_df.copy().reset_index(drop=True)
    for i, c in enumerate(state_vars):
        viz_df[f"{c}_pred"] = viz_df[f"{c}_lag1"] + y_pred_delta[:, i]

    return {
        "grid": grid,
        "metrics": metrics_df,
        "viz_df": viz_df,
        "viz_columns": state_vars,
        "method": "LightGBM"
    }
