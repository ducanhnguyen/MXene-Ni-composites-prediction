import numpy as np
import pandas as pd

from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def build_features(df, state_vars, process_features):
    X_parts = []
    for c in state_vars:
        X_parts.append(df[f"{c}_lag1"].values.reshape(-1, 1))
    for u in process_features:
        X_parts.append(df[u].values.reshape(-1, 1))
    return np.hstack(X_parts)


def run_kernel_ar(
    train_df,
    test_df,
    state_vars,
    process_features,
    alpha=1.0,
    gamma=0.1,
    kernel="rbf"
):
    models = {}
    metrics = []

    viz_df = test_df.copy().reset_index(drop=True)

    X_train = build_features(train_df, state_vars, process_features)
    X_test = build_features(test_df, state_vars, process_features)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    for c in state_vars:
        y_train = train_df[c].values
        y_test = test_df[c].values

        model = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        viz_df[f"{c}_pred"] = y_pred

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mean_actual = np.mean(np.abs(y_test))
        mae_pct = (mae / mean_actual * 100) if mean_actual != 0 else np.nan

        metrics.append([c, r2_score(y_test, y_pred), mae, mse, rmse, mae_pct])
        models[c] = model

    metrics_df = pd.DataFrame(
        metrics,
        columns=["State", "R2", "MAE", "MSE", "RMSE", "MAE_%"]
    )

    return {
        "models": models,
        "metrics": metrics_df,
        "viz_df": viz_df,
        "viz_columns": state_vars,
        "method": "KERNEL-AR"
    }