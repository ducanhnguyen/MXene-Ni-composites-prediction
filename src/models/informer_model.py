# src/models/informer_model.py (FIXED VERSION)
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import TransformerModel
from darts.dataprocessing.transformers import Scaler
from darts import concatenate
from sklearn.metrics import r2_score, mean_absolute_error


def _build_series(df_case, value_cols):
    df_tmp = (
        df_case
        .sort_values("Timestep1and2")
        .reset_index(drop=True)
        .copy()
    )
    df_tmp[value_cols] = df_tmp[value_cols].astype("float32")
    return TimeSeries.from_dataframe(
        df_tmp,
        value_cols=value_cols
    )


def run_informer(
        train_df,
        test_df,
        state_vars,
        process_features,
        input_chunk_length=20,
        output_chunk_length=1,
        n_epochs=100,
        batch_size=32,
        random_state=42,
):
    delta_vars = [f"d_{var}" for var in state_vars]

    df_all = pd.concat([train_df, test_df], ignore_index=True)
    df_all = df_all.sort_values(["CaseID", "Timestep1and2"]).reset_index(drop=True)

    for var in state_vars:
        df_all[f"d_{var}"] = df_all.groupby("CaseID")[var].diff()

    df_all = df_all.dropna(subset=delta_vars).reset_index(drop=True)
    full_series_delta = {}
    full_series_level = {}
    full_covs = {}
    split_index = {}

    for case_id, df_case in df_all.groupby("CaseID"):
        ts_delta = _build_series(df_case, delta_vars)
        ts_level = _build_series(df_case, state_vars)

        # Covariates (process features)
        ts_x = _build_series(df_case, process_features)

        n_train = len(train_df[train_df["CaseID"] == case_id])
        n_train = max(0, n_train - 1)

        full_series_delta[case_id] = ts_delta
        full_series_level[case_id] = ts_level
        full_covs[case_id] = ts_x
        split_index[case_id] = n_train
    train_series_delta = []
    train_covs = []

    for case_id in full_series_delta:
        n_train = split_index[case_id]
        if n_train > 0:
            train_series_delta.append(full_series_delta[case_id][:n_train])
            train_covs.append(full_covs[case_id][:n_train])

    if not train_series_delta:
        raise ValueError("No training data available after delta computation!")

    scaler_delta = Scaler()
    scaler_x = Scaler()

    all_train_delta = concatenate(train_series_delta, axis=0, ignore_time_axis=True)
    all_train_x = concatenate(train_covs, axis=0, ignore_time_axis=True)

    scaler_delta.fit(all_train_delta)
    scaler_x.fit(all_train_x)

    # Transform train data
    train_series_delta_scaled = [scaler_delta.transform(s) for s in train_series_delta]
    train_covs_scaled = [scaler_x.transform(c) for c in train_covs]

    # ======================================================
    # Model
    # ======================================================
    model = TransformerModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=1,
        dropout=0.1,
        batch_size=batch_size,
        n_epochs=n_epochs,
        random_state=random_state,
        optimizer_kwargs={"lr": 1e-3},
    )

    # ======================================================
    # Train
    # ======================================================
    print(f"[Informer] Training on {len(train_series_delta_scaled)} cases...")
    model.fit(
        series=train_series_delta_scaled,
        past_covariates=train_covs_scaled,
        verbose=True
    )

    # ======================================================
    # Predict
    # ======================================================
    viz_parts = []
    metrics = []

    for case_id in full_series_delta:
        ts_delta_full = full_series_delta[case_id]
        ts_level_full = full_series_level[case_id]
        ts_x_full = full_covs[case_id]
        n_train = split_index[case_id]
        n_test = len(ts_delta_full) - n_train

        if n_test <= 0:
            continue

        # Scale full series
        ts_delta_full_scaled = scaler_delta.transform(ts_delta_full)
        ts_x_full_scaled = scaler_x.transform(ts_x_full)

        pred_delta_scaled = model.predict(
            n=n_test,
            series=ts_delta_full_scaled[:n_train],
            past_covariates=ts_x_full_scaled
        )

        pred_delta = scaler_delta.inverse_transform(pred_delta_scaled)

        # ======================================================
        # 7. RECONSTRUCT: level(t) = level(t-1) + delta(t)
        # ======================================================
        df_level = ts_level_full.to_dataframe().reset_index(drop=True)
        df_delta_pred = pred_delta.to_dataframe().reset_index(drop=True)

        test_start_idx = n_train
        df_test = df_level.iloc[test_start_idx:].copy()

        if n_train > 0:
            last_train_values = df_level.iloc[n_train - 1][state_vars].values
        else:
            last_train_values = np.zeros(len(state_vars))

        reconstructed = []
        current_state = last_train_values.copy()

        for i in range(n_test):
            delta_pred = df_delta_pred.iloc[i][delta_vars].values

            # Update state: y(t) = y(t-1) + Δy(t)
            current_state = current_state + delta_pred
            reconstructed.append(current_state)

        reconstructed = np.array(reconstructed)

        # ======================================================
        # Compute metrics
        # ======================================================
        y_true = df_test[state_vars].values
        y_pred = reconstructed

        for idx, var in enumerate(state_vars):
            r2 = r2_score(y_true[:, idx], y_pred[:, idx])
            mae = mean_absolute_error(y_true[:, idx], y_pred[:, idx])

            metrics.append({
                "CaseID": case_id,
                "State": var,
                "R2": r2,
                "MAE": mae,
            })

        # ======================================================
        # Visualize
        # ======================================================
        df_viz = df_test.copy()
        for idx, var in enumerate(state_vars):
            df_viz[f"{var}_pred"] = y_pred[:, idx]

        df_viz["CaseID"] = case_id

        timesteps_test = (
            df_all[df_all["CaseID"] == case_id]
            .sort_values("Timestep1and2")
            .iloc[test_start_idx:test_start_idx + n_test]["Timestep1and2"]
            .values
        )
        df_viz["Timestep1and2"] = timesteps_test

        viz_parts.append(df_viz)

    # ======================================================
    # Aggregate results
    # ======================================================
    viz_df = pd.concat(viz_parts, ignore_index=True)

    metrics_df = (
        pd.DataFrame(metrics)
        .groupby("State")
        .agg({
            "R2": "mean",
            "MAE": "mean"
        })
        .reset_index()
    )

    print("\n" + "=" * 60)
    print("✅ INFORMER TRAINING COMPLETE (DELTA MODE)")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print("=" * 60)

    return {
        "model": model,
        "metrics": metrics_df,
        "viz_df": viz_df,
        "viz_columns": state_vars,
        "method": "Informer (Delta)"
    }
