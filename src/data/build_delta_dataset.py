import pandas as pd


def build_delta_train_test(
    df,
    state_vars,
    process_features,
    train_ratio=0.6
):
    df = df.sort_values(["CaseID", "Timestep1and2"]).copy()

    # Lag + Delta
    for col in state_vars:
        df[f"{col}_lag1"] = df.groupby("CaseID")[col].shift(1)
        df[f"d_{col}"] = df[col] - df.groupby("CaseID")[col].shift(1)

    df = df.dropna().reset_index(drop=True)

    lag_features = [f"{c}_lag1" for c in state_vars]
    input_features = process_features + lag_features
    output_targets = [f"d_{c}" for c in state_vars]

    train_parts, test_parts = [], []
    for _, df_case in df.groupby("CaseID"):
        split = int(train_ratio * len(df_case))
        train_parts.append(df_case.iloc[:split])
        test_parts.append(df_case.iloc[split:])

    return {
        "train_df": pd.concat(train_parts),
        "test_df": pd.concat(test_parts),
        "input_features": input_features,
        "output_targets": output_targets
    }
