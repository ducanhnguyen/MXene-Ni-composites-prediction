# /Users/ducanhnguyen/PycharmProjects/cuong/src/data/build_level_timeseries.py
import pandas as pd


def build_level_train_test(
    df,
    state_vars,
    process_features,
    train_ratio=0.6
):
    df = df.sort_values(["CaseID", "Timestep1and2"]).copy()

    train_parts, test_parts = [], []
    for _, df_case in df.groupby("CaseID"):
        split = int(train_ratio * len(df_case))
        train_parts.append(df_case.iloc[:split])
        test_parts.append(df_case.iloc[split:])

    return {
        "train_df": pd.concat(train_parts),
        "test_df": pd.concat(test_parts),
        "state_vars": state_vars,
        "process_features": process_features
    }
