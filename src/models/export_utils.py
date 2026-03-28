# src/models/export_utils.py
import pandas as pd


def export_gridsearch_results(grid, out_csv_path):
    cv = pd.DataFrame(grid.cv_results_)

    keep_cols = [
        "param_estimator__n_estimators",
        "param_estimator__max_depth",
        "param_estimator__subsample",
        "param_estimator__colsample_bytree",

        "mean_test_R2",
        "std_test_R2",
        "mean_test_MAE",
        "mean_test_MSE",
        "mean_test_RMSE",
        "rank_test_R2",

        "mean_fit_time",
        "std_fit_time",
    ]

    cv = cv[keep_cols]

    # đổi dấu metric lỗi
    for col in ["mean_test_MAE", "mean_test_MSE", "mean_test_RMSE"]:
        cv[col] = -cv[col]

    cv = cv.rename(columns={
        "param_estimator__n_estimators": "param_n_estimators",
        "param_estimator__max_depth": "param_max_depth",
        "param_estimator__subsample": "param_subsample",
        "param_estimator__colsample_bytree": "param_colsample_bytree",
        "mean_test_R2": "mean_R2",
        "std_test_R2": "std_R2",
        "rank_test_R2": "rank",
    })

    cv = cv.sort_values("rank").reset_index(drop=True)
    cv.to_csv(out_csv_path, index=False)

    return cv

# src/models/gridsearch_utils_lgbm.py
import pandas as pd


def export_gridsearch_results_lgbm(grid, out_csv_path):
    cv = pd.DataFrame(grid.cv_results_)

    keep_cols = [
        "param_estimator__n_estimators",
        "param_estimator__max_depth",
        "param_estimator__num_leaves",
        "param_estimator__subsample",
        "param_estimator__colsample_bytree",

        "mean_test_R2",
        "std_test_R2",
        "mean_test_MAE",

        "rank_test_R2",
        "mean_fit_time",
        "std_fit_time",
    ]

    cv = cv[keep_cols]

    # đổi dấu MAE
    cv["mean_test_MAE"] = -cv["mean_test_MAE"]

    cv = cv.rename(columns={
        "param_estimator__n_estimators": "param_n_estimators",
        "param_estimator__max_depth": "param_max_depth",
        "param_estimator__num_leaves": "param_num_leaves",
        "param_estimator__subsample": "param_subsample",
        "param_estimator__colsample_bytree": "param_colsample_bytree",
        "mean_test_R2": "mean_R2",
        "std_test_R2": "std_R2",
        "rank_test_R2": "rank",
    })

    cv = cv.sort_values("rank").reset_index(drop=True)
    cv.to_csv(out_csv_path, index=False)

    return cv
