# src/config/tension_config.py
# -------- Paths --------
EXCEL_PATH = None
GOOGLE_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/1RVRcfM41Z7x-oZpf02Zxjg7e-N5db7KK/export?format=xlsx"
)

CASE_SHEETS = [
    "CASE1", "CASE2", "CASE3", "CASE4",
    "CASE6", "CASE7", "CASE8",
    "CASE10", "CASE11", "CASE12", "CASE13-Ni"
]

# -------- Luu hai loai data thuc nghiem, co time step khac nhau --------
COLUMNS_EXP_A = [
    "Timestep1",
    "Number of particle",
    "Composition",
    "Temperature",
    "Strain rate input",
    "Strain",
    "Stress",
    "Total energy",
    "Von Mises stress",
    "CaseID",
]

COLUMNS_EXP_B = [
    "Timestep2",
    "Total dislocation length",
    "Dislocation length.1/2<110>",
    "Dislocation length.1/3<100>",
    "Dislocation length.1/3<111>",
    "Dislocation length.1/6<110>",
    "Dislocation length.1/6<112>",
    "BCC phase",
    "FCC phase",
    "HCP phase",
    "CaseID",
]


MERGED_TIMESTEP1and2_CSV = "tension.csv"

# -------- Input models --------
PROCESS_FEATURES = [
    "Timestep1and2",
    "Number of particle",
    "Composition",
    "Temperature",
    "Strain rate input",
]



# -------- Output models --------
STATE_VARS = [
    # Mechanical (t1)
    "Strain",
    "Stress",
    "Von Mises stress",
    "Total energy",

    # Microstructure (t2)
    "Total dislocation length",
    "Dislocation length.1/2<110>",
    "Dislocation length.1/3<100>",
    "Dislocation length.1/3<111>",
    "Dislocation length.1/6<110>",
    "Dislocation length.1/6<112>",
    "BCC phase",
    "FCC phase",
    "HCP phase",
]


# -------- Grid search (fixed BEST PARAMS) --------
# param_grid_XGBRegressor = {
#     "estimator__max_depth": [6],
#     "estimator__n_estimators": [300],
#     "estimator__subsample": [0.5],
#     "estimator__colsample_bytree": [0.6],
#     "estimator__min_child_weight": [1],
# }
param_grid_XGBRegressor = {
    "estimator__max_depth": [4, 6, 8],
    "estimator__n_estimators": [300, 600, 1000],
    "estimator__subsample": [0.5, 0.6, 0.8],
    "estimator__colsample_bytree": [0.5, 0.6, 0.8],
    "estimator__min_child_weight": [1, 3, 5],
}
param_grid_LGBM = {
    "estimator__n_estimators": [500],
    "estimator__max_depth": [8,],
    "estimator__num_leaves": [31],
    "estimator__subsample": [0.6],
    "estimator__colsample_bytree": [0.6],
}
param_grid_CATBOOST = {
    "estimator__iterations": [500, 1000],
    "estimator__depth": [6, 8],
    "estimator__learning_rate": [0.03, 0.05],
    "estimator__l2_leaf_reg": [3, 5],
}
