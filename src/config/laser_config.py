# src/config/laser_config.py
# ============================================================
# PATHS
# ============================================================

EXCEL_PATH = None
GOOGLE_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "183KgvmUCT79BmPexeQbsxTLjefvmwUuR/export?format=xlsx"
)
MERGED_TIMESTEP1and2_CSV = "laser.csv"

# ============================================================
# CASE SHEETS
# ============================================================

CASE_SHEETS = [
    "CS0", "CS1", "CS2", "CS3",
    "CS4", "CS6", "CS7", "CS8",
]

# ============================================================
# RAW COLUMNS (EXACT EXCEL HEADERS)
# ============================================================

# ---- Experiment A (t1)
COLUMNS_EXP_A = [
    "Timestep1",
    "Number of particle",
    "Strain rate",
    "Intensity",
    "Composition",
    "Temperature",

    "Strain",
    "Stress",
    "v_p1",
    "v_p2",
    "v_p3",
    "v_p4",
    "Total energy",
    "Press",
    "Von Mises stress",
    "v_hydro",

    "CaseID",
]

# ---- Experiment B (t2)
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

STATE_VARS = [
    # Mechanical response
    "Strain",
    "Stress",
    "Von Mises stress",
    "Total energy",

    # Microstructure
    "Total dislocation length",
    "Dislocation length.1/2<110>",
    "Dislocation length.1/6<110>",
    "Dislocation length.1/6<112>",

    # Phase
    "BCC phase",
    "FCC phase",
    "HCP phase",
]

PROCESS_FEATURES = [
    # "Timestep1and2",        # Timestep
    # "Number of particle",
    # "Strain rate",
    "Intensity",
    "Composition",
    "Temperature",
]
# STATE_VARS = [
#     "Timestep1and2",
#     "Number of particle",
#     "Intensity",
#     "Composition",
#     "Temperature",
#     "Strain rate",
#     "Strain",
#     "Stress",
# ]
#
# PROCESS_FEATURES = STATE_VARS.copy()

# ============================================================
# FINETUNE (XGB)
# ============================================================
# param_grid_XGBRegressor = {
#     "estimator__max_depth": [4],
#     "estimator__n_estimators": [300],
#     "estimator__subsample": [0.8],
#     "estimator__colsample_bytree": [0.5],
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
    "estimator__max_depth": [8],
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
