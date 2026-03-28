# src/config/polishing_config.py

EXCEL_PATH = "data/TiCNi-Polishing-PP3-MACHINE - CASE1-12.xlsx"
GOOGLE_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/1EHPqVT6uOG64yJjCdtRLqy4fdTvQliet/export?format=xlsx"
)

CASE_SHEETS = [
    "CASE-1", "CASE-2", "CASE-3", "CASE-4",
    "CASE-5", "CASE-7", "CASE-8",
    "CASE-10", "CASE-11", "CASE-12"
]

# ============================================================
# RAW COLUMNS — KHỚP 100% VỚI EXCEL
# ============================================================

COLUMNS_EXP_A = [
    "Timestep1",
    "Slding speed",
    "Depth polishing",
    "Particle size",
    "Composition",
    "Temperature",          # <-- sửa từ Temp
    "PotEng",
    "KinEng",
    "Total energy",         # <-- sửa từ TotEng
    "Press",
    "Von Mises stress",     # <-- sửa từ v_mises
    "v_hydro",
    "Force_X",              # <-- sửa từ v_fx
    "Force_Y",              # <-- sửa từ v_fy
    "Force_Z",              # <-- sửa từ v_fz
    "CaseID"
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
    "CaseID"
]

MERGED_TIMESTEP1and2_CSV = "polishing.csv"

# ============================================================
# INPUT — PROCESS PARAMETERS (THUẦN)
# ============================================================
PROCESS_FEATURES = [
    "Timestep1and2",
    "Slding speed",
    "Depth polishing",
    "Particle size",
    "Composition",
    "Temperature",
]

# ============================================================
# OUTPUT — PHYSICAL + MICROSTRUCTURE
# ============================================================
STATE_VARS = [
    # ---- Energy & stress ----
    "Total energy",
    "Von Mises stress",

    # ---- Forces ----
    "Force_X",
    "Force_Y",
    "Force_Z",

    # ---- Dislocation ----
    "Total dislocation length",
    "Dislocation length.1/2<110>",
    "Dislocation length.1/6<110>",
    "Dislocation length.1/6<112>",

    # ---- Phase ----
    "BCC phase",
    "FCC phase",
    "HCP phase",
]

# ============================================================
# GRID SEARCH
# ============================================================
# param_grid_XGBRegressor = {
#     "estimator__max_depth": [6],
#     "estimator__n_estimators": [300],
#     "estimator__subsample": [0.5],
#     "estimator__colsample_bytree": [0.8],
#     "estimator__min_child_weight": [5],
# }

param_grid_XGBRegressor = {
    "estimator__max_depth": [4, 6, 8],
    "estimator__n_estimators": [300, 600, 1000],
    "estimator__subsample": [0.5, 0.6, 0.8],
    "estimator__colsample_bytree": [0.5, 0.6, 0.8],
    "estimator__min_child_weight": [1, 3, 5],
}

param_grid_LGBM = {
    "estimator__n_estimators": [500, 1000],
    "estimator__max_depth": [8,16],
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
