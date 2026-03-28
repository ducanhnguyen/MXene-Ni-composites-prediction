# src/main.py
import re
import warnings
from datetime import datetime
from io import BytesIO

import pandas as pd
import requests

from src.XGB import run_xgb
from src.analysis.correlation import (
    export_and_plot_input_output_correlation
)
from src.analysis.statistics import export_statistics_and_visualize
from src.config import tension_config, polishing_config, laser_config
from src.data.build_delta_dataset import build_delta_train_test
from src.models.export_utils import export_gridsearch_results, export_gridsearch_results_lgbm
from src.models.informer_model import run_informer
from src.models.kernel_ar_model import run_kernel_ar
from src.models.lgbm_model import run_lgbm
from src.models.rf_ar_model import run_rf_ar
from src.models.ridge_ar_model import run_ridge_ar
from src.models.varx_model import run_varx
from src.utils import plot_case_predictions, plot_predictions_by_output, export_best_param_txt, export_best_result_eval_csv

warnings.filterwarnings('ignore')

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
torch.set_default_dtype(torch.float32)

EXCEL_PATH = None
GOOGLE_SHEET_URL = None
MERGED_TIMESTEP1and2_CSV = None
CASE_SHEETS = None
COLUMNS_EXP_A = None
COLUMNS_EXP_B = None
STATE_VARS = None
PROCESS_FEATURES = None
param_grid = None

DATA_TYPE = "tension"  # tension, polishing, laser
METHOD_NAME = "RF-AR"   # XGBRegressor | LightGBM | VARX | RIDGE-AR | KERNEL-AR | RF-AR | Informer

if DATA_TYPE == "polishing":
    EXCEL_PATH = polishing_config.EXCEL_PATH
    GOOGLE_SHEET_URL = polishing_config.GOOGLE_SHEET_URL
    MERGED_TIMESTEP1and2_CSV = polishing_config.MERGED_TIMESTEP1and2_CSV
    CASE_SHEETS = polishing_config.CASE_SHEETS
    COLUMNS_EXP_A = polishing_config.COLUMNS_EXP_A
    COLUMNS_EXP_B = polishing_config.COLUMNS_EXP_B
    STATE_VARS = polishing_config.STATE_VARS
    PROCESS_FEATURES = polishing_config.PROCESS_FEATURES

elif DATA_TYPE == "tension":
    EXCEL_PATH = tension_config.EXCEL_PATH
    GOOGLE_SHEET_URL = tension_config.GOOGLE_SHEET_URL
    MERGED_TIMESTEP1and2_CSV = tension_config.MERGED_TIMESTEP1and2_CSV
    CASE_SHEETS = tension_config.CASE_SHEETS
    COLUMNS_EXP_A = tension_config.COLUMNS_EXP_A
    COLUMNS_EXP_B = tension_config.COLUMNS_EXP_B
    STATE_VARS = tension_config.STATE_VARS
    PROCESS_FEATURES = tension_config.PROCESS_FEATURES

elif DATA_TYPE == "laser":
    EXCEL_PATH = laser_config.EXCEL_PATH
    GOOGLE_SHEET_URL = laser_config.GOOGLE_SHEET_URL
    MERGED_TIMESTEP1and2_CSV = laser_config.MERGED_TIMESTEP1and2_CSV
    CASE_SHEETS = laser_config.CASE_SHEETS
    COLUMNS_EXP_A = laser_config.COLUMNS_EXP_A
    COLUMNS_EXP_B = laser_config.COLUMNS_EXP_B
    STATE_VARS = laser_config.STATE_VARS
    PROCESS_FEATURES = laser_config.PROCESS_FEATURES

# -------- Other config --------
TRAIN_RATIO = 0.6
RANDOM_STATE = 42

label_map = {
    "Timestep1and2": "Timestep",
    # =======================
    # Thermo / Mechanical
    # =======================
    "Temp": "Temperature",
    "v_strain": "Strain",
    "v_stress_GPa": "Stress",
    "TotEng": "Total energy",
    "v_mises": "Von Mises stress",

    # Forces
    "v_fx": "Force_X",
    "v_fy": "Force_Y",
    "v_fz": "Force_Z",

    # =======================
    # Dislocation metrics
    # =======================
    "DislocationAnalysis.total_line_length": "Total dislocation length",
    "DislocationAnalysis.length.1/2<110>": "Dislocation length.1/2<110>",
    "DislocationAnalysis.length.1/3<100>": "Dislocation length.1/3<100>",
    "DislocationAnalysis.length.1/3<111>": "Dislocation length.1/3<111>",
    "DislocationAnalysis.length.1/6<110>": "Dislocation length.1/6<110>",
    "DislocationAnalysis.length.1/6<112>": "Dislocation length.1/6<112>",

    # =======================
    # Phase / CNA
    # =======================
    "CommonNeighborAnalysis.counts.BCC": "BCC phase",
    "CommonNeighborAnalysis.counts.FCC": "FCC phase",
    "CommonNeighborAnalysis.counts.HCP": "HCP phase",
}


# ============================================================
# INPUT–OUTPUT CORRELATION ANALYSIS
# ============================================================
BASE_OUT_DIR = "out"
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
OUT_DIR = os.path.join(BASE_OUT_DIR, METHOD_NAME + "_" + DATA_TYPE+"_"+ timestamp)
os.makedirs(OUT_DIR, exist_ok=True)
print("[OUT_DIR]", OUT_DIR)

# Load Excel
if EXCEL_PATH is not None and os.path.exists(EXCEL_PATH):
    print(f"[OK] Load local Excel: {EXCEL_PATH}")
    excel_file = EXCEL_PATH
else:
    print("[WARN] Local Excel not found. Downloading from Google Sheets...")
    response = requests.get(GOOGLE_SHEET_URL)
    response.raise_for_status()
    excel_file = BytesIO(response.content)
    print("[OK] Google Sheets downloaded as Excel")

# Read case sheets
df_list = []
for sheet in CASE_SHEETS:
    try:
        df_sheet = pd.read_excel(excel_file, sheet_name=sheet)
        df_sheet.columns = df_sheet.columns.str.strip()
        df_sheet["CaseID"] = sheet
        df_list.append(df_sheet)
        print(f"[OK] Loaded {sheet}: {len(df_sheet)} rows")
    except Exception as e:
        print(f"[ERROR] Cannot load {sheet} → {e}")



df_all = pd.concat(df_list, ignore_index=True)

# BUILD MERGED DATASET
df_input = df_all[COLUMNS_EXP_A].dropna(subset=["Timestep1"]).copy()
df_output = df_all[COLUMNS_EXP_B].dropna(subset=["Timestep2"]).copy()

df_input = df_input.rename(columns={"Timestep1": "Timestep"})
df_output = df_output.rename(columns={"Timestep2": "Timestep"})

merge_keys = ["Timestep", "CaseID"]

df_merged = pd.merge(df_input, df_output, on=merge_keys, how="inner")
df_merged = df_merged.rename(columns={"Timestep": "Timestep1and2"})
df_merged["Timestep1and2"] = df_merged["Timestep1and2"].astype(int)

df_merged.to_csv(os.path.join(OUT_DIR,MERGED_TIMESTEP1and2_CSV), index=False)
print(f"[OK] Saved merged dataset to: {os.path.join(OUT_DIR,MERGED_TIMESTEP1and2_CSV)}")


# CLEAN NUMERIC DATA
def clean_numeric(x):
    if isinstance(x, str):
        x = re.sub(r"[^0-9eE\+\-\.]", "", x)
        parts = x.split(".")
        if len(parts) > 2:
            x = parts[0] + "." + "".join(parts[1:])
    return x


df = pd.read_csv(os.path.join(OUT_DIR,MERGED_TIMESTEP1and2_CSV))

for col in df.columns:
    if col != "CaseID":
        df[col] = df[col].apply(clean_numeric)
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.fillna(df.mean(numeric_only=True))


export_and_plot_input_output_correlation(
    df=df,
    process_features=PROCESS_FEATURES,
    state_vars=STATE_VARS,
    label_map=label_map,
    out_dir=OUT_DIR,
    method="spearman"
)

export_and_plot_input_output_correlation(
    df=df,
    process_features=PROCESS_FEATURES,
    state_vars=STATE_VARS,
    label_map=label_map,
    out_dir=OUT_DIR,
    method="pearson"
)
# ============================================================
# DESCRIPTIVE STATISTICS + VISUALIZATION
# (NO TIMESERIES, ALL FILENAMES SANITIZED)
# ============================================================
ANALYSIS_DIR = os.path.join(OUT_DIR, "analysis")

export_statistics_and_visualize(
    df=df,
    process_features=PROCESS_FEATURES,
    state_vars=STATE_VARS,
    out_dir=ANALYSIS_DIR
)

# DEFINE MODEL INPUT / OUTPUT
lag_features = [f"{col}_lag1" for col in STATE_VARS]

INPUT_FEATURES = PROCESS_FEATURES + lag_features
OUTPUT_TARGETS = [f"d_{col}" for col in STATE_VARS]

# TEMPORAL TRAIN / TEST SPLIT (WITHIN CASE)
train_parts, test_parts = [], []

for case_id, df_case in df.groupby("CaseID"):
    split_idx = int(TRAIN_RATIO * len(df_case))
    train_parts.append(df_case.iloc[:split_idx])
    test_parts.append(df_case.iloc[split_idx:])

# FINETUNE
model = None
if METHOD_NAME == "XGBRegressor":
    if DATA_TYPE == "polishing":
        param_grid = polishing_config.param_grid_XGBRegressor
    elif DATA_TYPE == "tension":
        param_grid = tension_config.param_grid_XGBRegressor
    elif DATA_TYPE == "laser":
        param_grid = laser_config.param_grid_XGBRegressor
    data = build_delta_train_test(
        df=df,
        state_vars=STATE_VARS,
        process_features=PROCESS_FEATURES,
        train_ratio=TRAIN_RATIO
    )

    out = run_xgb(
        train_df=data["train_df"],
        test_df=data["test_df"],
        input_features=data["input_features"],
        state_vars=STATE_VARS,
        param_grid=param_grid,
        random_state=RANDOM_STATE
    )
elif METHOD_NAME == "LightGBM":
    if DATA_TYPE == "polishing":
        param_grid = polishing_config.param_grid_LGBM
    elif DATA_TYPE == "tension":
        param_grid = tension_config.param_grid_LGBM
    elif DATA_TYPE == "laser":
        param_grid = laser_config.param_grid_LGBM

    data = build_delta_train_test(
        df=df,
        state_vars=STATE_VARS,
        process_features=PROCESS_FEATURES,
        train_ratio=TRAIN_RATIO
    )
    out = run_lgbm(
        train_df=data["train_df"],
        test_df=data["test_df"],
        input_features=data["input_features"],
        state_vars=STATE_VARS,
        param_grid=param_grid,
        random_state=RANDOM_STATE
    )

elif METHOD_NAME == "VARX":
    data = build_delta_train_test(
        df=df,
        state_vars=STATE_VARS,
        process_features=PROCESS_FEATURES,
        train_ratio=TRAIN_RATIO
    )
    out = run_varx(
        train_df=data["train_df"],
        test_df=data["test_df"],
        state_vars=STATE_VARS,
        process_features=PROCESS_FEATURES,
        maxlags=1   # rất nên giữ =1 cho vật liệu
    )
elif METHOD_NAME == "RIDGE-AR":
    data = build_delta_train_test(
        df=df,
        state_vars=STATE_VARS,
        process_features=PROCESS_FEATURES,
        train_ratio=TRAIN_RATIO
    )
    out = run_ridge_ar(
        train_df=data["train_df"],
        test_df=data["test_df"],
        state_vars=STATE_VARS,
        process_features=PROCESS_FEATURES,
        alpha=1.0
    )
elif METHOD_NAME == "KERNEL-AR":
    data = build_delta_train_test(
        df=df,
        state_vars=STATE_VARS,
        process_features=PROCESS_FEATURES,
        train_ratio=TRAIN_RATIO
    )
    out = run_kernel_ar(
        train_df=data["train_df"],
        test_df=data["test_df"],
        state_vars=STATE_VARS,
        process_features=PROCESS_FEATURES,
        alpha=1.0,
        gamma=0.1
    )
elif METHOD_NAME == "RF-AR":
    data = build_delta_train_test(
        df=df,
        state_vars=STATE_VARS,
        process_features=PROCESS_FEATURES,
        train_ratio=TRAIN_RATIO
    )
    out = run_rf_ar(
        train_df=data["train_df"],
        test_df=data["test_df"],
        state_vars=STATE_VARS,
        process_features=PROCESS_FEATURES,
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5
    )
elif METHOD_NAME == "Informer":
    data = build_delta_train_test(
        df=df,
        state_vars=STATE_VARS,
        process_features=PROCESS_FEATURES,
        train_ratio=TRAIN_RATIO
    )

    out = run_informer(
        train_df=data["train_df"],
        test_df=data["test_df"],
        state_vars=STATE_VARS,
        # process_features=data["process_features"],
        process_features=PROCESS_FEATURES,
        input_chunk_length=40,
        output_chunk_length=1,
        n_epochs=30,
        batch_size=16,
        random_state=42
    )

else:
    raise ValueError(...)



# ============================================================
# Export
# ============================================================
out_path = os.path.join(OUT_DIR, "gridsearch_results_full_metrics.csv")
if METHOD_NAME == "XGBRegressor":
    export_gridsearch_results(out["grid"], out_path)
elif METHOD_NAME == "LightGBM":
    export_gridsearch_results_lgbm(out["grid"], out_path)
elif METHOD_NAME == "VARX":
    print("[INFO] VARX model – no grid search to export.")
elif METHOD_NAME == "RIDGE-AR":
    print("[INFO] RIDGE-AR model – no grid search to export.")
elif METHOD_NAME == "KERNEL-AR":
    print("[INFO] RIDGE-AR model – no grid search to export.")
elif METHOD_NAME == "RF-AR":
    print("[INFO] RF-AR model – no grid search to export.")
# else:
#     raise ValueError(f"Unsupported METHOD_NAME: {METHOD_NAME}")
print(f"[OK] Saved {out_path}")

# xuất best params
export_best_param_txt(
    out_dir=OUT_DIR,
    method_name=out["method"],
    grid=out.get("grid", None)
)

# xuất bảng evaluation theo output
export_best_result_eval_csv(
    out_dir=OUT_DIR,
    metrics_df=out["metrics"]
)

# ============================================================
# PLOT ALL CASES
# ============================================================
if METHOD_NAME in ["XGBRegressor", "LightGBM"]:
    print(out["metrics"].sort_values("R2", ascending=False))

elif METHOD_NAME == "Informer":
    print(
        out["metrics"]
        .sort_values("MAE", ascending=True)  # ✅ Đổi "MAE" → "AbsError"
    )

else:
    print(out["metrics"].sort_values("R2", ascending=False))


BY_CASE_DIR = os.path.join(OUT_DIR, "by_case")
BY_OUTPUT_DIR = os.path.join(OUT_DIR, "by_output")

os.makedirs(BY_CASE_DIR, exist_ok=True)
os.makedirs(BY_OUTPUT_DIR, exist_ok=True)

for case_id, df_case in out["viz_df"].groupby("CaseID"):
    print(f"[PLOT][CASE] {case_id}")
    plot_case_predictions(
        df_case=df_case,
        state_vars=out["viz_columns"],
        case_id=case_id,
        label_map=label_map,
        ncols=3,
        out_dir=BY_CASE_DIR,   # ⬅️ đổi ở đây
        save=True
    )

# ============================================================
# PLOT BY OUTPUT (ALL CASES)
# ============================================================
for out_var in out["viz_columns"]:
    plot_predictions_by_output(
        viz_df=out["viz_df"],
        output_var=out_var,
        label_map=label_map,
        out_dir=BY_OUTPUT_DIR,
        save=True,
        ncols=3
    )