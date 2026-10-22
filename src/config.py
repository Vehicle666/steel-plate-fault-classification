"""Đường dẫn và hằng số dùng chung cho toàn project."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "steel_plates_faults"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

NNA_PATH = RAW_DIR / "Faults.NNA"
VAR_PATH = RAW_DIR / "Faults27x7_var"

# 7 cột nhãn one-hot ở cuối file dữ liệu
LABEL_COLS = [
    "Pastry",
    "Z_Scratch",
    "K_Scatch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
