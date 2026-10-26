"""Đọc và tiền xử lý dataset Steel Plates Faults.

File gốc `Faults.NNA` không có header: 27 đặc trưng + 7 cột nhãn one-hot.
Nhiệm vụ chính ở đây là chuyển 7 cột one-hot về **một nhãn duy nhất**.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    LABEL_COLS,
    NNA_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
    VAR_PATH,
)


@dataclass(frozen=True)
class Dataset:
    """Tập dữ liệu đã tách nhãn, sẵn sàng cho sklearn."""

    X: pd.DataFrame
    y: pd.Series
    feature_names: list[str]
    class_names: list[str]

    def __len__(self) -> int:
        return len(self.X)


def load_raw() -> pd.DataFrame:
    """Đọc Faults.NNA và gán tên cột từ Faults27x7_var."""
    columns = VAR_PATH.read_text(encoding="utf-8").split()
    df = pd.read_csv(NNA_PATH, sep="\t", header=None, names=columns)
    if df.shape[1] != 34:
        raise ValueError(f"Kỳ vọng 34 cột, nhận được {df.shape[1]}")
    return df


def load_dataset() -> Dataset:
    """Trả về X (27 đặc trưng) và y (một nhãn chuỗi) từ one-hot."""
    df = load_raw()

    onehot = df[LABEL_COLS]
    # Mỗi dòng phải có đúng một nhãn = 1
    if not (onehot.sum(axis=1) == 1).all():
        raise ValueError("Có dòng không có đúng 1 nhãn one-hot")

    y = onehot.idxmax(axis=1)
    feature_names = [c for c in df.columns if c not in LABEL_COLS]

    return Dataset(
        X=df[feature_names],
        y=y,
        feature_names=feature_names,
        class_names=sorted(y.unique().tolist()),
    )


def split_dataset(
    dataset: Dataset,
    test_size: float = TEST_SIZE,
    validation_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Chia stratified train/validation/test; validation và test không dùng để fit."""
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size và validation_size phải nằm giữa 0 và 1")
    if test_size + validation_size >= 1:
        raise ValueError("Tổng test_size và validation_size phải nhỏ hơn 1")

    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.y,
    )
    relative_validation_size = validation_size / (1 - test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=relative_validation_size,
        random_state=random_state,
        stratify=y_train_validation,
    )
    return X_train, X_validation, X_test, y_train, y_validation, y_test
