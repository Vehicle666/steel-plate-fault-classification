"""Định nghĩa 5 mô hình, tất cả bọc trong One-vs-Rest (OvR).

Ghi chú: Random Forest / Extra Trees vốn hỗ trợ đa lớp sẵn, nhưng vẫn bọc OvR
để đồng nhất chiến lược theo yêu cầu đề tài. Gradient Boosting của sklearn
chỉ hỗ trợ nhị phân nên OvR là bắt buộc.
"""

from __future__ import annotations

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.config import RANDOM_STATE


def build_models(random_state: int = RANDOM_STATE) -> dict[str, OneVsRestClassifier]:
    """Trả về dict {tên mô hình: estimator OvR}."""
    return {
        "SVM": OneVsRestClassifier(
            SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"),
        ),
        "k-NN": OneVsRestClassifier(
            KNeighborsClassifier(n_neighbors=7, weights="uniform"),
        ),
        "Random Forest": OneVsRestClassifier(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        "Extra Trees": OneVsRestClassifier(
            ExtraTreesClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        "Gradient Boosting": OneVsRestClassifier(
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=2,
                min_samples_leaf=3,
                subsample=0.8,
                n_iter_no_change=10,
                validation_fraction=0.1,
                tol=0.001,
                random_state=random_state,
            ),
        ),
    }
