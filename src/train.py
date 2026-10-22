"""Huấn luyện và đánh giá 5 mô hình OvR trên Steel Plates Faults.

Chạy: python -m src.train
Kết quả: reports/results.csv, reports/classification_report_*.txt,
         reports/figures/*.png, models/best_model.joblib
"""

from __future__ import annotations

import argparse
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data import load_dataset, split_dataset
from src.models import build_models

sns.set_theme(style="whitegrid", context="notebook")


def evaluate(name, model, X_train, X_test, y_train, y_test, labels):
    """Fit một mô hình và trả về (metrics, y_pred)."""
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", model)])

    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    predict_time = time.perf_counter() - t0

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
    }
    return metrics, pipeline, y_pred


def save_confusion_matrix(name, y_test, y_pred, labels):
    """Lưu confusion matrix (số đếm + chuẩn hoá theo hàng)."""
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, matrix, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Số lượng", "Tỉ lệ theo hàng"],
        ["d", ".2f"],
    ):
        sns.heatmap(
            matrix,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar=True,
            square=True,
        )
        ax.set_title(f"{name} — {title}")
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    safe = name.lower().replace(" ", "_").replace("-", "")
    path = FIGURES_DIR / f"confusion_matrix_{safe}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cm


def save_combined_confusion(results, labels):
    """Grid confusion matrix (chuẩn hoá) của cả 5 mô hình."""
    n = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.ravel()
    for ax, (name, y_pred, y_test) in zip(axes, results):
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar=False,
            square=True,
            vmin=0,
            vmax=1,
        )
        ax.set_title(name)
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
        ax.tick_params(axis="x", rotation=45)
    for ax in axes[len(results) :]:
        ax.axis("off")
    fig.suptitle("Confusion matrix (chuẩn hoá theo hàng) — 5 mô hình OvR", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrices_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metrics_chart(df):
    """Biểu đồ cột so sánh accuracy / macro-F1 / weighted-F1."""
    melted = df.melt(
        id_vars="model",
        value_vars=["accuracy", "f1_macro", "f1_weighted"],
        var_name="metric",
        value_name="score",
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=melted, x="model", y="score", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Điểm số")
    ax.set_title("So sánh hiệu năng các mô hình (test set)")
    ax.legend(title="", loc="lower right", ncols=3)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=1)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Không ghi file, chỉ in kết quả",
    )
    args = parser.parse_args()

    for directory in (FIGURES_DIR, REPORTS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    labels = dataset.class_names
    X_train, X_test, y_train, y_test = split_dataset(
        dataset, test_size=args.test_size, random_state=args.random_state
    )

    print(f"Dữ liệu: {len(dataset)} mẫu | {len(dataset.feature_names)} đặc trưng")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print("Phân bố lớp trong train:")
    print(y_train.value_counts().reindex(labels).to_string(), "\n")

    rows = []
    predictions = []
    pipelines = {}

    for name, model in build_models(args.random_state).items():
        print(f"Đang huấn luyện {name} ...", end=" ", flush=True)
        metrics, pipeline, y_pred = evaluate(
            name, model, X_train, X_test, y_train, y_test, labels
        )
        rows.append(metrics)
        predictions.append((name, y_pred, y_test))
        pipelines[name] = pipeline
        print(f"accuracy={metrics['accuracy']:.4f} | macro-F1={metrics['f1_macro']:.4f}")

        if not args.no_save:
            report = classification_report(
                y_test, y_pred, labels=labels, digits=4, zero_division=0
            )
            safe = name.lower().replace(" ", "_").replace("-", "")
            (REPORTS_DIR / f"classification_report_{safe}.txt").write_text(
                f"Mô hình: {name}\n\n{report}", encoding="utf-8"
            )
            save_confusion_matrix(name, y_test, y_pred, labels)

    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    df["rank"] = np.arange(1, len(df) + 1)

    pretty = df.copy()
    for col in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]:
        pretty[col] = pretty[col].map(lambda v: f"{v:.4f}")
    for col in ["fit_time_s", "predict_time_s"]:
        pretty[col] = pretty[col].map(lambda v: f"{v:.3f}")

    print("\n=== Bảng xếp hạng (theo macro-F1) ===")
    print(pretty.to_string(index=False))

    if args.no_save:
        return

    df.to_csv(REPORTS_DIR / "results.csv", index=False, encoding="utf-8")
    (REPORTS_DIR / "results.md").write_text(
        "# Kết quả thực nghiệm\n\n"
        + pretty[["model", "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]]
        .to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )

    save_combined_confusion(predictions, labels)
    save_metrics_chart(df)

    best_name = df.iloc[0]["model"]
    joblib.dump(
        {"model": pipelines[best_name], "labels": labels, "features": dataset.feature_names},
        MODELS_DIR / "best_model.joblib",
    )
    print(f"\nMô hình tốt nhất: {best_name} -> models/best_model.joblib")
    print(f"Hình ảnh & báo cáo: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
