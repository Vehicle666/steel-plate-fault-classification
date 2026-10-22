"""Huấn luyện và đánh giá 5 mô hình OvR trên Steel Plates Faults.

Đặc điểm cho Colab:
  - Sau **mỗi mô hình** ("epoch") đều ghi checkpoint + báo cáo + hình.
  - `--push`  : commit & push ngay sau mỗi mô hình (cần GH_TOKEN).
  - `--resume`: bỏ qua các mô hình đã xong, train tiếp phần còn lại.

Ví dụ:
  python -m src.train --push                 # chạy mới, push từng bước
  python -m src.train --resume --push        # sau khi runtime chết
  python -m src.train --no-save              # chỉ in kết quả, không ghi file
"""

from __future__ import annotations

import argparse
import json
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
    CHECKPOINT_PATH,
    CHECKPOINT_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
)
from src.data import load_dataset, split_dataset
from src.git_sync import commit_and_push
from src.models import build_models

sns.set_theme(style="whitegrid", context="notebook")

STATE_VERSION = 1


def safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "")


# --------------------------------------------------------------------------- #
# Checkpoint
# --------------------------------------------------------------------------- #
def build_config(test_size: float, random_state: int, dataset) -> dict:
    return {
        "test_size": test_size,
        "random_state": random_state,
        "n_samples": len(dataset),
        "n_features": len(dataset.feature_names),
        "class_names": dataset.class_names,
    }


def load_checkpoint(config: dict) -> dict | None:
    if not CHECKPOINT_PATH.exists():
        return None
    state = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if state.get("version") != STATE_VERSION or state.get("config") != config:
        print("! Checkpoint cũ không cùng cấu hình (seed/test-size/dữ liệu) -> bỏ qua.")
        return None
    return state


def save_checkpoint(state: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# Huấn luyện & đánh giá
# --------------------------------------------------------------------------- #
def evaluate(name, model, X_train, X_test, y_train, y_test):
    """Fit một mô hình và trả về (metrics, pipeline, y_pred)."""
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
        axes, [cm, cm_norm], ["Số lượng", "Tỉ lệ theo hàng"], ["d", ".2f"]
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
    fig.savefig(FIGURES_DIR / f"confusion_matrix_{safe_name(name)}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_combined_confusion(results, labels):
    """Grid confusion matrix (chuẩn hoá) của các mô hình đã xong."""
    n = len(results)
    if n == 0:
        return
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
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(
        f"Confusion matrix (chuẩn hoá theo hàng) — {n}/5 mô hình OvR", fontsize=14
    )
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


def write_summary(df: pd.DataFrame) -> None:
    """Ghi results.csv + results.md từ các mô hình đã hoàn thành."""
    df.sort_values("f1_macro", ascending=False).to_csv(
        REPORTS_DIR / "results.csv", index=False, encoding="utf-8"
    )

    pretty = df.copy()
    for col in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]:
        pretty[col] = pretty[col].map(lambda v: f"{v:.4f}")
    for col in ["fit_time_s", "predict_time_s"]:
        pretty[col] = pretty[col].map(lambda v: f"{v:.3f}")

    (REPORTS_DIR / "results.md").write_text(
        "# Kết quả thực nghiệm\n\n"
        + pretty[["model", "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]]
        .sort_values("f1_macro", ascending=False)
        .to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--no-save", action="store_true", help="Chỉ in kết quả")
    parser.add_argument(
        "--push",
        action="store_true",
        help="Commit & push lên GitHub sau mỗi mô hình (cần GH_TOKEN)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Bỏ qua mô hình đã xong trong checkpoints/train_state.json",
    )
    args = parser.parse_args()

    for directory in (FIGURES_DIR, REPORTS_DIR, MODELS_DIR, CHECKPOINT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    labels = dataset.class_names
    X_train, X_test, y_train, y_test = split_dataset(
        dataset, test_size=args.test_size, random_state=args.random_state
    )

    config = build_config(args.test_size, args.random_state, dataset)
    state = load_checkpoint(config) if (args.resume and not args.no_save) else None
    if state is None:
        if args.resume:
            print("! Không có checkpoint phù hợp, bắt đầu train từ đầu.")
        state = {
            "version": STATE_VERSION,
            "config": config,
            "model_order": [],
            "completed": {},
        }

    print(f"Dữ liệu: {len(dataset)} mẫu | {len(dataset.feature_names)} đặc trưng")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    if state["completed"]:
        done = ", ".join(v["display_name"] for v in state["completed"].values())
        print(f"Checkpoint đã có: {done}")
    print()

    all_models = build_models(args.random_state)
    total = len(all_models)
    state["model_order"] = list(all_models.keys())

    rows: list[dict] = []
    predictions: list[tuple[str, np.ndarray, pd.Series]] = []
    pipelines: dict[str, Pipeline] = {}

    for index, (name, model) in enumerate(all_models.items(), start=1):
        key = safe_name(name)
        done = state["completed"].get(key)
        tag = f"[{index}/{total}]"

        if done and not args.no_save:
            metrics = done["metrics"]
            y_pred = np.asarray(done["y_pred"], dtype=object)
            print(f"{tag} {name}: đã xong trong checkpoint, bỏ qua "
                  f"(acc={metrics['accuracy']:.4f})")
        else:
            print(f"{tag} Đang huấn luyện {name} ...", end=" ", flush=True)
            metrics, pipeline, y_pred = evaluate(
                name, model, X_train, X_test, y_train, y_test
            )
            print(f"accuracy={metrics['accuracy']:.4f} | "
                  f"macro-F1={metrics['f1_macro']:.4f}")
            pipelines[name] = pipeline

            if not args.no_save:
                report = classification_report(
                    y_test, y_pred, labels=labels, digits=4, zero_division=0
                )
                (REPORTS_DIR / f"classification_report_{key}.txt").write_text(
                    f"Mô hình: {name}\n\n{report}", encoding="utf-8"
                )
                save_confusion_matrix(name, y_test, y_pred, labels)
                state["completed"][key] = {
                    "display_name": name,
                    "metrics": metrics,
                    "y_pred": [str(v) for v in y_pred],
                }
                save_checkpoint(state)

        rows.append(metrics)
        predictions.append((name, np.asarray(y_pred, dtype=object), y_test))

        if args.no_save:
            continue

        df = pd.DataFrame(rows)
        write_summary(df)
        save_combined_confusion(predictions, labels)
        save_metrics_chart(df)

        if args.push:
            commit_and_push(
                f"train: {tag} {name} — acc={metrics['accuracy']:.4f}, "
                f"macro-F1={metrics['f1_macro']:.4f}",
                paths=["reports", "checkpoints"],
                push=True,
            )

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

    best_name = df.iloc[0]["model"]
    if best_name in pipelines:
        joblib.dump(
            {
                "model": pipelines[best_name],
                "labels": labels,
                "features": dataset.feature_names,
            },
            MODELS_DIR / "best_model.joblib",
        )
        print(f"\nMô hình tốt nhất: {best_name} -> models/best_model.joblib")
    else:
        print("\n(Mô hình tốt nhất được load lại từ checkpoint — không fit lại nên "
              "không lưu .joblib. Chạy không --resume để có file model.)")

    print(f"Hình ảnh & báo cáo: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
