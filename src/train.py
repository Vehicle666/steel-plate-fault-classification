"""Huấn luyện, chọn mô hình bằng validation và đánh giá test một lần.

Ví dụ:
  python -m src.train --push
  python -m src.train --resume --push
  python -m src.train --no-save
"""

from __future__ import annotations

import argparse
import hashlib
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
    CHECKPOINT_DIR,
    CHECKPOINT_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    NNA_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
    VALIDATION_SIZE,
    VAR_PATH,
)
from src.data import load_dataset, split_dataset
from src.git_sync import commit_and_push
from src.models import build_models

sns.set_theme(style="whitegrid", context="notebook")

STATE_VERSION = 3


def safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "")


def build_config(
    test_size: float, validation_size: float, random_state: int, dataset
) -> dict:
    return {
        "test_size": test_size,
        "validation_size": validation_size,
        "random_state": random_state,
        "n_samples": len(dataset),
        "n_features": len(dataset.feature_names),
        "class_names": dataset.class_names,
        "dataset_sha256": hashlib.sha256(
            NNA_PATH.read_bytes() + b"\0" + VAR_PATH.read_bytes()
        ).hexdigest(),
    }


def load_checkpoint(config: dict) -> dict | None:
    if not CHECKPOINT_PATH.exists():
        return None
    state = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if state.get("version") != STATE_VERSION or state.get("config") != config:
        print("! Checkpoint cũ không cùng cấu hình/dữ liệu -> bỏ qua.")
        return None
    return state


def save_checkpoint(state: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = CHECKPOINT_PATH.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(CHECKPOINT_PATH)


def classification_metrics(prefix: str, y_true, y_pred) -> dict:
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        f"{prefix}_f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        f"{prefix}_f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }


def evaluate_candidate(model, X_train, X_validation, y_train, y_validation):
    """Fit một ứng viên trên train; chỉ validation được dùng để chọn mô hình."""
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", model)])

    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_time = time.perf_counter() - start

    train_pred = pipeline.predict(X_train)
    start = time.perf_counter()
    validation_pred = pipeline.predict(X_validation)
    predict_time = time.perf_counter() - start

    metrics = {
        **classification_metrics("train", y_train, train_pred),
        **classification_metrics("validation", y_validation, validation_pred),
        "f1_macro_gap": f1_score(
            y_train, train_pred, average="macro", zero_division=0
        )
        - f1_score(y_validation, validation_pred, average="macro", zero_division=0),
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
    }
    return metrics, validation_pred


def evaluate_final(model, X_train, X_validation, X_test, y_train, y_validation, y_test):
    """Fit ứng viên được chọn trên train+validation rồi đánh giá test một lần."""
    X_train_final = pd.concat([X_train, X_validation])
    y_train_final = pd.concat([y_train, y_validation])
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", model)])

    start = time.perf_counter()
    pipeline.fit(X_train_final, y_train_final)
    fit_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    predict_time = time.perf_counter() - start
    metrics = {
        **classification_metrics("test", y_test, y_pred),
        "final_fit_time_s": fit_time,
        "test_predict_time_s": predict_time,
    }
    return pipeline, y_pred, metrics


def save_confusion_matrix(name, y_true, y_pred, labels, suffix=""):
    """Lưu confusion matrix đếm số lượng và chuẩn hoá theo hàng."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    normalized = matrix.astype(float) / matrix.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, values, title, fmt in zip(
        axes, [matrix, normalized], ["Số lượng", "Tỉ lệ theo hàng"], ["d", ".2f"]
    ):
        sns.heatmap(
            values,
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
    fig.savefig(
        FIGURES_DIR / f"confusion_matrix_{safe_name(name)}{suffix}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_combined_confusion(results, labels):
    """Grid confusion matrix validation của các ứng viên đã hoàn thành."""
    n = len(results)
    if n == 0:
        return
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.ravel()
    for ax, (name, y_pred, y_true) in zip(axes, results):
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        normalized = matrix.astype(float) / matrix.sum(axis=1, keepdims=True).clip(min=1)
        sns.heatmap(
            normalized,
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
        f"Confusion matrix validation (chuẩn hoá theo hàng) — {n}/5 mô hình",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "confusion_matrices_all.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def save_metrics_chart(df):
    """So sánh macro-F1 train/validation để nhìn thấy khoảng cách tổng quát hoá."""
    melted = df.melt(
        id_vars="model",
        value_vars=["train_f1_macro", "validation_f1_macro"],
        var_name="split",
        value_name="macro_f1",
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=melted, x="model", y="macro_f1", hue="split", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Train và validation — chọn mô hình theo validation macro-F1")
    ax.legend(title="", loc="lower right", ncols=2)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=1)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "metrics_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def write_summary(df: pd.DataFrame) -> None:
    """Ghi bảng train/validation và điểm test cuối cùng của mô hình được chọn."""
    for column in (
        "test_accuracy",
        "test_balanced_accuracy",
        "test_f1_macro",
        "test_f1_weighted",
    ):
        if column not in df:
            df[column] = np.nan
    ordered = df.sort_values("validation_f1_macro", ascending=False)
    ordered.to_csv(REPORTS_DIR / "results.csv", index=False, encoding="utf-8")

    columns = [
        "model",
        "train_f1_macro",
        "validation_f1_macro",
        "f1_macro_gap",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_f1_macro",
        "test_f1_weighted",
    ]
    pretty = ordered[columns].copy()
    for column in columns[1:]:
        pretty[column] = pretty[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.4f}"
        )
    (REPORTS_DIR / "results.md").write_text(
        "# Kết quả thực nghiệm\n\n"
        "Mô hình được chọn theo validation macro-F1; test chỉ đánh giá mô hình được chọn.\n\n"
        + pretty.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--validation-size", type=float, default=VALIDATION_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--no-save", action="store_true", help="Không ghi kết quả ra file")
    parser.add_argument(
        "--push",
        action="store_true",
        help="Commit & push báo cáo, checkpoint và model lên GitHub",
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
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_dataset(
        dataset,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )

    config = build_config(
        args.test_size, args.validation_size, args.random_state, dataset
    )
    state = load_checkpoint(config) if args.resume else None
    if state is None:
        if args.resume:
            print("! Không có checkpoint phù hợp, bắt đầu train từ đầu.")
        state = {
            "version": STATE_VERSION,
            "config": config,
            "model_order": [],
            "completed": {},
        }
        if not args.no_save:
            save_checkpoint(state)
            if args.push:
                commit_and_push(
                    "train: initialize checkpoint",
                    paths=["checkpoints"],
                    push=True,
                )

    print(
        f"Dữ liệu: {len(dataset)} mẫu | {len(dataset.feature_names)} đặc trưng"
    )
    print(
        f"Train: {len(X_train)} | Validation: {len(X_validation)} | "
        f"Test: {len(X_test)}"
    )
    if state["completed"]:
        done = ", ".join(v["display_name"] for v in state["completed"].values())
        print(f"Checkpoint đã có: {done}")
    print()

    all_models = build_models(args.random_state)
    state["model_order"] = list(all_models.keys())
    validation_results = []
    rows = []

    for index, (name, model) in enumerate(all_models.items(), start=1):
        key = safe_name(name)
        done = state["completed"].get(key)
        tag = f"[{index}/{len(all_models)}]"

        if done:
            metrics = done["metrics"]
            validation_pred = np.asarray(done["y_validation_pred"], dtype=object)
            print(
                f"{tag} {name}: lấy từ checkpoint "
                f"(validation macro-F1={metrics['validation_f1_macro']:.4f})"
            )
        else:
            print(f"{tag} Đang huấn luyện {name} ...", end=" ", flush=True)
            metrics, validation_pred = evaluate_candidate(
                model, X_train, X_validation, y_train, y_validation
            )
            print(
                f"train macro-F1={metrics['train_f1_macro']:.4f} | "
                f"validation macro-F1={metrics['validation_f1_macro']:.4f}"
            )

            if not args.no_save:
                report = classification_report(
                    y_validation,
                    validation_pred,
                    labels=labels,
                    digits=4,
                    zero_division=0,
                )
                (REPORTS_DIR / f"classification_report_{key}.txt").write_text(
                    f"Mô hình: {name}\nTập đánh giá: validation\n\n{report}",
                    encoding="utf-8",
                )
                save_confusion_matrix(
                    name, y_validation, validation_pred, labels
                )
                state["completed"][key] = {
                    "display_name": name,
                    "metrics": metrics,
                    "y_validation_pred": [str(value) for value in validation_pred],
                }
                save_checkpoint(state)

        rows.append({"model": name, **metrics})
        validation_results.append((name, validation_pred, y_validation))

        if not args.no_save:
            results_frame = pd.DataFrame(rows)
            write_summary(results_frame)
            save_combined_confusion(validation_results, labels)
            save_metrics_chart(results_frame)

            if args.push:
                commit_and_push(
                    f"train: {tag} {name} — validation macro-F1="
                    f"{metrics['validation_f1_macro']:.4f}",
                    paths=["reports", "checkpoints"],
                    push=True,
                )

    results_frame = pd.DataFrame(rows).sort_values(
        "validation_f1_macro", ascending=False
    )
    best_row = results_frame.iloc[0]
    best_name = best_row["model"]
    best_validation_metrics = {
        key: value
        for key, value in best_row.items()
        if key != "model" and not key.startswith("test_")
    }
    print(f"\nMô hình chọn theo validation macro-F1: {best_name}")
    print(
        f"Train macro-F1={best_validation_metrics['train_f1_macro']:.4f} | "
        f"Validation macro-F1={best_validation_metrics['validation_f1_macro']:.4f} | "
        f"Gap={best_validation_metrics['f1_macro_gap']:.4f}"
    )

    final_pipeline, test_pred, test_metrics = evaluate_final(
        build_models(args.random_state)[best_name],
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )
    print(
        f"Test macro-F1={test_metrics['test_f1_macro']:.4f} | "
        f"test balanced accuracy={test_metrics['test_balanced_accuracy']:.4f}"
    )
    for metric_name, value in test_metrics.items():
        results_frame.loc[results_frame["model"] == best_name, metric_name] = value

    if args.no_save:
        return

    key = safe_name(best_name)
    test_report = classification_report(
        y_test, test_pred, labels=labels, digits=4, zero_division=0
    )
    (REPORTS_DIR / f"classification_report_{key}_test.txt").write_text(
        f"Mô hình: {best_name}\nTập đánh giá: test (chỉ dùng sau khi chọn mô hình)\n\n"
        f"{test_report}",
        encoding="utf-8",
    )
    save_confusion_matrix(best_name, y_test, test_pred, labels, suffix="_test")
    write_summary(results_frame)
    save_metrics_chart(results_frame)

    joblib.dump(
        {
            "model": final_pipeline,
            "model_name": best_name,
            "validation_metrics": best_validation_metrics,
            "test_metrics": test_metrics,
            "labels": labels,
            "features": dataset.feature_names,
        },
        MODELS_DIR / "best_model.joblib",
    )
    state["selected_model"] = best_name
    state["test_metrics"] = test_metrics
    state["y_test_pred"] = [str(value) for value in test_pred]
    save_checkpoint(state)

    if args.push:
        commit_and_push(
            f"train: final {best_name} — test macro-F1="
            f"{test_metrics['test_f1_macro']:.4f}",
            paths=["reports", "checkpoints", "models"],
            push=True,
        )

    print("\nXếp hạng ứng viên theo validation macro-F1:")
    print(
        results_frame[
            ["model", "train_f1_macro", "validation_f1_macro", "f1_macro_gap"]
        ].to_string(index=False)
    )
    print(f"\nModel, báo cáo và checkpoint đã lưu trong {MODELS_DIR}, {REPORTS_DIR}.")


if __name__ == "__main__":
    main()
