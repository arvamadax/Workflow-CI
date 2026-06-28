"""
modelling.py — MLflow Project training script untuk Workflow CI.

Script ini melakukan training Random Forest dengan hyperparameter terbaik
yang sudah ditemukan dari proses tuning di Kriteria 2 (RandomizedSearchCV).
Training dilakukan secara deterministik dan reproducible — sesuai best practice
MLOps di mana tuning dilakukan di fase research, dan production training
menggunakan hasil tuning tersebut.

Hyperparameter terbaik (hasil tuning K2):
    n_estimators      : 200
    max_depth         : 30
    min_samples_split : 2
    min_samples_leaf  : 4
    max_features      : "sqrt"
    class_weight      : "balanced"

Penggunaan via MLflow Project (recommended):
    mlflow run . --env-manager=local

Atau dengan parameter override:
    mlflow run . -P n_estimators=300 --env-manager=local

Penggunaan langsung (testing):
    python modelling.py

Author : Arva Mada Jayastu
"""

import argparse
import os
import sys
from pathlib import Path

# Set matplotlib backend SEBELUM import library lain (untuk lingkungan headless CI)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ============================================================
# Konfigurasi
# ============================================================
DATA_DIR = "namadataset_preprocessing"
TARGET_COL = "Churn"
EXPERIMENT_NAME = "telco_churn_ci"


# ============================================================
# CLI args
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Random Forest for Telco Churn")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=30)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=4)
    parser.add_argument("--max_features", type=str, default="sqrt")
    parser.add_argument("--class_weight", type=str, default="balanced")
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


# ============================================================
# Data loading
# ============================================================
def load_data(data_dir: str = DATA_DIR) -> tuple:
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            f"train.csv/test.csv tidak ditemukan di {data_dir}/"
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]

    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Test : X={X_test.shape}, y={y_test.shape}")
    return X_train, X_test, y_train, y_test


# ============================================================
# Metrics
# ============================================================
def compute_metrics(y_true, y_pred, y_proba, prefix: str = "") -> dict:
    return {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}f1_score": f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}roc_auc": roc_auc_score(y_true, y_proba),
        f"{prefix}log_loss": log_loss(y_true, y_proba),
    }


# ============================================================
# Artefak helpers
# ============================================================
def save_confusion_matrix(y_true, y_pred, path: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"], ax=ax,
    )
    ax.set_title("Confusion Matrix — Test Set", fontsize=14)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(model, feature_names, path: str, top_n: int = 20) -> None:
    importances = model.feature_importances_
    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(
        data=df_imp, x="importance", y="feature",
        hue="feature", palette="viridis", legend=False, ax=ax,
    )
    ax.set_title(f"Top {top_n} Feature Importance — Random Forest", fontsize=14)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(y_true, y_proba, path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, lw=2, label=f"Random Forest (AUC = {auc:.3f})", color="#2980b9")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.2, color="#2980b9")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test Set", fontsize=14)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_classification_report(y_true, y_pred, path: str) -> None:
    report = classification_report(
        y_true, y_pred, target_names=["No Churn", "Churn"], digits=4,
    )
    with open(path, "w") as f:
        f.write("Classification Report — Telco Customer Churn\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write("\n")


# ============================================================
# Main training pipeline
# ============================================================
def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("MLflow Project: Telco Churn RF (CI)")
    print("=" * 60)
    print(f"\nHyperparameters:")
    for k, v in vars(args).items():
        print(f"  {k:20s} = {v}")

    # Setup tracking (skip set_experiment kalau MLFLOW_EXPERIMENT_ID sudah ada
    # dari mlflow run command)
    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"\nExperiment   : {EXPERIMENT_NAME}")
    print(f"Tracking URI : {mlflow.get_tracking_uri()}")

    # Load data
    print("\n[1/4] Load data...")
    X_train, X_test, y_train, y_test = load_data()
    feature_names = X_train.columns.tolist()

    # Train model (DETERMINISTIK — pakai best params dari K2)
    print("\n[2/4] Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        class_weight=args.class_weight,
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluasi
    print("\n[3/4] Evaluasi...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_test = model.predict_proba(X_test)[:, 1]

    train_metrics = compute_metrics(y_train, y_pred_train, y_proba_train, "train_")
    test_metrics = compute_metrics(y_test, y_pred_test, y_proba_test, "test_")

    print(f"  Train accuracy : {train_metrics['train_accuracy']:.4f}")
    print(f"  Test accuracy  : {test_metrics['test_accuracy']:.4f}")
    print(f"  Test F1        : {test_metrics['test_f1_score']:.4f}")
    print(f"  Test ROC-AUC   : {test_metrics['test_roc_auc']:.4f}")

    # MLflow logging
    print("\n[4/4] Logging ke MLflow...")
    artifacts_dir = Path("artifacts_tmp")
    artifacts_dir.mkdir(exist_ok=True)

    # Log ke run yang aktif. Logika:
    # - Kalau dijalankan via `mlflow run`, MLFLOW_RUN_ID sudah di-set env var,
    #   dan log_param/log_metric otomatis pakai run itu — TIDAK BOLEH start_run lagi.
    # - Kalau dijalankan langsung (python modelling.py), kita start run manual.
    started_run_manually = False
    if os.environ.get("MLFLOW_RUN_ID"):
        # Via mlflow run — operasi logging akan otomatis pakai run yang sudah ada
        pass
    elif mlflow.active_run() is None:
        mlflow.start_run(run_name="rf_ci_training")
        started_run_manually = True

    # Log params
    for k, v in vars(args).items():
        mlflow.log_param(k, v)

    # Log metrics
    for name, value in {**train_metrics, **test_metrics}.items():
        mlflow.log_metric(name, value)

    # Log model
    signature = infer_signature(X_train, y_pred_train)
    input_example = X_train.head(5)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
    )
    print("  > Model artifact logged")

    # Extra artifacts (5 extras di luar autolog standard)
    cm_path = artifacts_dir / "confusion_matrix.png"
    save_confusion_matrix(y_test, y_pred_test, str(cm_path))
    mlflow.log_artifact(str(cm_path))
    print("  > [Extra 1] Confusion matrix logged")

    fi_path = artifacts_dir / "feature_importance.png"
    save_feature_importance(model, feature_names, str(fi_path))
    mlflow.log_artifact(str(fi_path))
    print("  > [Extra 2] Feature importance logged")

    roc_path = artifacts_dir / "roc_curve.png"
    save_roc_curve(y_test, y_proba_test, str(roc_path))
    mlflow.log_artifact(str(roc_path))
    print("  > [Extra 3] ROC curve logged")

    cr_path = artifacts_dir / "classification_report.txt"
    save_classification_report(y_test, y_pred_test, str(cr_path))
    mlflow.log_artifact(str(cr_path))
    print("  > [Extra 4] Classification report logged")

    # Simpan info ringkas untuk CI: run_id ke file
    # Ambil run_id baik dari env var (mlflow run) atau dari active_run (standalone)
    run_id = os.environ.get("MLFLOW_RUN_ID") or mlflow.active_run().info.run_id
    with open("run_info.txt", "w") as f:
        f.write(f"{run_id}\n")
    mlflow.log_artifact("run_info.txt")
    print("  > [Extra 5] Run info logged")

    # Tutup run kalau kita yang start
    if started_run_manually:
        mlflow.end_run()

    # Cleanup
    import shutil
    shutil.rmtree(artifacts_dir, ignore_errors=True)
    if os.path.exists("run_info.txt"):
        # Pertahankan run_info.txt di working dir untuk akses CI
        pass

    print("\n" + "=" * 60)
    print(f"Training selesai. run_id: {run_id}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
