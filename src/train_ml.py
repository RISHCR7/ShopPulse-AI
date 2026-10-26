"""
ShopPulse AI - Classical Machine Learning Training Pipeline

Models:
- Logistic Regression
- Gaussian Naive Bayes
- Random Forest

Workflow:
- 60% training
- 20% validation
- 20% testing

Model and threshold selection use validation data only.
The test set is held out until final evaluation.
"""

from pathlib import Path
import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "orders_model.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

RANDOM_STATE = 42


# --------------------------------------------------
# FEATURES AND TARGET
# --------------------------------------------------

FEATURES = [
    "item_count",
    "order_value",
    "freight_value",
    "payment_total",
    "payment_installments_max",
    "estimated_delivery_days",
    "purchase_month",
    "purchase_weekday",
    "purchase_hour",
]

TARGET = "late_delivery"


# --------------------------------------------------
# MODEL DEFINITIONS
# --------------------------------------------------

def get_models():
    """Return fresh, untrained model instances."""
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "GaussianNB": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianNB()),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_metrics(y_true, predictions, probabilities):
    """Calculate threshold-dependent and threshold-independent metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, predictions, zero_division=0
        )),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
    }


# --------------------------------------------------
# THRESHOLD SELECTION
# --------------------------------------------------

def select_threshold(y_true, probabilities):
    """
    Select the threshold with the highest validation F1.
    In an F1 tie, prefer the higher threshold (fewer predicted alerts).
    """
    thresholds = [
        0.10, 0.15, 0.20, 0.25, 0.30,
        0.35, 0.40, 0.45, 0.50, 0.55,
        0.60, 0.65, 0.70, 0.75, 0.80,
    ]

    best_threshold = 0.50
    best_f1 = -1.0
    threshold_results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_true, predictions, zero_division=0
        )
        recall = recall_score(
            y_true, predictions, zero_division=0
        )
        f1 = f1_score(y_true, predictions, zero_division=0)

        threshold_results.append({
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "predicted_late_orders": int(predictions.sum()),
        })

        if (
            f1 > best_f1
            or (
                abs(f1 - best_f1) < 1e-12
                and threshold > best_threshold
            )
        ):
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, threshold_results


# --------------------------------------------------
# PLOTS
# --------------------------------------------------

def save_confusion_matrix(name, y_true, predictions):
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["On Time", "Late"],
    )
    display.plot()
    plt.title(f"{name} - Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / f"{name.lower()}_confusion_matrix.png",
        dpi=300,
    )
    plt.close()


def save_roc_curve(name, model, X_test, y_test):
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"{name} - Test ROC Curve")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / f"{name.lower()}_roc_curve.png",
        dpi=300,
    )
    plt.close()


def save_comparison_chart(results):
    """Plot validation metrics at the default 0.50 threshold."""
    metrics_df = pd.DataFrame(results).T
    metrics_df[
        ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ].plot(kind="bar", figsize=(12, 6))

    plt.title("ShopPulse AI - Validation Model Comparison")
    plt.ylabel("Score")
    plt.xlabel("Machine Learning Model")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "model_comparison.png", dpi=300)
    plt.close()


# --------------------------------------------------
# MAIN TRAINING FUNCTION
# --------------------------------------------------

def train():
    print("=" * 60)
    print("ShopPulse AI - Machine Learning Training")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset.
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run python src/prepare_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    # Validate required columns.
    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Remove rows with missing model features or target.
    df = df.dropna(subset=required_columns).copy()

    # Validate binary target.
    if df[TARGET].nunique() != 2:
        raise ValueError("Expected a binary target with two classes.")

    df[TARGET] = df[TARGET].astype(int)
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target values must be encoded as 0 and 1.")

    X = df[FEATURES]
    y = df[TARGET]

    print("\nDataset shape:", df.shape)
    print("\nTarget distribution:")
    print(y.value_counts())
    print("\nTarget percentages:")
    print(y.value_counts(normalize=True).mul(100).round(2))

    # --------------------------------------------------
    # 60% TRAIN / 40% TEMPORARY
    # --------------------------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Temporary 40% -> 20% validation / 20% test.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print("\nDataset split:")
    print("Training:", len(X_train))
    print("Validation:", len(X_val))
    print("Testing:", len(X_test))

    print("\nLate orders in each split:")
    print("Training:", int(y_train.sum()))
    print("Validation:", int(y_val.sum()))
    print("Testing:", int(y_test.sum()))

    # --------------------------------------------------
    # TRAIN AND COMPARE MODELS ON VALIDATION
    # --------------------------------------------------

    trained_models = {}
    validation_results = {}
    threshold_results_by_model = {}

    for name, model in get_models().items():
        print("\n" + "-" * 60)
        print("Training:", name)
        print("-" * 60)

        model.fit(X_train, y_train)
        validation_probabilities = model.predict_proba(X_val)[:, 1]

        # Metrics at the conventional 0.50 threshold.
        default_predictions = (
            validation_probabilities >= 0.50
        ).astype(int)
        default_metrics = calculate_metrics(
            y_val, default_predictions, validation_probabilities
        )

        # Tune threshold on validation data only.
        optimized_threshold, threshold_results = select_threshold(
            y_val, validation_probabilities
        )
        optimized_predictions = (
            validation_probabilities >= optimized_threshold
        ).astype(int)
        optimized_metrics = calculate_metrics(
            y_val, optimized_predictions, validation_probabilities
        )

        # Store default metrics at top level for comparison chart,
        # plus optimized metrics for model selection.
        validation_results[name] = {
            **default_metrics,
            "optimized_threshold": float(optimized_threshold),
            "optimized_precision": optimized_metrics["precision"],
            "optimized_recall": optimized_metrics["recall"],
            "optimized_f1": optimized_metrics["f1"],
            "optimized_average_precision": (
                optimized_metrics["average_precision"]
            ),
        }

        trained_models[name] = model
        threshold_results_by_model[name] = threshold_results

        print("\nMetrics at threshold 0.50:")
        print(json.dumps(default_metrics, indent=4))
        print("\nBest validation threshold:", optimized_threshold)
        print("\nMetrics at optimized validation threshold:")
        print(json.dumps(optimized_metrics, indent=4))

    # Save validation metrics and comparison chart.
    with open(
        MODEL_DIR / "validation_metrics.json", "w", encoding="utf-8"
    ) as file:
        json.dump(validation_results, file, indent=4)

    save_comparison_chart(validation_results)

    # Save threshold results for every model.
    all_threshold_results = []
    for model_name, rows in threshold_results_by_model.items():
        for row in rows:
            all_threshold_results.append({
                "model": model_name,
                **row,
            })

    pd.DataFrame(all_threshold_results).to_csv(
        REPORT_DIR / "threshold_validation_all_models.csv",
        index=False,
    )

    # --------------------------------------------------
    # SELECT MODEL USING OPTIMIZED VALIDATION F1
    # --------------------------------------------------
    # Average Precision is used as a tie-breaker. The test set is not
    # consulted during model or threshold selection.

    best_model_name = max(
        validation_results,
        key=lambda name: (
            validation_results[name]["optimized_f1"],
            validation_results[name]["average_precision"],
        ),
    )

    best_model = trained_models[best_model_name]
    best_threshold = validation_results[best_model_name][
        "optimized_threshold"
    ]

    print("\nSelected model:", best_model_name)
    print("Selected validation threshold:", best_threshold)
    print(
        "Selected optimized validation F1:",
        validation_results[best_model_name]["optimized_f1"],
    )

    # Save selected model's threshold report.
    selected_threshold_df = pd.DataFrame(
        threshold_results_by_model[best_model_name]
    )
    selected_threshold_df.to_csv(
        REPORT_DIR / "threshold_validation.csv",
        index=False,
    )

    with open(
        MODEL_DIR / "threshold_validation.json", "w", encoding="utf-8"
    ) as file:
        json.dump({
            "selected_model": best_model_name,
            "selected_threshold": float(best_threshold),
            "selection_metric": "optimized_validation_f1",
            "validation_results": threshold_results_by_model[
                best_model_name
            ],
        }, file, indent=4)

    # --------------------------------------------------
    # FINAL TEST EVALUATION (ONE TIME, AFTER SELECTION)
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    test_probabilities = best_model.predict_proba(X_test)[:, 1]

    # Test metrics at default threshold.
    test_predictions_default = (
        test_probabilities >= 0.50
    ).astype(int)
    default_test_metrics = calculate_metrics(
        y_test, test_predictions_default, test_probabilities
    )

    # Test metrics at validation-selected threshold.
    test_predictions_selected = (
        test_probabilities >= best_threshold
    ).astype(int)
    selected_test_metrics = calculate_metrics(
        y_test, test_predictions_selected, test_probabilities
    )

    print("\nTest metrics at threshold 0.50:")
    print(json.dumps(default_test_metrics, indent=4))
    print("\nTest metrics at validation-selected threshold:")
    print(json.dumps(selected_test_metrics, indent=4))

    print("\nTest classification report:")
    print(classification_report(
        y_test,
        test_predictions_selected,
        target_names=["On Time", "Late"],
        zero_division=0,
    ))

    test_results = {
        "selected_model": best_model_name,
        "selected_threshold": float(best_threshold),
        "default_threshold_0_50": default_test_metrics,
        "validation_selected_threshold": selected_test_metrics,
    }

    with open(
        MODEL_DIR / "final_test_metrics.json", "w", encoding="utf-8"
    ) as file:
        json.dump(test_results, file, indent=4)

    # Save final classification report.
    report = classification_report(
        y_test,
        test_predictions_selected,
        target_names=["On Time", "Late"],
        zero_division=0,
        output_dict=True,
    )
    with open(
        MODEL_DIR / "final_test_report.json", "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    # Save final test plots.
    save_confusion_matrix(
        best_model_name, y_test, test_predictions_selected
    )
    save_roc_curve(best_model_name, best_model, X_test, y_test)

    # --------------------------------------------------
    # SAVE FINAL MODEL AND METADATA
    # --------------------------------------------------

    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

    metadata = {
        "best_model": best_model_name,
        "target": TARGET,
        "features": FEATURES,
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "split": {
            "train": 0.60,
            "validation": 0.20,
            "test": 0.20,
        },
        "selection_metric": "optimized_validation_f1",
        "average_precision_used_as_tiebreaker": True,
        "selected_threshold": float(best_threshold),
        "test_evaluation": "untouched_test_set",
    }

    with open(
        MODEL_DIR / "model_metadata.json", "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print("\nBest model:", best_model_name)
    print("Selected threshold:", best_threshold)
    print("\nFinal test metrics:")
    print(json.dumps(selected_test_metrics, indent=4))
    print("\nSaved model:")
    print(MODEL_DIR / "best_model.joblib")
    print("\nSaved validation metrics:")
    print(MODEL_DIR / "validation_metrics.json")
    print("\nSaved final test metrics:")
    print(MODEL_DIR / "final_test_metrics.json")
    print("\nSaved threshold results:")
    print(REPORT_DIR / "threshold_validation.csv")
    print("\nSaved figures:")
    print(FIGURE_DIR)

    return test_results


if __name__ == "__main__":
    train()
