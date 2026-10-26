
"""
ShopPulse AI
Classical Machine Learning Training Pipeline

Target:
    late_delivery

Models:
    Logistic Regression
    Gaussian Naive Bayes
    Random Forest
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

    return {

        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]),

        "GaussianNB": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianNB()),
        ]),

        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),

    }


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {

        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),

        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_test,
                probabilities,
            )
        ),

    }

    return predictions, probabilities, metrics


# --------------------------------------------------
# PLOT CONFUSION MATRIX
# --------------------------------------------------

def save_confusion_matrix(name, y_test, predictions):

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["On Time", "Late"],
    )

    display.plot()

    plt.title(f"{name} - Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / f"{name.lower()}_confusion_matrix.png",
        dpi=300,
    )

    plt.close()


# --------------------------------------------------
# PLOT ROC CURVE
# --------------------------------------------------

def save_roc_curve(name, model, X_test, y_test):

    display = RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
    )

    plt.title(f"{name} - ROC Curve")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / f"{name.lower()}_roc_curve.png",
        dpi=300,
    )

    plt.close()


# --------------------------------------------------
# MODEL COMPARISON CHART
# --------------------------------------------------

def save_comparison_chart(results):

    metrics_df = pd.DataFrame(results).T

    metrics_df[
        ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ].plot(
        kind="bar",
        figsize=(12, 6),
    )

    plt.title("ShopPulse AI - Model Performance Comparison")

    plt.ylabel("Score")

    plt.xlabel("Machine Learning Model")

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "model_comparison.png",
        dpi=300,
    )

    plt.close()


# --------------------------------------------------
# MAIN TRAINING FUNCTION
# --------------------------------------------------

def train():

    print("=" * 60)

    print("ShopPulse AI - Machine Learning Training")

    print("=" * 60)


    # Create directories

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # Load dataset

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run python src/prepare_data.py first."
        )


    df = pd.read_csv(DATA_PATH)


    # Validate required columns

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


    # Remove rows with missing model features or target

    df = df.dropna(
        subset=FEATURES + [TARGET]
    ).copy()


    # Validate target

    if df[TARGET].nunique() != 2:

        raise ValueError(
            "Expected a binary target with two classes."
        )


    # Ensure target is numeric

    df[TARGET] = df[TARGET].astype(int)


    # Dataset summary

    print("\nDataset shape:", df.shape)

    print("\nTarget distribution:")

    print(df[TARGET].value_counts())

    print("\nTarget percentages:")

    print(
        df[TARGET]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )


    # Features and target

    X = df[FEATURES]

    y = df[TARGET]


    # Stratified train-test split

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y,

    )


    print("\nTraining samples:", len(X_train))

    print("Testing samples:", len(X_test))


    # Initialize models

    models = get_models()


    results = {}

    trained_models = {}


    # Train and evaluate

    for name, model in models.items():

        print("\n" + "=" * 60)

        print("Training:", name)

        print("=" * 60)


        model.fit(
            X_train,
            y_train,
        )


        predictions, probabilities, metrics = evaluate_model(

            model,

            X_test,

            y_test,

        )


        results[name] = metrics

        trained_models[name] = model


        # Save model

        model_path = MODEL_DIR / f"{name.lower()}.joblib"

        joblib.dump(
            model,
            model_path,
        )


        # Save classification report

        report = classification_report(

            y_test,

            predictions,

            target_names=[
                "On Time",
                "Late",
            ],

            zero_division=0,

            output_dict=True,

        )


        with open(

            MODEL_DIR / f"{name.lower()}_report.json",

            "w",

            encoding="utf-8",

        ) as file:

            json.dump(
                report,
                file,
                indent=4,
            )


        # Save evaluation plots

        save_confusion_matrix(

            name,

            y_test,

            predictions,

        )


        save_roc_curve(

            name,

            model,

            X_test,

            y_test,

        )


        print("\nEvaluation metrics:")

        print(
            json.dumps(
                metrics,
                indent=4,
            )
        )


        print("\nClassification report:")

        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0,
            )
        )


    # Save metrics

    metrics_path = MODEL_DIR / "ml_metrics.json"

    with open(

        metrics_path,

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )


    # Save comparison chart

    save_comparison_chart(
        results,
    )


    # Select best model using F1 score

    best_model_name = max(

        results,

        key=lambda name: results[name]["f1"],

    )


    best_model = trained_models[
        best_model_name
    ]


    # Save best model

    joblib.dump(

        best_model,

        MODEL_DIR / "best_model.joblib",

    )


    # Save metadata

    metadata = {

        "best_model": best_model_name,

        "target": TARGET,

        "features": FEATURES,

        "train_samples": len(X_train),

        "test_samples": len(X_test),

        "random_state": 42,

        "selection_metric": "f1",

    }


    with open(

        MODEL_DIR / "model_metadata.json",

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )


    print("\n" + "=" * 60)

    print("TRAINING COMPLETED")

    print("=" * 60)


    print("\nBest model:", best_model_name)

    print("\nBest model metrics:")

    print(
        json.dumps(
            results[best_model_name],
            indent=4,
        )
    )


    print("\nSaved files:")

    print("Models:", MODEL_DIR)

    print("Metrics:", metrics_path)

    print("Figures:", FIGURE_DIR)


    return results


# --------------------------------------------------
# EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    train()
