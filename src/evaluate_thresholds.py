
"""
ShopPulse AI
Evaluate classification thresholds for late-delivery prediction.
"""

from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "orders_model.csv"

MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

OUTPUT_PATH = BASE_DIR / "models" / "threshold_metrics.json"


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


def main():

    print("=" * 60)

    print("ShopPulse AI - Threshold Evaluation")

    print("=" * 60)


    # Load dataset

    df = pd.read_csv(DATA_PATH)

    df = df.dropna(
        subset=FEATURES + [TARGET]
    )


    X = df[FEATURES]

    y = df[TARGET].astype(int)


    # Reproduce the original test split

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y,

    )


    # Load trained model

    model = joblib.load(MODEL_PATH)


    # Obtain predicted probabilities

    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    print("\nActual test distribution:")

    print(y_test.value_counts())


    print("\nROC-AUC:")

    print(
        round(
            roc_auc_score(
                y_test,
                probabilities,
            ),
            4,
        )
    )


    results = []


    # Test different thresholds

    thresholds = [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
    ]


    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)


        metrics = {

            "threshold": threshold,

            "accuracy": float(
                accuracy_score(
                    y_test,
                    predictions,
                )
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

            "predicted_late_orders": int(
                predictions.sum()
            ),

        }


        results.append(metrics)


        print("\nThreshold:", threshold)

        print(
            json.dumps(
                metrics,
                indent=2,
            )
        )


    # Save results

    OUTPUT_PATH.write_text(

        json.dumps(
            results,
            indent=4,
        ),

        encoding="utf-8",

    )


    print("\nSaved results:")

    print(OUTPUT_PATH)


if __name__ == "__main__":

    main()
