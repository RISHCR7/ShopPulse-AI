
"""
ShopPulse AI
Model probability and feature diagnostic.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "orders_model.csv"

MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"


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

    df = pd.read_csv(DATA_PATH)

    df = df.dropna(
        subset=FEATURES + [TARGET]
    )

    X = df[FEATURES]

    y = df[TARGET].astype(int)


    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y,

    )


    model = joblib.load(MODEL_PATH)


    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    predictions = model.predict(
        X_test
    )


    diagnostic = pd.DataFrame({

        "actual": y_test.to_numpy(),

        "probability": probabilities,

        "prediction": predictions,

    })


    print("\nMODEL INFORMATION")

    print("=" * 50)

    print("Model:", type(model).__name__)

    print("Test rows:", len(X_test))

    print("Actual late rate:", round(y_test.mean(), 4))

    print("ROC-AUC:", round(
        roc_auc_score(y_test, probabilities), 4
    ))


    print("\nPROBABILITY DISTRIBUTION")

    print("=" * 50)

    print(
        diagnostic["probability"].describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )


    print("\nPROBABILITY BY ACTUAL DELIVERY STATUS")

    print("=" * 50)

    print(

        diagnostic.groupby("actual")["probability"]

        .agg([
            "count",
            "mean",
            "median",
            "min",
            "max",
        ])

        .round(4)

    )


    print("\nPREDICTION DISTRIBUTION")

    print("=" * 50)

    print(
        diagnostic["prediction"].value_counts()
    )


    print("\nFEATURE SUMMARY")

    print("=" * 50)

    print(
        df[FEATURES].describe().round(2)
    )


if __name__ == "__main__":

    main()
