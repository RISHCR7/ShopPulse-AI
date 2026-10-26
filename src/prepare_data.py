
"""Build an order-level modeling table from Olist CSV files.

Run from the repository root after placing the official CSVs in data/raw.
"""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW = BASE_DIR / "data" / "raw"
OUT = BASE_DIR / "data" / "processed" / "orders_model.csv"


def load_csv(name: str) -> pd.DataFrame:
    path = RAW / name

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download the Olist dataset and place CSVs in data/raw/."
        )

    return pd.read_csv(path)


def build_order_table() -> pd.DataFrame:
    orders = load_csv("olist_orders_dataset.csv")
    items = load_csv("olist_order_items_dataset.csv")
    payments = load_csv("olist_order_payments_dataset.csv")
    reviews = load_csv("olist_order_reviews_dataset.csv")

    timestamp_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for col in timestamp_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    item_features = (
        items.groupby("order_id")
        .agg(
            item_count=("order_item_id", "count"),
            order_value=("price", "sum"),
            freight_value=("freight_value", "sum"),
        )
        .reset_index()
    )

    payment_features = (
        payments.groupby("order_id")
        .agg(
            payment_total=("payment_value", "sum"),
            payment_installments_max=("payment_installments", "max"),
        )
        .reset_index()
    )

    review_features = (
        reviews.groupby("order_id")
        .agg(
            review_score=("review_score", "mean"),
            review_count=("review_id", "count"),
        )
        .reset_index()
    )

    df = orders.merge(item_features, on="order_id", how="left")
    df = df.merge(payment_features, on="order_id", how="left")
    df = df.merge(review_features, on="order_id", how="left")

    # Keep orders with completed delivery status.
    df = df[df["order_status"].eq("delivered")].copy()

    # Target requires both actual and estimated delivery timestamps.
    target_timestamp_cols = [
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    missing_target_timestamps = df[target_timestamp_cols].isna().any(axis=1)
    removed_missing_target = int(missing_target_timestamps.sum())

    df = df.loc[~missing_target_timestamps].copy()

    # Actual elapsed delivery duration.
    df["actual_delivery_days"] = (
        df["order_delivered_customer_date"]
        - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    # Estimated delivery duration available at purchase time.
    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"]
        - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    # Binary target: delivered later than estimated.
    df["late_delivery"] = (
        df["order_delivered_customer_date"]
        > df["order_estimated_delivery_date"]
    ).astype("int8")

    # Purchase-time features.
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour

    feature_cols = [
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

    target = "late_delivery"

    result = df[
        [
            "order_id",
            *feature_cols,
            target,
            "actual_delivery_days",
            "review_score",
        ]
    ].copy()

    # Remove missing predictor values.
    before_feature_drop = len(result)

    result = result.dropna(subset=feature_cols + [target])

    removed_missing_features = before_feature_drop - len(result)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)

    print(f"Saved {len(result):,} rows to {OUT}")
    print(f"Removed missing target timestamps: {removed_missing_target:,}")
    print(f"Removed missing feature rows: {removed_missing_features:,}")

    print("\nTarget distribution:")
    print(
        result[target]
        .value_counts()
        .rename_axis("late_delivery")
        .to_frame("orders")
    )

    print("\nTarget proportions:")
    print(result[target].value_counts(normalize=True).round(4))

    print("\nFeature columns:")
    print(feature_cols)

    return result


if __name__ == "__main__":
    build_order_table()
