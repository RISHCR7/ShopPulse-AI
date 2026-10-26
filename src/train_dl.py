
"""ShopPulse-AI: Deep Learning late-delivery classifier."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

DATA = Path("data/processed/orders_model.csv")
MODEL_DIR = Path("models")
FEATURES = [
    "item_count", "order_value", "freight_value", "payment_total",
    "payment_installments_max", "estimated_delivery_days",
    "purchase_month", "purchase_weekday", "purchase_hour"
]
TARGET = "late_delivery"
SEED = 42


def best_f1_threshold(y_true, probabilities):
    best = {"threshold": 0.5, "f1": -1.0}
    for threshold in np.arange(0.05, 0.81, 0.01):
        pred = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best["f1"]:
            best = {"threshold": round(float(threshold), 2),
                    "f1": float(score)}
    return best


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(
            y_true, predictions, zero_division=0)),
        "recall": float(recall_score(
            y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(
            average_precision_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_true, predictions).tolist()
    }


def train():
    if not DATA.exists():
        raise FileNotFoundError(
            "Dataset missing. Run python src/prepare_data.py first."
        )

    tf.keras.utils.set_random_seed(SEED)
    MODEL_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA)
    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.int32)

    # 60% train, 20% validation, 20% final test.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50,
        random_state=SEED, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # Compensate for the imbalanced late-delivery classes.
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    class_weights = {
        0: 1.0,
        1: negatives / max(positives, 1)
    }

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURES),)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "models/dl_best_checkpoint.keras",
            monitor="val_auc", mode="max",
            save_best_only=True
        )
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=256,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2
    )

    # Select threshold using validation data only.
    val_prob = model.predict(X_val, verbose=0).ravel()
    selected = best_f1_threshold(y_val, val_prob)
    threshold = selected["threshold"]

    # Final evaluation on untouched test data.
    test_prob = model.predict(X_test, verbose=0).ravel()
    metrics = calculate_metrics(y_test, test_prob, threshold)
    metrics["validation_best_f1"] = selected["f1"]
    metrics["dataset_rows"] = int(len(df))
    metrics["features"] = FEATURES
    metrics["split"] = "60% train / 20% validation / 20% test"

    model.save("models/keras_late_delivery.keras")
    joblib.dump(scaler, "models/dl_scaler.joblib")

    Path("models/dl_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    Path("models/dl_metadata.json").write_text(
        json.dumps({
            "features": FEATURES,
            "threshold": threshold,
            "model_type": "TensorFlow Keras neural network"
        }, indent=2),
        encoding="utf-8"
    )

    print("\nDEEP LEARNING TEST RESULTS")
    print(json.dumps(metrics, indent=2))
    print("\nSaved model: models/keras_late_delivery.keras")
    print("Saved scaler: models/dl_scaler.joblib")
    print("Saved metrics: models/dl_metrics.json")


if __name__ == "__main__":
    train()
