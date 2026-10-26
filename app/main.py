from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

from pathlib import Path
import json

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / "models"

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

app = FastAPI(
    title="ShopPulse-AI",
    description="Late delivery prediction and review sentiment API",
    version="1.0.0",
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE / "app" / "static")),
    name="static"
)

templates = Jinja2Templates(
    directory=str(BASE / "app" / "templates")
)


@app.get("/ui", response_class=HTMLResponse)
def user_interface(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

class DeliveryRequest(BaseModel):
    item_count: float = Field(ge=0)
    order_value: float = Field(ge=0)
    freight_value: float = Field(ge=0)
    payment_total: float = Field(ge=0)
    payment_installments_max: float = Field(ge=0)
    estimated_delivery_days: float
    purchase_month: int = Field(ge=1, le=12)
    purchase_weekday: int = Field(ge=0, le=6)
    purchase_hour: int = Field(ge=0, le=23)


def load_json(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


# Load trained ML model and its threshold metadata.
ml_model = None
ml_threshold = 0.20

ml_path = MODELS / "best_model.joblib"
if ml_path.exists():
    ml_model = joblib.load(ml_path)

ml_metadata = load_json(MODELS / "metadata.json")
ml_threshold = float(ml_metadata.get("threshold", ml_threshold))

# Load DL model and preprocessing scaler.
dl_model = None
dl_scaler = None
dl_threshold = 0.64

dl_path = MODELS / "keras_late_delivery.keras"
scaler_path = MODELS / "dl_scaler.joblib"

if dl_path.exists() and scaler_path.exists():
    dl_model = load_model(dl_path)
    dl_scaler = joblib.load(scaler_path)

dl_metadata = load_json(MODELS / "dl_metadata.json")
dl_threshold = float(dl_metadata.get("threshold", dl_threshold))

# Optional NLP pipeline.
sentiment_model = None
sentiment_path = MODELS / "review_sentiment.joblib"

if sentiment_path.exists():
    sentiment_model = joblib.load(sentiment_path)


@app.get("/")
def home():
    return {
        "project": "ShopPulse-AI",
        "message": "Late delivery prediction API",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "ml_model_loaded": ml_model is not None,
        "dl_model_loaded": dl_model is not None,
        "sentiment_model_loaded": sentiment_model is not None,
    }


def input_array(request):
    values = request.model_dump()
    return np.array([[values[f] for f in FEATURES]], dtype=np.float32)


@app.post("/predict")
def predict_ml(request: DeliveryRequest):
    if ml_model is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not found. Train ML model first."
        )

    X = input_array(request)

    try:
        probability = float(ml_model.predict_proba(X)[0][1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    prediction = int(probability >= ml_threshold)

    return {
        "model": "Random Forest / trained ML model",
        "late_delivery_prediction": prediction,
        "label": "Late" if prediction else "On time",
        "late_probability": round(probability, 4),
        "threshold": ml_threshold,
    }


@app.post("/predict/dl")
def predict_dl(request: DeliveryRequest):
    if dl_model is None or dl_scaler is None:
        raise HTTPException(
            status_code=503,
            detail="DL model or scaler not found. Train DL first."
        )

    X = input_array(request)
    X_scaled = dl_scaler.transform(X).astype(np.float32)

    probability = float(dl_model.predict(X_scaled, verbose=0)[0][0])
    prediction = int(probability >= dl_threshold)

    return {
        "model": "TensorFlow Keras Neural Network",
        "late_delivery_prediction": prediction,
        "label": "Late" if prediction else "On time",
        "late_probability": round(probability, 4),
        "threshold": dl_threshold,
    }


class ReviewRequest(BaseModel):
    review_text: str = Field(min_length=1, max_length=5000)


@app.post("/sentiment")
def sentiment(request: ReviewRequest):
    if sentiment_model is None:
        raise HTTPException(
            status_code=503,
            detail="Sentiment model not found. Run src/review_nlp.py first."
        )

    prediction = int(sentiment_model.predict([request.review_text])[0])

    result = {
        1: "Positive",
        0: "Neutral or negative",
    }

    response = {
        "sentiment_prediction": prediction,
        "sentiment": result.get(prediction, "Unknown"),
    }

    if hasattr(sentiment_model, "predict_proba"):
        probabilities = sentiment_model.predict_proba(
            [request.review_text]
        )[0]
        response["class_probabilities"] = {
            str(label): round(float(prob), 4)
            for label, prob in zip(
                sentiment_model.classes_, probabilities
            )
        }

    return response
