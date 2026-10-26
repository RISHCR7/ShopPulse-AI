"""Review text baseline: sentiment label from review_score and TF-IDF features."""
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

RAW = Path("data/raw/olist_order_reviews_dataset.csv")
OUT = Path("models/review_sentiment.joblib")

def train():
    if not RAW.exists(): raise FileNotFoundError("Place olist_order_reviews_dataset.csv in data/raw/.")
    df = pd.read_csv(RAW).dropna(subset=["review_comment_message", "review_score"])
    df = df[df.review_comment_message.str.strip().ne("")].copy()
    df["sentiment"] = (df.review_score >= 4).astype(int)  # 1=positive, 0=neutral/negative
    X_train, X_test, y_train, y_test = train_test_split(df.review_comment_message, df.sentiment, test_size=.2, random_state=42, stratify=df.sentiment)
    pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1,2), min_df=2, strip_accents="unicode")), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    pipe.fit(X_train, y_train); pred = pipe.predict(X_test)
    print(classification_report(y_test, pred, zero_division=0)); OUT.parent.mkdir(exist_ok=True); joblib.dump(pipe, OUT)
    return pipe

if __name__ == "__main__": train()
