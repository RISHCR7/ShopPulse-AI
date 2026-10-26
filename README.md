# ShopPulse-AI — E-Commerce Delivery Prediction & Review Sentiment Analysis

An end-to-end MCA Data Science project that predicts e-commerce late-delivery risk and analyzes customer review sentiment using the Olist Brazilian E-Commerce dataset.

## Project Overview

ShopPulse-AI combines data preprocessing, exploratory analysis, feature engineering, machine learning, deep learning, NLP, and a FastAPI web application to support e-commerce delivery analysis and customer feedback understanding.

## Features

- Processed and prepared an order-level dataset containing 96,469 rows and 13 columns.
- Engineered nine features related to order details, payment, estimated delivery, and purchase timing.
- Developed a Random Forest model for late-delivery prediction.
- Built a TensorFlow/Keras deep-learning model for delivery classification.
- Implemented customer review sentiment analysis using TF-IDF and Logistic Regression.
- Created a FastAPI backend with prediction and sentiment endpoints.
- Developed a browser-based frontend for interacting with the application.

## Technologies Used

Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, TF-IDF, Logistic Regression, FastAPI, Uvicorn, HTML, CSS, JavaScript.

## Dataset

This project uses the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

The dataset contains e-commerce order, payment, item, and review information.

The original dataset is not included in this repository. Download it from Kaggle and place the required CSV files in `data/raw/`.

## Project Structure

```text
ShopPulse-AI/
├── app/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── src/
│   ├── prepare_data.py
│   ├── train_ml.py
│   ├── train_dl.py
│   ├── review_nlp.py
│   └── evaluate_thresholds.py
├── app/
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ShopPulse-AI
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the FastAPI application from the project root:

```bash
uvicorn app.main:app --reload
```

Open the local frontend:

http://127.0.0.1:8000/ui

API documentation:

http://127.0.0.1:8000/docs

The application requires the trained model and sentiment artifacts to be available in the configured locations.

## Model Evaluation

The project evaluates delivery prediction models using accuracy, precision, recall, F1-score, ROC-AUC, and average precision.

The delivery prediction task is imbalanced, so model thresholds and precision-recall trade-offs are considered during evaluation.

## Limitations

ShopPulse-AI is an educational project. Predictions are estimates and may not generalize to future orders. Model performance can be affected by class imbalance, dataset limitations, and changes in delivery patterns.

## Student

Rudraksh Dosi  
MCA Data Science

## License

The code is intended for educational use. The Olist dataset remains subject to its own terms and conditions.