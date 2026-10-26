# Implementation and report plan

## Dataset validation
- Download Olist files from the linked Kaggle source.
- Record source, retrieval date, licenses, row counts, columns, missingness and duplicates.
- Use order-level late delivery as the initial target: delivered timestamp > estimated timestamp.
- Exclude post-delivery fields (actual delivery duration, review score) from prediction features to prevent leakage.

## Required experiments
- ML: Logistic Regression, Gaussian Naive Bayes, Random Forest; stratified 80/20 split and fixed seed.
- Metrics: accuracy, precision, recall, F1, ROC-AUC; discuss class imbalance and threshold limitations.
- DL: Keras dense neural network baseline on standardized tabular features.
- NLP: TF-IDF + Logistic Regression on non-empty review comments; review score >=4 is positive, lower scores grouped as neutral/negative (explicitly documented limitation).
- GenAI: Ollama local model summarizing only computed metrics, with no fabricated values.
- Deployment: FastAPI `/health` and `/predict` endpoints.

## Report deliverables
Cover page; TOC; problem statement and dataset; preparation; EDA/visualization; ML; DL; NLP/GenAI; deployment; results; conclusion; references. Populate charts, metric tables and screenshots only after actual runs.
