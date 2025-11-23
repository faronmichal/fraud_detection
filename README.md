# Credit Card Fraud Detection System

## Project Overview
This repository provides an end-to-end Machine Learning system for detecting fraudulent credit card transactions.

The project addresses the challenge of extreme class imbalance (0.17% fraud rate) and demonstrates a production-ready MLOps pipeline that includes training, serialization, containerization, and serving via an API for real-time inference.

---

## Key Technical Features

### 1. Advanced Preprocessing & Feature Engineering
- Cyclic time encoding (`Hour_Sin`, `Hour_Cos`) to preserve temporal patterns.
- Robust scaling to reduce the effect of outliers.
- Shared preprocessing module (`src/preprocessing.py`) to ensure consistency between training and inference environments.

### 2. Model Optimization (AutoML)
- Model comparison: XGBoost, LightGBM, CatBoost.
- Hyperparameter tuning using Optuna (Bayesian Optimization).
- Final model selection: XGBoost based on AUC-PR and recall performance.

### 3. Production Architecture (MLOps)
- REST API built with FastAPI and Pydantic.
- Fully containerized using a Python 3.12-slim Docker image.
- Serialized artifacts (model, scaler, threshold) loaded efficiently on startup via `joblib`.

---

## Results & Metrics

| Metric     | Value   | Description |
|------------|---------|-------------|
| AUC-PR     | 0.8075  | Primary metric for imbalanced data |
| Precision  | High    | Minimized false positives |
| Recall     | Robust  | Increased fraud detection rate |
| Threshold  | Dynamic | Determined automatically through cross-validation |

---

## Repository Structure

```
fraud-detection-project/
├── notebooks/             # Exploratory analysis & Optuna experiments
├── src/
│   ├── app.py             # FastAPI entry point
│   ├── train.py           # Training pipeline
│   └── preprocessing.py   # Shared preprocessing logic
├── models/                # Serialized model artifacts
├── data/                  # Dataset directory
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## Installation & Usage

### Method 1: Docker (Recommended)

Build the Docker image:
```bash
docker build -t fraud-detector:v1 .
```

Run the container:
```bash
docker run -p 8000:8000 fraud-detector:v1
```

API documentation (Swagger UI):
```
http://localhost:8000/docs
```

---

### Method 2: Local Environment

Install dependencies:
```bash
pip install -r requirements.txt
```

Train the model:
```bash
python src/train.py
```

Start the API server:
```bash
python -m uvicorn src.app:app --reload
```

---

## Author
Michał Faron  