import os
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


DATA_PATH = "winequality_preprocessing.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["quality_label"])
    y = df["quality_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "class_weight": "balanced"
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    os.makedirs("artifacts", exist_ok=True)

    model_path = "artifacts/ci_random_forest_model.pkl"
    metrics_path = "artifacts/ci_metrics.json"

    joblib.dump(model, model_path)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    mlflow.log_artifact(model_path)
    mlflow.log_artifact(metrics_path)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    mlflow.sklearn.save_model(
        sk_model=model,
        path="model"
    )

    print("CI training selesai.")
    print(metrics)
    print("Model berhasil disimpan ke folder model.")


if __name__ == "__main__":
    main()
