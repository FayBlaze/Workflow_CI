import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "titanic_preprocessing")

def load_data():
    X = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

def main():
    # JANGAN set_tracking_uri di MLflow Project (biar ikut tracking milik Projects)
    mlflow.set_experiment("Titanic-MLflow-Basic")
    mlflow.sklearn.autolog(log_models=True)

    X_train, X_valid, y_train, y_valid = load_data()

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_valid)
    proba = model.predict_proba(X_valid)[:, 1]

    mlflow.log_metric("val_accuracy", accuracy_score(y_valid, preds))
    mlflow.log_metric("val_f1", f1_score(y_valid, preds))
    mlflow.log_metric("val_auc", roc_auc_score(y_valid, proba))

if __name__ == "__main__":
    main()
