import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "titanic_preprocessing")  # folder ada di MLproject/

def load_data():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

def main():
    mlflow.set_tracking_uri("file://" + os.path.join(BASE_DIR, "mlruns"))
    mlflow.set_experiment("Titanic-MLflow-Basic")
    mlflow.sklearn.autolog(log_models=True)

    X_train, X_valid, y_train, y_valid = load_data()

    with mlflow.start_run(run_name="logreg_baseline"):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict(X_valid)
        proba = model.predict_proba(X_valid)[:, 1]

        acc = accuracy_score(y_train, preds)
        f1 = f1_score(y_train, preds)
        auc = roc_auc_score(y_train, proba)

        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("val_f1", f1)
        mlflow.log_metric("val_auc", auc)

if __name__ == "__main__":
    main()
