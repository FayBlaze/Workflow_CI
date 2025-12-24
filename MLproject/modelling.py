from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# Load data
X_train = pd.read_csv("membangun_model/titanic_preprocessing/X_train.csv")
X_test = pd.read_csv("membangun_model/titanic_preprocessing/X_test.csv")
y_train = pd.read_csv("membangun_model/titanic_preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("membangun_model/titanic_preprocessing/y_test.csv").values.ravel()

# MLflow setup
mlflow.set_experiment("Titanic-MLflow-Basic")
mlflow.sklearn.autolog(log_models=True)

# Start run
with mlflow.start_run(run_name="logreg_baseline"):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)

    mlflow.log_metric("val_accuracy", acc)
    mlflow.log_metric("val_f1", f1)
    mlflow.log_metric("val_auc", auc)
