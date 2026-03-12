"""
Session 04 – Step 4: Evaluation
Loads model from MLflow run, evaluates on test set, and logs metrics.
"""

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score

def evaluate(test_data, run_id):
    # Kolom terakhir adalah target
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy", acc)

    print(f"✅ Evaluasi Selesai. Akurasi Model Titanic: {acc:.4f}")
    return acc
