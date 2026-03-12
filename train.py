"""
Session 04 – Step 3: Training
Trains a Random Forest classifier and logs to MLflow.
"""
import mlflow
import mlflow.sklearn
import joblib
from sklearn.linear_model import LogisticRegression

def train(train_data):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Spaceship_Titanic_LR")

    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    with mlflow.start_run() as run:
        # Menggunakan Logistic Regression sesuai instruksi
        model = LogisticRegression(C=0.1, max_iter=1000)
        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(model, "model")
        joblib.dump(model, "artifacts/model.pkl")
        
        print(f"✅ Model Logistic Regression Berhasil Dilatih.")
        return run.info.run_id