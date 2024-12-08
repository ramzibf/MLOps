import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import pickle
import os

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("Car Price Prediction")

def train_model(model_name, train_path, model_path):
    data = pd.read_csv(train_path)
    X = data.drop(columns=["price"])
    y = data["price"]
    
    models = {
        "random_forest": RandomForestRegressor(),
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(),
    }
    
    model = models[model_name]
    model.fit(X, y)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    
    # Logging dans MLflow
    with mlflow.start_run():
        mlflow.log_param("model", model_name)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, f"models/{model_name}")
    
    # Sauvegarder localement
    os.makedirs(model_path, exist_ok=True)
    with open(f"{model_path}/{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train_model("random_forest", "data/processed/train.csv", "models")
