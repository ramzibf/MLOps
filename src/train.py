import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import yaml
from models import get_models

# Load configuration
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

# Load data
data = pd.read_csv("data/processed_data.csv")
X = data.drop(columns=["selling_price"])
y = data["selling_price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["data"]["test_size"], random_state=params["data"]["random_state"]
)

# Load models
models = get_models()

# MLflow experiment
mlflow.set_experiment("Car Price Prediction")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)

        # Log model
        mlflow.sklearn.log_model(model, f"models/{name}")
        print(f"Model {name} logged with R2: {r2}")
