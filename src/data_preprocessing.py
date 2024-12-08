import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess_data(input_path, output_path):
    data = pd.read_csv(input_path)
    # Prétraitement basique
    data["make_model"] = data["make"] + "_" + data["model"]
    data = data.drop(columns=["make", "model"])
    
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    
    os.makedirs(output_path, exist_ok=True)
    train.to_csv(f"{output_path}/train.csv", index=False)
    test.to_csv(f"{output_path}/test.csv", index=False)

if __name__ == "__main__":
    preprocess_data("data/raw/cars.csv", "data/processed")
