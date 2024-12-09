import pandas as pd

def preprocess_data(input_path, output_path):
    # Load data
    data = pd.read_csv(input_path)

    # Handle missing values
    data.fillna(method='ffill', inplace=True)

    # One-hot encode categorical variables
    data = pd.get_dummies(data, columns=['fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)

    # Save processed data
    data.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_data("data/raw_data.csv", "data/processed_data.csv")
