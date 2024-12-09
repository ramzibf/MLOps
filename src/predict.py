from flask import Flask, request, jsonify
import mlflow.sklearn

app = Flask(__name__)

# Load model
model = mlflow.sklearn.load_model("models:/RandomForest/production")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    predictions = model.predict([data])
    return jsonify({"predictions": predictions.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
