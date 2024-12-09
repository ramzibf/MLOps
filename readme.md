# Car Price Prediction MLOps Project

This project uses MLOps tools and techniques to train machine learning models for car price prediction. It includes data versioning (DVC), model tracking (MLflow), and a deployment pipeline.

## Features
- Continuous Training Pipeline
- MLflow for Experiment Tracking
- Random Forest, Decision Tree, Linear Regression Models
- Docker and Docker Compose for Containerization

## Steps to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Start services: `docker-compose up`
3. Train models: `bash scripts/run_training.sh`
4. Serve predictions: `bash scripts/run_server.sh`
