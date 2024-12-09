# Use Python as base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variable for MLflow
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Default command
CMD ["bash"]
