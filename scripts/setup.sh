#!/bin/bash

echo "Setting up the environment..."
pip install --no-cache-dir -r requirements.txt
dvc init
dvc add data/raw_data.csv
git add data/.gitignore data/raw_data.csv.dvc
git commit -m "Add raw data to DVC"
