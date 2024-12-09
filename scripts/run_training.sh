#!/bin/bash
set -e

# Preprocess data
python src/preprocess.py

# Train models
python src/train.py
