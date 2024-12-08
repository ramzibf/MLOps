FROM python:3.9-slim

# Installer des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Définir le répertoire de travail
WORKDIR /workspace/car_price_prediction
