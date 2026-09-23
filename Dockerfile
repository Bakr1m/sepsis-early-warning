FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (LightGBM needs libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Serving deps first for better layer caching
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# App code + production model (no data/, notebooks/, tests/)
COPY src/ ./src/
COPY api/ ./api/
COPY models/lgbm_windows.joblib ./models/lgbm_windows.joblib

EXPOSE 8000

CMD ["python", "api/main.py"]
