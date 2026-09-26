FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (LightGBM needs libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Serving deps first for better layer caching
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# App code (no data/, notebooks/, tests/)
COPY src/ ./src/
COPY api/ ./api/

# Production model: fetched by exact release version, SHA256-verified.
# Never a moving tag, never baked from a developer laptop. Provenance:
# https://github.com/Bakr1m/sepsis-early-warning/releases/tag/v1.0.0
ARG MODEL_TAG=v1.0.0
ARG MODEL_SHA256=03aedad33bcc3fc817c380b276d0cf50c907b6259a990f3a308c1b63585be5ad
RUN mkdir -p models && \
    curl -fsSL -o models/lgbm_windows.joblib \
      "https://github.com/Bakr1m/sepsis-early-warning/releases/download/${MODEL_TAG}/lgbm_windows.joblib" && \
    echo "${MODEL_SHA256}  models/lgbm_windows.joblib" | sha256sum -c - && \
    python -c "import joblib; m=joblib.load('models/lgbm_windows.joblib'); print('artifact OK:', type(m).__name__)"

EXPOSE 8000

CMD ["python", "api/main.py"]
