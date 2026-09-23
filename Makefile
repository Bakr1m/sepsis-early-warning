.PHONY: install test lint train serve dashboard docker-build docker-run clean

install:            ## Install full local stack
	.venv/bin/pip install -r requirements.txt

test:               ## Hermetic test suite (no data/ needed)
	.venv/bin/python -m pytest tests/ -q

lint:               ## Lint everything
	.venv/bin/ruff check src api tests dashboard

train:              ## Features -> baseline -> LightGBM -> CV (needs data/, ~15 min)
	.venv/bin/python src/features.py
	.venv/bin/python src/baseline.py
	.venv/bin/python src/lgbm.py
	.venv/bin/python src/evaluate.py

serve:              ## API on :8000
	.venv/bin/python api/main.py

dashboard:          ## Streamlit trend viewer on :8501
	.venv/bin/streamlit run dashboard/app.py

docker-build:       ## Build serving image (serving deps only)
	docker build -t sepsis-api .

docker-run:         ## Run serving image on host :8001 (host :8000 often taken)
	docker run -p 8001:8000 sepsis-api

clean:              ## Caches only (never data/, models/, mlruns/)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
