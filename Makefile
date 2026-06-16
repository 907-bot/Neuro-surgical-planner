.PHONY: install test lint format dashboard api docker-up docker-down demo clean

# ─── Setup ────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install black isort flake8 mypy pytest-cov

# ─── Dev ──────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

lint:
	flake8 src/ scripts/ tests/ --max-line-length=100 --ignore=E501,W503

format:
	black src/ scripts/ tests/ notebooks/ --line-length=100
	isort src/ scripts/ tests/ --profile=black

# ─── Run ──────────────────────────────────────────────────────────────────────
dashboard:
	streamlit run scripts/dashboard.py

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

demo:
	python scripts/run_pipeline.py --causal-only --tumor-size 0.5 --n-simulations 200

demo-mock:
	python scripts/run_pipeline.py --mock --patient-id DEMO_001

# ─── Training ─────────────────────────────────────────────────────────────────
generate-data:
	python scripts/train_gnn.py generate --n-patients 100 --output-dir data/processed/graphs

train-gnn:
	python scripts/train_gnn.py train \
		--data-dir data/processed/graphs \
		--epochs 100 \
		--batch-size 8 \
		--output models/gnn_checkpoint.pt

# ─── Docker ───────────────────────────────────────────────────────────────────
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-build:
	docker-compose build

docker-logs:
	docker-compose logs -f api

# ─── Notebook ─────────────────────────────────────────────────────────────────
notebook:
	cd notebooks && jupyter lab

# ─── Clean ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache htmlcov .coverage outputs/hydra
	echo "Cleaned."
