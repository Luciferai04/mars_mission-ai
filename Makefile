.PHONY: help setup install test lint format run clean

help:
	@echo "Mars Mission Planning Assistant - Makefile"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup         - Setup virtual environment and install dependencies"
	@echo "  make install       - Install dependencies only"
	@echo "  make test          - Run test suite"
	@echo "  make lint          - Run linting checks"
	@echo "  make format        - Format code with black"
	@echo "  make run           - Start FastAPI server"
	@echo "  make run-dev       - Start server in development mode"
	@echo "  make clean         - Clean cache and temporary files"
	@echo "  make fetch-demo-dem- Download demo DEM"
	@echo ""

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -r requirements.txt
	@echo "Setup complete! Activate with: source .venv/bin/activate"

install:
	pip install -r requirements.txt

test:
	pytest -v tests/

test-cov:
	pytest --cov=src --cov-report=html tests/

lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	black src/ tests/
	ruff check --fix src/ tests/

run:
	uvicorn src.interfaces.web_interface:app --host 0.0.0.0 --port 8000

run-dev:
	uvicorn src.interfaces.web_interface:app --reload --host 0.0.0.0 --port 8000

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

fetch-demo-dem:
	@echo "Downloading demo DEM from USGS Planetary Maps..."
	mkdir -p data/dem
	curl -o data/dem/jezero_demo.tif $(url)

fetch-official-dems:
	python scripts/fetch_official_dems.py --dems mola hrsc gale

train-hazard-geotiff:
	python scripts/train_from_geotiff.py \
		--dem data/dem/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif \
		--dem data/dem/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif \
		--lat-min 18.2 --lat-max 18.7 --lon-min 77.2 --lon-max 77.7 \
		--output models/hazard_detector_mars_dems.pkl

build-hazard-ds:
	python scripts/build_hazard_dataset.py --dem $(dem) --output $(out) --lat $(lat)

train-hazard:
	python scripts/train_hazard_model.py --manifest $(manifest) --output $(out)

eval-hazard:
	python scripts/eval_hazard_model.py --manifest $(manifest) --model $(model)

run-scenario:
	curl -X POST http://localhost:8000/plan_scenario \
		-H "Content-Type: application/json" \
		-d @$(file)

plan-cli:
	python scripts/plan_cli.py --input $(file) --output $(out)

fetch-mastcamz-meta:
	python scripts/fetch_mastcamz_metadata.py --sol $(sol) --output $(out)
