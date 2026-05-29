.PHONY: help setup lint test db-up db-down load-raw etl ge eda features train dashboard kafka-up kafka-down demo clean

PYTHON ?= python
VENV   ?= .venv
PIP    := $(VENV)/Scripts/pip
PY     := $(VENV)/Scripts/python
COMPOSE := docker compose -f docker/docker-compose.yml

help:                  ## Show this help.
	@echo "Targets:"
	@echo "  setup        Create venv and install requirements"
	@echo "  lint         Ruff + black --check"
	@echo "  test         Run pytest"
	@echo "  db-up        docker compose up Postgres + Airflow + Kafka + MLflow"
	@echo "  db-down      docker compose down"
	@echo "  load-raw     Load CSV files into raw schema"
	@echo "  etl          Trigger Airflow DAG ingest_acv_daily"
	@echo "  ge           Run Great Expectations checkpoints"
	@echo "  eda          Run notebooks 01-03 with papermill"
	@echo "  features     Build patient-level feature store"
	@echo "  train        Train models, log to MLflow"
	@echo "  dashboard    Regenerate reports/dashboard.html from DW"
	@echo "  kafka-up     Start producer + consumer"
	@echo "  demo         Full end-to-end demo"
	@echo "  clean        Tear down environment"

setup:                 ## Create venv and install dependencies.
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@if not exist .env copy .env.example .env

lint:
	$(PY) -m ruff check src tests
	$(PY) -m black --check src tests

test:
	$(PY) -m pytest -q

db-up:
	$(COMPOSE) up -d postgres mlflow

db-down:
	$(COMPOSE) down

load-raw:
	$(PY) -m acv.io.load_csv_to_raw

etl:
	@echo "[etl] triggering DAG ingest_acv_daily (implemented in Phase 3)"

ge:
	@echo "[ge] running checkpoints (implemented in Phase 3)"

eda:
	@echo "[eda] executing notebooks (implemented in Phase 4)"

features:
	@echo "[features] building agg_patient (implemented in Phase 5)"

train:
	@echo "[train] training models (implemented in Phase 6)"

dashboard:
	@echo "[dashboard] regenerating reports/dashboard.html (implemented in Phase 7)"

kafka-up:
	$(COMPOSE) up -d zookeeper kafka

demo:
	@echo "[demo] end-to-end run (Phase 10)"

clean:
	$(COMPOSE) down -v
	@if exist $(VENV) rmdir /s /q $(VENV)
