.PHONY: up down logs rebuild shell nb sync fmt lint precommit

up:
	docker compose up -d --build
	docker compose ps
	echo "Open Jupyter: http://localhost:$${JUPYTER_PORT:-8888}"

e2e: up sync nb ## 全流程：啟動 + 安裝 + 開啟 notebook

sync:
	docker compose exec algolab bash -lc "uv sync --all-extras"

nb:
	@echo "Jupyter running at http://localhost:$${JUPYTER_PORT:-8888}"

shell:
	docker compose exec algolab bash

logs:
	docker compose logs -f --tail=200

fmt:
	docker compose exec algolab bash -lc "uv run black . && uv run ruff check --fix ."

lint:
	docker compose exec algolab bash -lc "uv run ruff check . && uv run mypy src || true"

down:
	docker compose down

rebuild:
	docker compose build --no-cache algolab

api-up:
	docker compose up -d fastapi

api-logs:
	docker compose logs -f fastapi

ui-up:
	docker compose up -d streamlit

ui-logs:
	docker compose logs -f streamlit
