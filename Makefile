.PHONY: install demo test lint train fetch api ui

install:
	uv sync --extra dev

demo:
	uv run python -m triagedesk.cli train-demo

fetch:
	uv run python -m triagedesk.cli fetch-data

train:
	uv run python -m triagedesk.cli train

test:
	uv run pytest -q

lint:
	uv run ruff check .

api:
	uv run uvicorn triagedesk.serving.app:app --reload --port 8000

ui:
	uv run streamlit run ui/app.py
