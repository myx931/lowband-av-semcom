.PHONY: setup lint format test smoke clean

PYTHON ?= python

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements/dev.txt
	$(PYTHON) -m pip install -e . --no-deps
	$(PYTHON) -m pre_commit install

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

format:
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m ruff format src tests

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) -m pytest -m smoke

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist htmlcov
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
	find src -type d -name '*.egg-info' -prune -exec rm -rf {} +
