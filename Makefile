.PHONY: setup lint format test smoke motion-smoke audio-smoke jscc-smoke gate-smoke scorer-smoke clean

PYTHON ?= python

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements/dev.txt
	$(PYTHON) -m pip install -e . --no-deps
	$(PYTHON) -m pre_commit install

lint:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m ruff format --check src tests scripts

format:
	$(PYTHON) -m ruff check --fix src tests scripts
	$(PYTHON) -m ruff format src tests scripts

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) -m pytest -m smoke

motion-smoke:
	$(PYTHON) -m pytest tests/smoke/test_motion_sensitivity_pipeline.py

audio-smoke:
	$(PYTHON) -m pytest tests/smoke/test_audio_motion_pipeline.py

jscc-smoke:
	$(PYTHON) -m pytest tests/smoke/test_jscc_pipeline.py

gate-smoke:
	$(PYTHON) -m pytest tests/smoke/test_channel_gate_pipeline.py

scorer-smoke:
	$(PYTHON) -m pytest tests/smoke/test_residual_scorer_pipeline.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist htmlcov
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
	find src -type d -name '*.egg-info' -prune -exec rm -rf {} +
