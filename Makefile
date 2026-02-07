.PHONY: install test demo

install:
	python -m pip install -U pip
	python -m pip install -e ".[dev]"

test:
	python -m pip install -U pip
	python -m pip install -e ".[dev]"
	pytest -q

demo:
	@echo "No demo steps defined."
