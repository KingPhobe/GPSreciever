.PHONY: install test demo verify verify-mc

install:
	python -m pip install -U pip
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

demo:
	@echo "No demo steps defined."

verify:
	python -m sim.validation.verification_suite --run-root runs_verify --quick

verify-mc:
	python -m sim.validation.verification_suite --run-root runs_verify --monte-carlo 30
