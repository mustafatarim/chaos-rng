.PHONY: help install install-dev setup-dev test test-slow lint format type-check build check-build pre-commit clean release-check

PYTHON ?= python3
PIP := $(PYTHON) -m pip

help: ## Show available commands
	@echo "Chaos RNG development commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "README-first release flow: documentation lives in README.md for v0.1.1."

install: ## Install package in editable mode
	$(PIP) install -e .

install-dev: ## Install dev/test/crypto dependencies
	$(PIP) install -e .[dev,test,crypto]

setup-dev: install-dev ## Install dependencies and git hooks
	$(PYTHON) -m pre_commit install

test: ## Run main test suite (exclude slow)
	$(PYTHON) -m pytest tests -q -m "not slow"

test-slow: ## Run slow tests
	$(PYTHON) -m pytest tests -q -m slow

lint: ## Run project quality checks (ruff, black, mypy)
	ruff check src tests
	black --check src tests
	mypy src

format: ## Auto-fix lint issues and format code
	ruff check --fix src tests
	black src tests

type-check: ## Run mypy type checking
	mypy src

build: clean ## Build sdist and wheel
	$(PYTHON) -m build

check-build: build ## Validate built package metadata
	$(PYTHON) -m twine check dist/*

pre-commit: ## Run pre-commit hooks on all files
	$(PYTHON) -m pre_commit run --all-files

release-check: lint test test-slow build ## Run local release gate checks
	@echo "Release gate checks completed."

clean: ## Remove build and cache artifacts
	rm -rf build dist *.egg-info htmlcov .pytest_cache .mypy_cache .ruff_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
