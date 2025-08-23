.PHONY: help install install-dev test test-fast test-statistical test-coverage test-benchmark clean lint format type-check security docs docs-serve build upload upload-test release pre-commit setup-dev

# Default target
help: ## Show this help message
	@echo "Chaos RNG Development Commands"
	@echo "==============================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Common workflows:"
	@echo "  make setup-dev     # Set up development environment"
	@echo "  make test          # Run all tests"
	@echo "  make lint          # Check code quality"
	@echo "  make format        # Format code"
	@echo "  make docs-serve    # Serve documentation locally"
	@echo "  make release       # Create a release"

# Environment setup
install: ## Install the package
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e .[all]
	pip install -r requirements-dev.txt

setup-dev: install-dev ## Complete development environment setup
	pre-commit install
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify everything works."

# Testing
test: ## Run all tests
	pytest tests/ -v

test-fast: ## Run fast tests only (exclude slow statistical tests)
	pytest tests/ -v -m "not slow"

test-statistical: ## Run statistical validation tests
	pytest tests/ -v -m statistical

test-integration: ## Run integration tests
	pytest tests/ -v -m integration

test-coverage: ## Run tests with coverage reporting
	pytest tests/ -v --cov=chaos_rng --cov-report=html --cov-report=xml --cov-report=term

test-benchmark: ## Run performance benchmarks
	pytest tests/ -k benchmark --benchmark-only --benchmark-json=benchmark.json

test-parallel: ## Run tests in parallel
	pytest tests/ -v -n auto

# Code quality
lint: ## Run all linting checks
	@echo "Running ruff..."
	ruff check src/ tests/
	@echo "Running black check..."
	black --check src/ tests/
	@echo "Running mypy..."
	mypy src/
	@echo "Running isort check..."
	isort --check-only src/ tests/

format: ## Format code with black and isort
	black src/ tests/
	isort src/ tests/
	@echo "Code formatted successfully!"

format-check: ## Check if code is properly formatted
	black --check src/ tests/
	isort --check-only src/ tests/

type-check: ## Run type checking
	mypy src/

# Security
security: ## Run security checks
	bandit -r src/ -f json -o bandit-report.json
	safety check
	pip-audit

# Documentation
docs: ## Build documentation
	cd docs && make html

docs-clean: ## Clean documentation build
	cd docs && make clean

docs-serve: docs ## Build and serve documentation locally
	cd docs/_build/html && python -m http.server 8000

docs-live: ## Live-reload documentation server
	cd docs && sphinx-autobuild . _build/html --host 0.0.0.0 --port 8000

docs-linkcheck: ## Check for broken links in documentation
	cd docs && make linkcheck

# Building and packaging
clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean ## Build source and wheel packages
	python -m build

check-build: build ## Check built packages
	twine check dist/*

# Publishing
upload-test: build check-build ## Upload to Test PyPI
	twine upload --repository testpypi dist/*

upload: build check-build ## Upload to PyPI
	twine upload dist/*

# Version management
bump-patch: ## Bump patch version
	bump2version patch

bump-minor: ## Bump minor version
	bump2version minor

bump-major: ## Bump major version
	bump2version major

# Release workflow
release-patch: ## Create a patch release
	@echo "Creating patch release..."
	$(MAKE) test
	$(MAKE) lint
	$(MAKE) docs
	bump2version patch
	git push origin main --tags
	@echo "Patch release created! Check GitHub Actions for build status."

release-minor: ## Create a minor release
	@echo "Creating minor release..."
	$(MAKE) test
	$(MAKE) lint
	$(MAKE) docs
	bump2version minor
	git push origin main --tags
	@echo "Minor release created! Check GitHub Actions for build status."

release-major: ## Create a major release
	@echo "Creating major release..."
	$(MAKE) test
	$(MAKE) lint
	$(MAKE) docs
	bump2version major
	git push origin main --tags
	@echo "Major release created! Check GitHub Actions for build status."

# Pre-commit
pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	pre-commit autoupdate

# Development utilities
install-hooks: ## Install git hooks
	pre-commit install

profile: ## Profile the code performance
	python -m cProfile -o profile.stats -m pytest tests/ -k benchmark
	@echo "Profile saved to profile.stats. View with: python -c 'import pstats; pstats.Stats(\"profile.stats\").sort_stats(\"cumulative\").print_stats(20)'"

memory-profile: ## Run memory profiling
	mprof run --python python -m pytest tests/ -k benchmark
	mprof plot

# Docker (if you add Docker support later)
docker-build: ## Build Docker image
	docker build -t chaos-rng .

docker-test: ## Run tests in Docker
	docker run --rm chaos-rng make test

# Environment info
env-info: ## Show environment information
	@echo "Python version: $$(python --version)"
	@echo "Pip version: $$(pip --version)"
	@echo "Current directory: $$(pwd)"
	@echo "Git branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repository')"
	@echo "Git status:"
	@git status --short 2>/dev/null || echo "Not a git repository"

# All-in-one quality check
qa: lint type-check security test-fast ## Run all quality assurance checks

# CI simulation
ci: clean install-dev lint type-check security test docs build ## Simulate CI pipeline locally

# Quick development check
dev-check: format lint test-fast ## Quick development workflow check
