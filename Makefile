# Makefile

.PHONY: all req hooks ruff test clean

all: hooks clean ## Run hooks and clean pycache

hooks: ## Run pre-commit hooks
	pre-commit run --all-files

req: ## Install requirements
	uv pip install -r pyproject.toml --all-extras
	uv lock

ruff: ## Run Ruff linter separately
	ruff check . --fix --exit-non-zero-on-fix --show-fixes

test: ## Run tests seprately
	pytest


clean: ## Clean up generated files
	rm -rf __pycache__

help: ## Display this help message
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help
