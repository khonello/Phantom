.PHONY: help install lint typecheck test run clean setup-dev

# Default target
.DEFAULT_GOAL := help

# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    VENV_BIN := venv/bin
    ACTIVATE := . venv/bin/activate
endif
ifeq ($(UNAME_S),Darwin)
    VENV_BIN := venv/bin
    ACTIVATE := . venv/bin/activate
endif
ifeq ($(OS),Windows_NT)
    VENV_BIN := venv\Scripts
    ACTIVATE := venv\Scripts\activate.bat
endif

help:
	@echo "roop-cam Makefile - Development Commands"
	@echo "========================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup-dev      Create virtual environment and install dependencies"
	@echo "  make install        Install dependencies (assumes venv exists)"
	@echo "  make clean          Remove virtual environment and cache files"
	@echo ""
	@echo "Development:"
	@echo "  make lint           Run flake8 linting checks"
	@echo "  make typecheck      Run mypy type checking"
	@echo "  make test           Run all checks + integration test"
	@echo ""
	@echo "Running:"
	@echo "  make run            Launch GUI (python pipeline.py)"
	@echo "  make run-cli        Run CLI help (python pipeline.py --help)"
	@echo ""
	@echo "Examples:"
	@echo "  make setup-dev      # First time setup"
	@echo "  make test           # Before committing code"
	@echo "  make run            # Start the application"

setup-dev:
	@echo "Setting up development environment..."
ifeq ($(UNAME_S),Linux)
	@bash local/local-setup.sh
else ifeq ($(UNAME_S),Darwin)
	@bash local/local-setup.sh
else
	@echo "Windows detected - run: local\local-setup-windows.bat"
endif

install:
	@echo "Installing dependencies..."
ifeq ($(UNAME_S),Linux)
	$(ACTIVATE) && pip install -r requirements-ci.txt
else ifeq ($(UNAME_S),Darwin)
	$(ACTIVATE) && pip install -r requirements-ci.txt
else
	@echo "Activate venv first: venv\Scripts\activate.bat"
	@echo "Then run: pip install -r requirements-ci.txt"
endif

lint:
	@echo "Running flake8..."
	flake8 pipeline.py pipeline
	@echo "✓ Linting passed"

typecheck:
	@echo "Running mypy..."
	mypy pipeline.py pipeline
	@echo "✓ Type checking passed"

test:
	@echo "Running all tests..."
ifeq ($(UNAME_S),Linux)
	@bash local/local-run-tests.sh
else ifeq ($(UNAME_S),Darwin)
	@bash local/local-run-tests.sh
else
	@echo "Windows detected - run: local\local-run-tests.bat"
endif

run:
	@echo "Launching roop-cam GUI..."
	python pipeline.py

run-cli:
	@echo "CLI options:"
	python pipeline.py --help

clean:
	@echo "Cleaning up..."
	rm -rf venv/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf temp/
	rm -rf .test_output.mp4
	@echo "✓ Cleanup complete"

.PHONY: help setup-dev install lint typecheck test run run-cli clean
