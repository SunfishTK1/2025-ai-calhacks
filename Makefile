# GraphRAG Mayavi Visualization Makefile
# =====================================

# Default Python interpreter
PYTHON := python3

# Default parameters
MAX_NODES := 2000
MAX_EDGES := 1000

# Colors for output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m # No Color

.PHONY: help graph demo-2d clean install

# Default target - Show help
help:
	@echo "$(BLUE)GraphRAG Mayavi Visualization Commands$(NC)"
	@echo "======================================"
	@echo ""
	@echo "$(GREEN)Available commands:$(NC)"
	@echo "  $(YELLOW)make graph$(NC)     - Run interactive 3D graph visualization"
	@echo "  $(YELLOW)make demo-2d$(NC)   - Run 2D matplotlib visualization demo"
	@echo "  $(YELLOW)make clean$(NC)     - Clean up generated files"
	@echo "  $(YELLOW)make install$(NC)   - Install required dependencies"
	@echo "  $(YELLOW)make help$(NC)      - Show this help message"
	@echo ""
	@echo "$(GREEN)Examples:$(NC)"
	@echo "  make graph                      # Default 3D visualization"
	@echo "  make demo-2d                    # 2D visualization"
	@echo "  MAX_NODES=200 make graph        # Custom node limit"
	@echo "  MAX_NODES=300 MAX_EDGES=150 make graph  # Custom limits"

# Main graph visualization command
graph:
	@echo "$(BLUE)🌐 Running Interactive 3D Graph Visualization...$(NC)"
	@echo "Parameters: nodes=$(MAX_NODES), edges=$(MAX_EDGES)"
	@$(PYTHON) graphrag_mayavi.py $(MAX_NODES) $(MAX_EDGES) 3d
	@echo "$(GREEN)✓ 3D visualization completed!$(NC)"

# 2D visualization demo
demo-2d:
	@echo "$(BLUE)📊 Running 2D Graph Visualization Demo...$(NC)"
	@echo "Using default limits with 2D matplotlib visualization"
	@$(PYTHON) graphrag_mayavi.py 2d
	@echo "$(GREEN)✓ 2D visualization completed!$(NC)"

# Clean up generated files
clean:
	@echo "$(YELLOW)🧹 Cleaning up generated files...$(NC)"
	@rm -f *.png
	@rm -f *.pkl
	@rm -f *.json
	@rm -rf __pycache__/
	@rm -rf *.pyc
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "$(GREEN)✓ Cleanup completed!$(NC)"

# Install dependencies
install:
	@echo "$(BLUE)📦 Installing required dependencies...$(NC)"
	@if [ -f requirements.txt ]; then \
		echo "Installing from requirements.txt..."; \
		$(PYTHON) -m pip install -r requirements.txt; \
	else \
		echo "Installing core dependencies..."; \
		$(PYTHON) -m pip install networkx numpy matplotlib mayavi; \
	fi
	@echo "$(GREEN)✓ Dependencies installed!$(NC)"

# Default target when just running 'make'
.DEFAULT_GOAL := help
