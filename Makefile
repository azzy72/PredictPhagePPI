SCRIPTS_DIR = scripts
HOOKS_PATH = $(SCRIPTS_DIR)
REQUIREMENTS_FILE = requirements.txt
ENV_NAME = PredPPI
PYTHON_VERSION = 3.11

.PHONY: setup folders hooks ignore help requirements clean-reqs env clean-env

setup: folders ignore hooks requirements env
	@echo "🚀 Project fully initialized with .gitignore, requirements.txt, and conda env!"

folders:
	@echo "📁 Creating directories..."
	@mkdir -p data_prod tmp
	@touch data_prod/.gitkeep tmp/.gitkeep

ignore:
	@echo "📝 Writing .gitignore..."
	@/bin/bash $(SCRIPTS_DIR)/post-checkout
	@echo "  [✓] .gitignore created with custom rules."

hooks:
	@echo "🔗 Linking Git hooks..."
	@git config core.hooksPath $(HOOKS_PATH)
	@chmod +x $(SCRIPTS_DIR)/post-checkout

requirements:
	@echo "🔍 Scanning for dependencies in notebooks and scripts..."
	@pip install -q pipreqsnb
	@pipreqsnb . --force --ignore $(SCRIPTS_DIR),tmp
	@sort -o $(REQUIREMENTS_FILE) $(REQUIREMENTS_FILE)
	@echo "  [✓] $(REQUIREMENTS_FILE) updated and sorted."

env:
	@echo "🐍 Creating isolated conda environment '$(ENV_NAME)'..."
	@conda create --name $(ENV_NAME) python=$(PYTHON_VERSION) --no-default-packages -y
	@echo "📦 Installing dependencies..."
	@$$(conda info --base)/envs/$(ENV_NAME)/bin/pip install -r $(REQUIREMENTS_FILE) --no-cache-dir --isolated
	@echo "  [✓] Conda env '$(ENV_NAME)' ready. Run: conda activate $(ENV_NAME)"

clean-reqs:
	@rm -f $(REQUIREMENTS_FILE)
	@echo "  [✓] $(REQUIREMENTS_FILE) removed."

clean-env:
	@echo "🗑️  Removing conda environment '$(ENV_NAME)'..."
	@conda env remove --name $(ENV_NAME) -y
	@echo "  [✓] Conda env '$(ENV_NAME)' removed."