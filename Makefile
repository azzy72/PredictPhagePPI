SCRIPTS_DIR = scripts
HOOKS_PATH = $(SCRIPTS_DIR)
REQUIREMENTS_FILE = requirements.txt

.PHONY: setup folders hooks ignore help requirements clean-reqs

setup: folders ignore hooks requirements
	@echo "🚀 Project fully initialized with .gitignore and requirements.txt!"

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

clean-reqs:
	@rm -f $(REQUIREMENTS_FILE)
	@echo "  [✓] $(REQUIREMENTS_FILE) removed."