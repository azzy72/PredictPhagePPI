SCRIPTS_DIR = scripts
HOOKS_PATH = $(SCRIPTS_DIR)

.PHONY: setup folders hooks ignore help

setup: folders ignore hooks
	@echo "🚀 Project fully initialized with .gitignore!"

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