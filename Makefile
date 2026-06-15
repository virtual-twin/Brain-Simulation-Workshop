# Workshop slide deck — common tasks.
# Run `make` or `make help` to list targets.

.DEFAULT_GOAL := help
.PHONY: help preview render clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

preview: ## Live-preview slides.qmd with chapter reload (pass ARGS=...)
	./scripts/slides-preview.sh $(ARGS)

render: ## Render slides.qmd to the output directory
	uv run quarto render slides.qmd

clean: ## Remove Quarto render/freeze caches
	rm -rf _site .quarto
