# Minimal Makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help clean html open livehtml strict

help:
	@echo "Documentation build targets:"
	@echo "  html      - Build HTML documentation"
	@echo "  clean     - Remove build directory"
	@echo "  open      - Build and open in browser"
	@echo "  livehtml  - Build with auto-reload (requires sphinx-autobuild)"
	@echo "  strict    - Build with warnings as errors"

clean:
	rm -rf $(BUILDDIR)

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)
	@echo "Build finished. HTML pages are in $(BUILDDIR)/html."

open: html
	@if [ "$$(uname)" = "Darwin" ]; then \
		open $(BUILDDIR)/html/index.html; \
	elif [ "$$(uname)" = "Linux" ]; then \
		xdg-open $(BUILDDIR)/html/index.html; \
	else \
		echo "Open $(BUILDDIR)/html/index.html in your browser"; \
	fi

livehtml:
	sphinx-autobuild $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)

strict:
	$(SPHINXBUILD) -b html -W --keep-going $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)
	@echo "Strict build finished."
