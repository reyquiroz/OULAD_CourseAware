# Makefile for the paper draft
#
# First-time setup (after installing BasicTeX via `brew install --cask basictex`):
#   eval "$(/usr/libexec/path_helper)"   # reload PATH in current shell
#   sudo tlmgr update --self
#   sudo tlmgr install acmart booktabs microtype latexmk caption
#
# Usage:
#   make          — build PDF with latexmk (recommended)
#   make once     — single pdflatex + bibtex pass (faster, may need re-run)
#   make clean    — remove build artefacts
#   make watch    — continuous rebuild on save

MAIN = main

.PHONY: all once clean watch check

# latexmk handles the full pdflatex → bibtex → pdflatex × 2 sequence automatically
all:
	latexmk -pdf -interaction=nonstopmode $(MAIN).tex

# Manual sequence (fallback if latexmk is not installed)
once:
	pdflatex -interaction=nonstopmode $(MAIN).tex
	bibtex   $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz

watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(MAIN).tex

# Check that required TeX packages are installed
check:
	@kpsewhich acmart.cls  > /dev/null 2>&1 && echo "acmart.cls   OK" || echo "acmart.cls   MISSING — run: sudo tlmgr install acmart"
	@kpsewhich booktabs.sty > /dev/null 2>&1 && echo "booktabs.sty OK" || echo "booktabs.sty MISSING — run: sudo tlmgr install booktabs"
	@kpsewhich microtype.sty > /dev/null 2>&1 && echo "microtype.sty OK" || echo "microtype.sty MISSING — run: sudo tlmgr install microtype"
	@which latexmk > /dev/null 2>&1 && echo "latexmk      OK" || echo "latexmk      MISSING — run: sudo tlmgr install latexmk"
