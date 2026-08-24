# Makefile for the paper draft
# Usage:
#   make          — build PDF (requires pdflatex + bibtex in PATH)
#   make clean    — remove build artefacts
#   make watch    — rebuild on save (requires latexmk + inotifywait or fswatch)

MAIN = main

.PHONY: all clean watch

all:
	pdflatex $(MAIN).tex
	bibtex   $(MAIN)
	pdflatex $(MAIN).tex
	pdflatex $(MAIN).tex

# Alternatively, use latexmk (recommended):
# all:
# 	latexmk -pdf $(MAIN).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz

watch:
	latexmk -pdf -pvc $(MAIN).tex
