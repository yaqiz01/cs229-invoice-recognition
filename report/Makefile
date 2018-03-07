PAPER = report
TEX = $(wildcard *.tex text/*.tex)
BIB = references.bib
FIGS = $(wildcard figs/*.pdf figs/*.png figure/*.pdf figure/*.png)

.PHONY: all export clean

$(PAPER).pdf: $(TEX) $(BIB) $(FIGS)
	echo $(FIGS)
	pdflatex $(PAPER)
	bibtex $(PAPER)
	pdflatex $(PAPER)
	pdflatex $(PAPER)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out $(PAPER).pdf

