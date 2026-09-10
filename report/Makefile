# Needs a TeX distribution: sudo pacman -S texlive-latexrecommended
fleetsplit.pdf: fleetsplit.tex
	pdflatex -interaction=nonstopmode fleetsplit.tex
	pdflatex -interaction=nonstopmode fleetsplit.tex   # second pass for refs

clean:
	rm -f fleetsplit.aux fleetsplit.log fleetsplit.out fleetsplit.pdf
