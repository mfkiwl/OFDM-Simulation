# References
1. https://github.com/kornthum/OFDM_64QAM_Python
2. https://dspillustrations.com/pages/posts/misc/python-ofdm-example.html

# Converting
## Convert notebook to pdf
```
cd Source
jupyter nbconvert --to pdf 64_QAM_OFDM.ipynb
```

## Convert notebook to markdown
```
cd Source
jupyter nbconvert --to markdown 64_QAM_OFDM.ipynb
```
## Convert using pandoc
Install pandoc with winget
```
winget install --source winget --exact --id JohnMacFarlane.Pandoc
pandoc --version
```
Convert to latex
```
pandoc --listings -f markdown -t latex 64_QAM_OFDM.md -o 64_QAM_OFDM.tex
pdflatex Source\64_QAM_OFDM.tex
```
Using [eisvogel](https://github.com/Wandmalfarbe/pandoc-latex-template) template
```
pandoc --listings -f markdown --template eisvogel Source\64_QAM_OFDM.md -o Source\64_QAM_OFDM.tex
pandoc --listings -f markdown --template eisvogel Source\64_QAM_OFDM.md -o Source\64_QAM_OFDM.pdf
```
## To convert notebook to latex
```
jupyter nbconvert --to latex Source\64_QAM_OFDM.ipynb
```
# Helpful command
## Clear output before commit
```
jupyter nbconvert --clear-output --inplace Source\64_QAM_OFDM.ipynb
```

## Slide shows
```
jupyter nbconvert --to slides --post serve Source\64_QAM_OFDM.ipynb
```