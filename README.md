# References
1. https://github.com/kornthum/OFDM_64QAM_Python
2. https://dspillustrations.com/pages/posts/misc/python-ofdm-example.html

# To Convert notebook to pdf
```
cd Source
jupyter nbconvert --to pdf 64_QAM_OFDM.ipynb
```

# To Convert notebook to markdown
```
cd Source
jupyter nbconvert --to markdown 64_QAM_OFDM.ipynb
```

# To convert markdown to latex and then into pdf
Install pandoc with winget,
```
winget install --source winget --exact --id JohnMacFarlane.Pandoc
pandoc --version
```
```
pandoc --listings -f markdown -t latex 64_QAM_OFDM.md -o 64_QAM_OFDM.tex
pdflatex 64_QAM_OFDM.tex
```
# To convert notebook to latex
```
jupyter nbconvert --to latex Source\64_QAM_OFDM.ipynb
```

# Clear output before commit
```
jupyter nbconvert --clear-output --inplace Source\64_QAM_OFDM.ipynb
```

# Slide shows
```
jupyter nbconvert --to slides --post serve Source\64_QAM_OFDM.ipynb
```