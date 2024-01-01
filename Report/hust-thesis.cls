\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{hust-thesis}[This template is translated by Trang Nguyen Minh and Hanh Nguyen Hong, from BKIC 611 lab with love <3]
% Original : https://www.overleaf.com/latex/templates/graduation-thesis-template-eng-ver/hbnynvndycyr
\LoadClass{article}
\RequirePackage[utf8]{inputenc}
\RequirePackage[T5]{fontenc} % Vietnamese
\RequirePackage[fontsize=13pt]{scrextend} % Set fontsize=13pt
%\RequirePackage[paperheight=29.7cm,paperwidth=21cm,right=2cm,left=3cm,top=2cm,bottom=2.5cm]{geometry}
\RequirePackage[paperheight=29.7cm,paperwidth=21cm,right=2cm,left=3cm,top=2cm,bottom=2.5cm,twoside]{geometry}
\RequirePackage{mathptmx} % Time New Roman
\RequirePackage{graphicx}
\RequirePackage{float}
\RequirePackage{tikz}
\RequirePackage{subcaption}

\usetikzlibrary{calc}
\RequirePackage{indentfirst}
\renewcommand{\baselinestretch}{1.2} % Line space 1.2
\setlength{\parskip}{6pt} % Spacing after
\setlength{\parindent}{1cm} % Set indent for each paragraph
\RequirePackage{titlesec}
\setcounter{secnumdepth}{4} % 4 Heading
\titlespacing*{\section}{0pt}{0pt}{30pt} % Heading 1
\titleformat*{\section}{\fontsize{16pt}{19.2pt}\selectfont \bfseries \centering}

\titlespacing*{\subsection}{0pt}{10pt}{0pt} % Heading 2
\titleformat*{\subsection}{\fontsize{14pt}{16.8pt}\selectfont \bfseries}

\titlespacing*{\subsubsection}{0pt}{10pt}{0pt} % Heading 3
\titleformat*{\subsubsection}{\fontsize{13pt}{15.6pt}\selectfont \bfseries \itshape}

\titlespacing*{\paragraph}{0pt}{10pt}{0pt} % Heading 4
\titleformat*{\paragraph}{\fontsize{13pt}{15.6pt}\selectfont \itshape}

\renewcommand{\figurename}{\fontsize{12pt}{0pt}\selectfont \bfseries Figure}
\renewcommand{\thefigure}{\thesection.\arabic{figure}}
\RequirePackage[font=bf]{caption}
\captionsetup[figure]{labelsep=space}

\renewcommand{\tablename}{\fontsize{12pt}{0pt}\selectfont \bfseries Table}
\renewcommand{\thetable}{\thesection.\arabic{table}}
\captionsetup[table]{labelsep=space}

\RequirePackage{multicol,multirow,tabularx}
\newcolumntype{s}{>{\hsize=.3\hsize}X}
\newcolumntype{y}{>{\hsize=.4\hsize}X}
\newcolumntype{d}{>{\hsize=.1\hsize}X}
\newcolumntype{a}{>{\hsize=1.1\hsize}X}
\newcolumntype{g}{>{\hsize=5\hsize}X}
\newcolumntype{C}[1]{>{\hsize=#1\hsize\centering\arraybackslash}X}
\newcolumntype{R}[1]{>{\hsize=#1\hsize\raggedleft\arraybackslash}X}
\newcolumntype{L}[1]{>{\hsize=#1\hsize\raggedright\arraybackslash}X}
\renewcommand{\tabularxcolumn}[1]{>{\small}m{#1}}

\renewcommand{\theequation}{\thesection.\arabic{equation}}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{defn}[theorem]{Defenition}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{lemma}[theorem]{Lemma}

\RequirePackage{lipsum} % Thư viện tạo chữ linh tinh.
\renewcommand{\contentsname}{TABLE OF CONTENTS}
\renewcommand{\listfigurename}{LIST OF FIGURES}
\renewcommand{\listtablename}{LIST OF TABLES}
\renewcommand{\refname}{REFERENCES}

\RequirePackage[unicode]{hyperref}
\RequirePackage{colortbl}
\definecolor{LightCyan}{rgb}{0.88,1,1}
\RequirePackage{forloop}
\newcounter{loopcntr}
\newcommand{\rpt}[2][1]{\forloop{loopcntr}{0}{\value{loopcntr}<#1}{#2}}

\RequirePackage[ddmmyyyy]{datetime}
\RequirePackage{amsmath}
\RequirePackage{nameref}
\RequirePackage{rotating}
\RequirePackage{bm}
\DeclareRobustCommand{\vect}[1]{\bm{#1}}
\pdfstringdefDisableCommands{%
  \renewcommand{\vect}[1]{#1}%
}