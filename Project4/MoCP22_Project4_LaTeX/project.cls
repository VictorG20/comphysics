\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{project}[2022/11/01 Methods of Computational Physics Project]

\LoadClass[fontsize=11pt, draft=false, overfullrule=true]{scrartcl}

\RequirePackage{graphicx}
\RequirePackage{amsmath,amssymb}
\RequirePackage{physics}
\RequirePackage[extstyle]{exframe}
\RequirePackage[dvipsnames]{xcolor}
\RequirePackage{hyperref}
\RequirePackage{siunitx}

\newcommand\myshade{85}
\colorlet{mylinkcolor}{violet}
\colorlet{mycitecolor}{YellowOrange}
\colorlet{myurlcolor}{Aquamarine}

\hypersetup{
  linkcolor  = mylinkcolor!\myshade!black,
  citecolor  = mycitecolor!\myshade!black,
  urlcolor   = myurlcolor!\myshade!black,
  colorlinks = true,
}

\newcommand{\semirange}[2]{\ensuremath{[#1,#2)}}
\newcommand{\goto}{\raisebox{0.225em}{\tiny \ensuremath{\blacktriangleright}}}

\defexercisestyle{myplainheader}{%
\exerciseconfig{styleheadertitle}{\LARGE\bfseries}%
\exerciseconfig{styleheadercourse}{}%
\exerciseconfig{styleheaderbelow}{\bfseries}%
\exerciseconfig{skipheaderbelow}{1em}%
\exerciseconfig{composeheaderbelowleft}{}%
\exerciseconfig{composeheaderbelowright}{}%
\exerciseconfig{composeheaderbelowcenter}{}%
\exerciseconfig{insertsheettitle}{\noindent%
\begin{minipage}{\textwidth}%
{\getexerciseconfig{styleheadercourse}%
  \makebox[0pt][l]{\getexercisedata{university}}\par%
  \makebox[0pt][l]{\getexercisedata{department}}\par%
  \makebox[0pt][l]{\getexercisedata{course}, \getexercisedata{period}}\par%
  \makebox[0pt][l]{Instructor: \getexercisedata{instructor}}\par%
  \makebox[0pt][l]{Tutors: \getexercisedata{tutors}}\par}%
\vspace*{2em}%
{\getexerciseconfig{styleheadertitle}%
  \makebox[\textwidth]{\getsheetdata{title}}\par}%
{\def\tmp{}%
 \exerciseconfigempty{composeheaderbelowleft}{}{\def\tmp{.}}%
 \exerciseconfigempty{composeheaderbelowcenter}{}{\def\tmp{.}}%
 \exerciseconfigempty{composeheaderbelowright}{}{\def\tmp{.}}%
 \exerciseifnotempty{\tmp}%
  {\vspace*{1em}%
  \getexerciseconfig{styleheaderbelow}\vphantom{\^A}%
   \makebox[0pt][l]{\getexerciseconfig{composeheaderbelowleft}}%
   \hfill\makebox[0pt][c]{\getexerciseconfig{composeheaderbelowcenter}}%
   \hfill\makebox[0pt][r]{\getexerciseconfig{composeheaderbelowright}}%
   \vspace*{-\baselineskip}\vspace*{-\parskip}\par}}%
\end{minipage}%
\par\addvspace{\getexerciseconfig{skipheaderbelow}}\noindent%
Go to \href{https://www.studip.uni-goettingen.de/plugins.php/clockedplugin/admin/overview/index?cid=1a9a322e21be361e60889361d2ce1a0f}{Stud.IP \goto\ Übung: Methods of Computational Physics \goto\ CloCked}
to upload a single PDF document with your results and answers by \getsheetdata{due}.}}

\defexercisedata{university}
\defexercisedata{department}
\defexercisedata{tutors}
\exercisedata{
  author={Enrico Bothmann},
  course={Methods of Computational Physics},
  university={University of Göttingen},
  department={Institute for Theoretical Physics},
  instructor={Prof.\ Dr.\ S.\ Schumann},
  tutors={Dr.\ E.\ Bothmann, M.\ Knobbe},
  period={WiSe 2022/23},
  material={Projects},
}
\exercisestyle{myplainheader}

% To switch on solutions via the command line, so something like that:
% pdflatex '\def\ignorelowerbox{true}\input{<filename>}'
\ifdefined\enablesolutions\exercisesetup{solutions=true}\fi
