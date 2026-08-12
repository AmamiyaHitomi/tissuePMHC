# TissePMHC v8 Builder Description

## This update

- Replace the old descriptive components with the formal equivalent components of Human/Mouse, 3 Seeds.
- Joined MHC-only CNN, task-to-task matching statistics and input non-transform audits.
- Adds formal three seed single-mixture ISM disturbance authentication, SHAP-ISM consistency analysis, anchor position check and Figure 10.
- Synchronizes the experimental settings, summaries, discussions, limitations, conclusions and translations in Chinese.
- Retains the original v7; old PDFs, supporting files and old renderer maps from v7 have been deleted in v8 to avoid misconstruing the result of the v8 compilation.

## Compile

Set `main.tex` as the main file in Overleaf or run in an environment where the complete TeX Live/MiKTeX is installed:

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

If `latexmk` is not used, at least twice consecutive times `pdflatex main.tex` is run to interpret the cross-references to tables and page numbers.

## Audits completed

- The ZXQ0QZ environment for 17 TeX files is closed.
- No repetition `label`.
- There is no missing cross-reference to the target except for `LastPage`, which was generated at the time of compilation of the `lastpage` package.
- 25 Bibliography entries cover all 23 text references; 8 master chart files exist.
- Adds ISM images that have been independently rendered and seen checked; the TeX source file does not have Unicode replacement characters or an abnormal serial question mark.
- The outdated expressions "formally melted uncompleted" and "trainable missing control".
- The new result is reconciled with the formal summary and pair statistics in `results/occurrence_equal_ablation_mhc_only`, item by item.
- ISM text values are checked with the official results of `extra_occurrence_equal_dataset/results/e29_tuned_ism` by item.

Neither the Windows nor the WSL environment has a usable LaTeX compiler and therefore no v8 PDF is generated or verified on this machine.
