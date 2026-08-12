# TisuePMHC v8 Builder Description

## This update

- Replace the old descriptive component comparison with the formal equivalent components of Human/Mouse, 3 Seeds.
- Joined MHC-only CNN, task-to-task matching statistics and input non-transform audits.
- Adds formal three seed single-mixed ISM disturbance authentication, SHAP-ISM consistency analysis, anchor location check and Figure 10.
- Synchronizes the experimental settings, summaries, discussions, limitations, conclusions and translations in Chinese.
- Retain v7 original; old PDFs, supporting documents and old renderer maps from v7 have been deleted in v8 to avoid misconstruing the result as v8.

## Compile

Set `main.tex` as the main file in Overleaf or operate in a complete TeX Live/MiKTeX environment:

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

If `latexmk` is not used, at least twice consecutive times `pdflatex main.tex` is used to interpret the table and page number cross-references.

## Permanent audits completed

- The ZXQ0QZ environment for 17 TeX files is closed.
- No repetition `label`.
- There are no missing cross-references to targets, except for `LastPage`, which was generated during the compilation of the `lastpage` package.
- 25 Bibliography entries cover all 23 text references; 8 master chart files exist.
- Adds an ISM image that has been read and checked independently; the TeX source file does not have a Unicode replacement character or an unusual continuous question mark.
- Obsolete expressions such as "officially melted uncompleted" and "trainable tissue-blind control" have been removed.
- The new result is reconciled with the formal aggregation and pairing statistics in `results/occurrence_equal_ablation_mhc_only`, item by item.
- ISM text values are checked item by item with the official results of `extra_occurrence_equal_dataset/results/e29_tuned_ism`.

Neither the Windows nor the WSL environment has a usable LaTeX compiler, and therefore no v8 PDF was generated or visually verified on this machine.
