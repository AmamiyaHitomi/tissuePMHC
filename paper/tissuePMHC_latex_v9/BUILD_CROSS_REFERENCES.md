# Cross-references to main and supplementary texts

`main.tex` and `supplementary_main.tex` are cross-referenced with `xr-hyper`. The two documents cannot be compiled separately from each other; otherwise the pre-compiled document cannot read the ZXQ3QZ file in another document, the valid reference will be queued into `??`.

Runs in a fully micTeX or TeX Live environment:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_papers.ps1
```

Scripts run XeLaTeX alternately in the order of " Supplementary Materials - > Master " until two `.aux` files are stabilized and undissected references, undissected references to literature, no external labels and duplicate tags are checked before the end.

If you use Overleaf, you should compile `supplementary_main.tex` first, and `main.tex` later, and then re-compose them alternately until the log does not suggest that there is a need for a redaction and `??` is not present in the text.
