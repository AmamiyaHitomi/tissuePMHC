# Results version labels

The official results of this directory are:

- Chinese: **tissuePMHC (ref. following; mouse training set CV selection)**
- English: **TissuePMHC (tuned; mouse-training-CV-selected)**
- Internal lock profile ID: `c0_combined_top4 '

This result uses the mouse 's occurrence-equival training set to select the CV configuration using three official seeds
Re-training on the complete training set. The fixed test set is not used to select the lock configuration.

When these values are quoted in papers, tables, graphs or text, the " after reference " or `tuned ' shall be retained:

- Average of the three-seed indicators: AUROC 0.8194, AUPPC 0.8279;
- Three-seed line-by-line forecast: AUROC 0.8351, AUPRC 0.8447.

The original default configuration of the TissuePMHC is 0.7705/0.7785 for AUROC/AUPRC, and the two sets of results cannot be confused.
