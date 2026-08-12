# Remaining issues of the paper: Universal pMHC baseline versus strict architecture

## Current status

The present findings of the paper already support the following conclusions:

- In standard pair-disjoint benchmark, the human TissePMMHC means task AUROC is 0.844;
- In the matchped peptide-disjoint OOF, the human AUROC is 0.7652;
- The number of people who live in the country is 0.8562;
- If you want to be a part of the world, you can be a part of the world.
- Both species still retain higher-sortized sorting signals under peptide-disjoint conditions, but there is a marked decline in physical generalization.

There are two issues that still need to be addressed by additional experiments:

1. Item 5: Lack of a generic peptide-MHC baseline for binting/presentation baseline;
2. Block 9: Lack of architecture comparison under the same block split.

---

## Block 5: Lack of a baseline for a common combination/presentation model

### What's the problem?

The current model has achieved a high AUROC, but the results alone do not make it possible to determine that these performances are mainly derived from:

- general peptide-MHC binting/presentation strength; or
- TissePMHC learned lesson-convention preference.

Although the positive and negative peptide match the same MHC response and the parent UniProt sequence, the two sequences of peptide are different for each pair. The positive sample may still naturally have a stronger general HLA/H2 combination or delivery capability. Therefore, MHC and parent-protein matchping can reduce mixing, but cannot completely exclude the general pMHC signature.

### Why does it matter?

The paper repeatedly stressed in Relating Work and Discussion that the task is different from the general binting/presentation preparation. If generic predicor is not actually included, the difference is supported mainly by the definition of the task, not by the experimental comparison.

> A generic pMHC predicor that doesn't know anything about Tissue, how much AUROC can be achieved on the same pairing task?

### Minimum Test

Select at least two generic predictors that can freeze operations. Priority overwrite:

1. binting-only copy, for example, NetMHCpan BA/affenity;
2. Presentation scores for the establishment-like code, e.g. NetMHCpan EL, MHCflurry preparation or other operational points.

These predictors should not use this label to fine-tune.

- Tool names and precise versions;
- (a) Model or data file version;
- scoring mode;
- Allele support range;
- Unable to rate the peptide-MHC combination;
- The fractional direction, such as the lower the percentile rank, should be converted to the uniform "highest the better";
- Runs the date, command line and environment.

### Proposed additional internal comparison

In addition to external predicor, it is proposed to add a model for capacitation-matched HLA-only/H2-only:

\[
s=f(\text{peptide},\text{MHC}),
\]

The model uses the same peptide encoder, training budget and split, but cannot access the tisue or tissue-MHC task identity.

> How much performance can be explained by using only the peptide-MHC information under the same data and training conditions?

The existing share-task-head, global branch or HLA-special branch cannot entirely replace this comparison, as they are still trained under the Tissue-convention label or Tissue-MHC task headers.

### Assessment of agreements

All generic scores and HLA-only/H2-only models should be reported separately in the following agreements:

- human standard fixed test;
- human matched standard OOF;
- human peptide-disjoint OOF;
- mouse standard fixed test;
- mouse matched standard OOF;
- mouse peptide-disjoint OOF.

At least calculated in each tissue-MHC task:

- AUROC;
- AUPRC;
- PairAcc;
- Number of task-related tasks and coverage.

Of which:

\[
\mathrm{PairAcc}
=
\frac{1}{N_{\mathrm{pairs}}}
\sum_i
\mathbf{1}
\left[
s(p_i^+)>s(p_i^-)
\right].
\]

The recommendations are reported simultaneously:

- task-macro mean/median;
- Task-by-task differential between the most general base line and TissePMHC;
- win/tie/loss;
- median difference;
- Hodges–Lehmann difference;
- task-bootstrap confidence interval;
- Wilcoxon signed-rank test and BH-FDR.

### Recommended decision logic

If the following results are to be found:

- The general predicor is significantly lower than TissePMHC;
- (a) HLA-only/H2-only models are significantly lower than complete models;
- The complete model still has a steady increment after controlling the generic score;

The blogger writes:

> General peptide–MHC binding/presentation propensity explains part of the benchmark signal but does not fully account for the performance of the tissue-conditioned model.

If the generic predicor or HLA-only model is close to complete, it cannot be claimed that performance is mainly derived from the problem conditioning.

> Much of the observed ranking performance is consistent with general peptide–MHC presentation propensity, while the incremental contribution of tissue conditioning is limited.

### Data leaks attention

External presence predctor may be trained in language materials containing IEDB or related immunopeptidomics data. Even if this is a peptide-disjoint OOF, it is not guaranteed that the peptide-disjoint will be compared to the external model training set.

It is therefore necessary to:

- (b) Reporting on training data sources for external models;
- (b) Clearly identify potential pre-training overlaps where it is not possible to restore the list of training entities;
- It is described as a "general signal contrast" rather than a fair model competition with no leakage at all.

### Output to Save

It is recommended that a separate result directory be added and at least save:

- unique peptide–MHC score cache;
- row-level predictions;
- per-task metrics;
- PairAcc;
- coverage/missing-allele audit;
- paired statistical comparison;
- Tool versions, compands and metadata;
- Summary tables and graphs for the papers.

### Completion criteria

A fifth solution can be considered to be a solution only if the following conditions are met:

- At least two generic pMHC scoring modes or predictors are actually running;
- The results overlay the sstandard and peptide-disjoint agreements;
- Reports AUROC, AUPRC, PairAcc and coverage;
- Task-by-task matching with the main model;
- The paper limited or enhanced the process-conventioned process based on actual results.

---

## Item 9: no structure comparison under the stric protocol

### What's the problem?

The current peptide-disjoint OOF report only the main freezing model:

- human:frozen TissuePMHC;
- mouse:frozen five-seed Factorized MMoE.

So strict results support:

> Learning signals are still available under the conditions of seenn-task, unseen-peptide.

But it cannot be supported:

> TisuePMHC or Factorized MMOE is better than a simpler structure under strict conditions.

The reason is that the structural advantages observed in the standard benchmark may depend on the peptide overlap; it is not possible to judge whether these advantages can be retained if the baseline does not run on exactly the same peptide-disjoint folds.

### Human Minimum Requirements Baseline

At least run on existing human agreed-component peptide-disjoint folds:

1. one-hot logistic regression;
2. strongest traditional peptide baseline;
3. shared peptide encoder with task-specific heads;
4. MLP dual-branch baseline;
5. auxiliary dual branch;
6. frozen TissuePMHC.

The most critical of these structures are:

- shared heads vs auxiliary branch;
- auxiliary/MLP dual branch vs multi-kernel CNN;
- single branch vs rank fusion;
- single seed vs frozen three-seed ensemble.

If the calculation is limited, the minimum acceptable combination is:

- shared encoder with task heads;
- strongest MLP/auxiliary dual-branch baseline;
- TissuePMHC.

### Mouse Minimum Requirements Baseline

At least on the existing mode connected-component peptide-disjoint folds:

1. BLOSUM62 random forest;
2. shared encoder;
3. single-seed Factorized MMoE;
4. frozen five-seed Factorized MMoE.

H2-Kk supplementary experiment may be used as a supplemental experiment but not as a minimum requirement to address block 9.

### Fair comparison requirements

All structures must be used:

- Same pair pool;
- Same agreed-component documents;
- Same outer olds;
- Same task inclusion rule;
- (b) The same evaluation is achieved;
- (b) The same training epoch, batt size and optimizer rules, unless the model itself does require different settings;
- Prefixed seed collection;
- Model selection is made without reading the total combination of cross test or stric OOF results.

It is important to avoid:

- More than ever, the main model is used, and only one unstable Seed is used for the baseline, but differences are interpreted directly as structural benefits;
- Adjusting hyperparameters on the pool OFF projections;
- Regeneration of inconsistent peptide-disjoint folds for different models;
- Only average overall comparisons are made, without reporting mandate-to-task differences.

### Statistical analysis of recommendations

Compare each baseline with the main model by taking a task-paired and reporting at least:

- mean task AUROC/AUPRC;
- worst-group AUROC;
- PairAcc;
- I'm not sure, single-said means and standard treatment;
- Results of the Esemble;
- mean/median task difference;
- Hodges–Lehmann difference;
- win/tie/loss;
- task-bootstrap interval;
- Wilcoxon signed-rank test;
- BH-FDR for predefined models - indicators familyly.

The task sharing of the taskue, MHC, parent protein and peptide components, therefore the result of the task-bootstrap and Wilcoxon should continue to be marked as nominal task-level evidence and cannot be interpreted as evidence of an independent foreign force.

### Recommended decision logic

If TissePMHC stabilizes on the same strictolds better than the share-head and MLP/ auxiliary dual-branch baselines, it can be written:

> The architectural advantage observed under the standard benchmark is retained under connected-component peptide-disjoint evaluation.

If differences decrease or disappear under the condition of strict, write:

> The standard-benchmark architectural advantage does not clearly persist after peptide-identity separation; the strict results support task learnability rather than model superiority.

If only some components retain proceeds, they should be stated on a case-by-case basis, for example:

- (a) Auxiliary retention of proceeds;
- (a) the reduced returns of multi-kernel encoder;
- The Queens are also a part of the world's population.
- The ensemble is primarily reducing random fluctuations.

### Output to Save

Suggested saving for human and mouse:

- frozen split manifest;
- The row-level preparations for each model,seed,old;
- The single-seed and the edemble per-task metrics;
- pair-level PairAcc;
- strict architecture comparison table;
- paired statistics;
- Parameter count, training time and visibility;
- Full metadata with command line.

### Completion criteria

A solution in block 9 can be considered only if the following conditions are met:

- At least one simple share baseline and one strong dual-branch/structured baseline are completed on the same standard olds;
- The main model and baseline have comparable and comparable treatments;
- Report on task-to-task statistics and PairAcc;
- The paper clearly distinguishes between "trust task research" and "project awareness".

---

## Recommended order of execution

1. Freeze and verify existing human/mouse contact fold manpowers;
2. (a) Run item 9 internally;
3. To co-organize all the only peptide-MHC combinations and create external predicor code cache;
4. Runs the generic binting/presentation predicors for block 5;
5. Merge the outer parts into the same row-level evaluation frame;
6. (a) The production of arc, AUPRC, PairAcc, portfolio and pair statistics;
7. Updates the Reults, Discussion and Limiteds based on the results;
8. Finally, decide whether to strengthen the tissue-conventioned and the rule artiecture capims.

## Final delivery

After completing two questions, at least one new paper should be added:

- A generic pMHC comparison table;
- A list of strict anarchicity comparison tables;
- A task-by-task margin map or a sandard/stric comparison map;
- External model coverage and missing allele audits;
- Two complete sets of task-paired statuses;
- The limitations on the overlap of pre-training, the relevance of the mandate and the scope of application are explained.
