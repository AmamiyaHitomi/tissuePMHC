#occurence-equal tisuePHC: E29 ISM Disturbation Validation Report

# Analyse design

- Model: official three-said E29 checkpoints, including the global-auxiliary and HLA-specific branches.
- Sample: 20 complete positive or negative pairs per mission for 3080 peptunium.
- Disturbation: each of the 9-mer points is replaced by the remaining 19 standard amino acids.
- Total mutant scores: 526,680 average three-sied monoamino acid replacements.
- Effect measure: the branch logit change and the final recalculated percentage change in task by reference distribution for fixed tests.
- Definition of mutant fractions: minus original platinum fractions; negative values represent mutation reduction model projections.

# Quality control

- The greatest absolute error in the probability of the original branch of the pyridium and the preservation of the forecast: 4.42e-07.
- Maximum absolute error in recalculating rank result with saving result: 1.11e-16.
- Three-seed positions - replacement effect Spearman median: global = 0.53999, HLA = 0.5484, Fusion = 0.5414.
- Total running time: 155.304 seconds.

# The main result

# Position sensitivity

- The most sensitive positions for the final integration scores for the entire sample are, in order of precedence: P2, P3, P9, P1, P5.
- Among positive pyromium, the most significant disruptions to positive predictions were in the following places: P2, P3, P9, P1, P7.
- P2/P9 average sensitivity minus 0.059010, 95% of the remaining position of the Bootstrap CI [0.0566335, 0.061458], one side Wilcoson FDR < 1e-300.
- In the positive and negative pair analysis, the three largest loss locations for the positive platinum classification are: P2, P3, P9.

# Consistency with Expected-IG/SHAP

- Average mutation losses of ISM by sample are related to the median number of ISMs associated with Expected-IG observed-residement: global=0.78770 and HLA=0.8820.
- The two measures are not identical: ISM compares the original pyromium to all single-point mutagenics relative to the background of the mission; therefore the medium-high correlation constitutes independent disturbance evidence, but does not require an equal value.

# The strongest drop mutation (final rank Fusion)

The sample is the original position of the sample, which is the replacement of the lang fasion.
|---|---|---:|:---:|---:|---|---|
|humanPMHC_test_000000003415|QLSLRTVSL|2|L→P|-0.915000|lung|HLA-A*02:01|
|humanPMHC_test_000000002759|NYAPAFTML|2|Y→P|-0.888333|bone|HLA-A*02:01|
|humanPMHC_test_000000003415|QLSLRTVSL|2|L→Y|-0.881667|lung|HLA-A*02:01|
|humanPMHC_test_000000005027|KPMKTSPEM|2|P→F|-0.866667|lymphoid|HLA-B*07:02|
|humanPMHC_test_000000002773|FYIWRPLRI|2|Y→L|-0.864167|bone|HLA-A*02:01|
|humanPMHC_test_000000004529|KLDKQYESL|3|D→Y|-0.863333|lymph node|HLA-C*05:01|
|humanPMHC_test_000000004529|KLDKQYESL|3|D→F|-0.861667|lymph node|HLA-C*05:01|
|humanPMHC_test_000000005005|SPMGSTEDL|2|P→F|-0.855000|lymphoid|HLA-B*07:02|
|humanPMHC_test_000000004529|KLDKQYESL|3|D→L|-0.848333|lymph node|HLA-C*05:01|
|humanPMHC_test_000000003424|STAPPAHGV|2|T→P|-0.841667|lung|HLA-A*02:01|

# The most powerful ascending mutation (final rank Fusion)

The sample is the original position of the sample, which is the replacement of the lang fasion.
|---|---|---:|:---:|---:|---|---|
|humanPMHC_test_000000000278|SPDLRLTWL|2|P→W|0.920000|blood|HLA-A*02:01|
|humanPMHC_test_000000000278|SPDLRLTWL|2|P→F|0.918333|blood|HLA-A*02:01|
|humanPMHC_test_000000000278|SPDLRLTWL|2|P→Y|0.896667|blood|HLA-A*02:01|
|humanPMHC_test_000000004406|IHQDGIHIL|2|H→G|0.836667|lymph node|HLA-C*03:04|
|humanPMHC_test_000000000278|SPDLRLTWL|2|P→H|0.826667|blood|HLA-A*02:01|
|humanPMHC_test_000000007444|IFDVLPNFF|3|D→M|0.810000|uterine cervix|HLA-A*24:02|
|humanPMHC_test_000000001502|QEVMKWNGW|2|E→L|0.800000|blood|HLA-B*44:02|
|humanPMHC_test_000000007444|IFDVLPNFF|3|D→V|0.793333|uterine cervix|HLA-A*24:02|
|humanPMHC_test_000000007444|IFDVLPNFF|3|D→G|0.791667|uterine cervix|HLA-A*24:02|
|humanPMHC_test_000000000278|SPDLRLTWL|1|S→H|0.776667|blood|HLA-A*02:01|

# Explain the boundary

1. ISM has demonstrated that the model relies on the calculation of single point replacement and is not equivalent to internal causal effects.
2. The mutant thorium may not be in the training distribution; the extreme effects shall be reviewed in conjunction with the known HLA motif, the combined experiments or the external forecaster.
The final rank effect is fixed by the original test task fractions and replaces the original sample with the mutant; it is not a re-entry of the entire test set into a combined mutation.
4. P2/P9 is a classic anchor position assumption for cross-HLA aggregation, and the equivalent genetic speciality model should be based on the HLA stratification result.
5. This analysis follows the same sample of the SHAP for each mission test Pair; the conclusion represents the sample explained and should not be extrapolated to the full potential of platinum.
