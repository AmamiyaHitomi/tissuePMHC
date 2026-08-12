#occurence-equal tissuePHC: E29 SHAP Biological Interpretation Report

# Analyse object

- Data: Human occurrence-equival fixed train/test; positive and negative samples maintain pair structure.
- Model: Freezing E29 (global-auxiliary and HLA-special).
- Method: Each Tissue-HLA task uses its own training sample as a gradient SHAP background to explain the test sequence of the thong.
- Scale: the logit of the SHAP counterpart; the thermal map is the average of the positive samples, minus the average of the negative samples.

# Quality control

- Re-release of the results: Maximum absolute forecast difference = 1.11022e-16; Minimum Pearson r = 1.000000.
- Seed stability: Seed two two Spearman r median = 0.8539.
- Maximum SHAP plus and error: 3.44598.
- Total running time: 1550.631 seconds.

# The strongest signal to the base - position

The ranking of the -negative means SHAP-
|---:|---:|:---:|---:|
|1|3|D|0.213195|
|2|2|L|0.165970|
|3|2|Y|0.095409|
|4|1|K|0.094628|
|5|2|P|0.078184|
|6|2|F|0.075305|
|7|2|H|0.072986|
|8|9|V|0.068989|
|9|2|R|0.064750|
|10|9|L|0.063400|

# The most negative base - position signal

The ranking of the -negative means SHAP-
|---:|---:|:---:|---:|
|1|7|D|-0.002200|
|2|4|H|-0.001817|
|3|4|C|-0.000683|
|4|9|P|-0.000604|
|5|9|S|-0.000255|
|6|9|N|-0.000201|
|7|5|N|0.000128|
|8|9|D|0.000232|
|9|1|P|0.000293|
|10|2|N|0.000527|

# Explain the boundary

1. Ultimately, E29 uses the percentile-rank integration in task, which is not insignificant; therefore the two branches of SHAP are precise branch interpretations, and the branch consensus is merely descriptive averages.
The SHAP description model relies on the non-proven causal mechanisms; location/residual base conclusions need to be cross-established with known HLA motif or independent experimental validation.
The background of the task condition removes the baseline differences in the organization/HLA; the result is explained by the sequence differentiation within the same tissue-HLA, rather than by the causal effect of the tissue variable itself.
4. The design of the occurrence-equival has reduced the number of interorganizational mixings, but has not eliminated the differences in source protein, detection processes and database intake.
