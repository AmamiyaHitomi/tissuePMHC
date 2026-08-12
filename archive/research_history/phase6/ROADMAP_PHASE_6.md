# Phase 6 Roadmap: Difficulties of mouse tissue-H2 H2 Orientation Upgrade

Number range: Continue Phase 5, beginning with **E25**.
Development anchor: Phase 4 E15,5-seed task-balanced Factorized MMOE probability ensemble.
Current development results: mean task AUROC ZXQ0QZ, mean task AUPRC ZXQ1QZ, first-6 AUROC ZXQ2QZ.

## 1. Research issues and boundaries

The low score of E15 H2 is not equivalent: all tested baselines for H2-Kb are low, indicating that the group may have weak signals; H2-Kd has declined in the global sharing of E3b relative to E1 suggesting that there may still be residual negative migration. The objective of Phase 6 is to improve Kb/Kd with ** new input information related to the organizational processing process** and ** the restricted H2 private capacity**, instead of continuing to search for generic MTL optimizations on the same 9-mer features.

Only `data/mousePMHC/mousePMHC_train.csv.gz` and existing Train-only OOF files are readable at this stage. Fixed test is banned throughout the development and selection filter. All performance conclusions are Train-only OOF; they cannot be described as independent test performance.

## Evidence available

E1 shared E3b single (3-seed)
|---|---:|---:|---:|---:|---|
Db 0.74771 0.8570 0.8968 0.8946 To maintain the global backbone and set up a protective line
Kb0.7690.69690.7230.733 absolute signal weakness; priority added to the profile
Kd0.76480.77550076620.79770 Possibilities for transport; test for binding
Kk0.8870.81000.798800.8292 is not used as a target for this phase

E15 The IEs relative to the original average value of the 3-seed E3b single model are estimated at ZXQ0QZ/`+0.0308`, higher than Db. Therefore, the common co-factors are not added to seed; hesmatic information that reduces the error of the Kb/Kd system should be sought.

Directions that have been stopped and not retested: H2 hard grouping, fixed global/H2 soft foundation, CNN encoder, tessue/H2 auxiliary containment, catch running loss, PCGrad/CAGrad/Rotograd, tier interposition head, continued addition of E15 co-editor seeds.

## 3. General agreement and promotion rules

- The scope of the task is fixed at 24 tissues x H2 task; 3-old plain-grouped OOF, plain seen ZXQ1QZ.
- The first round of candidates uses seeds ZXQ0QZ; only the only structure of the first round of gate allows for the pre-assigned seeds ZXQ1QZ.
- Main indicators: mean task AUROC. Protection indicators: mean task AUPRC, work-6 AUROC, H2 macro AUROC, each task AUROC, Brier score.
- In addition to E25, which is explicitly stated as diagnostic, each new candidate must be compared to the read, and the old E3b, and not only to E15, which is integrated.
- III Seed promotion:

\[
\Delta\mathrm{AUROC}_{\mathrm{macro}}\ge 0.0030,\quad
\Delta\mathrm{AUPRC}_{\mathrm{macro}}\ge -0.0010,\quad
\Delta\mathrm{worst6}\ge -0.0030.
\]

Also, requests: any Db, Kk group should not exceed ZXQ0QZ; Kb or Kd at least one increase `0.0120`; Kb/Kd total at least 6/9 task improvement; task-paired bootststrap AUROC 95% below the `-0.0010`. Each structure allows only one predefined hyperparametric configuration; the hyperparametric melts only once in its corresponding experiment using prefixed rules.

## 4. Experimental sequences

Number,  Candidate,  Main hypothesis,  Actions following the results
|---|---|---|---|
E25 Kb/Kd data and model achievement Low score from label conflict, data dilution, Seed deviation or specific motif/samples  Diagnostic only, not model winner
E26  E3b + source-protein flank/position brach N/Cflank, protein position to fill tissue processing signal  gate only enters E29
E27 E3b + Kd zero-init low-rank apperter Kd can be repaired by small private debris without harming Db/Kk  satisfied date to enter E29
E28  E3b + Kb/Kd zero-init low-rank adjustments  while private fixes two low scores H2 may exceed single Kd apperter  by direct comparison with E27 and keep at most one
E29  E26 and E27/E28 freeze combination  processing information and private capacity complement  only when both father candidates pass
E30  Fixed H2-conventioned probability ensemble specialist only in Kb/Kd with fixed weight E15  only allowed pre-registered weights of 0.25; task-wise not allowed transfer
E315-sead communication  to verify the yield of the only 3-bed winner

### E25: Data and disability audit

E25 Read-only Train and E15 OOF forecasts, no training models, no reading of fixed test. The output must be: pair label integrity, task/H2 samples and UniProt overlay, conflict with task peptide labels, AUROC/ AUPRC of E15 with pairs of margin, five seed, amino acid concentration at H2 and Kb/Kd list of difficult tasks. The purpose of this is to exclude data construction problems and to determine whether E26's flank extraction is feasible; the tabk special model should not be created based on the observations of a task.

### E26:flank/position processing branch

Extracts the target from the recorded plain UniProt sequence from 10 aa, N/C end distance and relative position of each of the target points up and down. Enter E3b 9-mer trunks with new branch:

\[
h = h_{\mathrm{E3b}} + \alpha\,h_{\mathrm{flank}}(N_{10},C_{10},\mathrm{position},e_{\mathrm{tissue}}),
\]

ZXQ0QZ) is initialized. If the parent sequence cannot be uniquely mapped, the sample is not allowed to delete the samples. The width of branch is fixed at 32, and droopout at 0.2, freezing E3b's main training 8 epochs, then unfreezing the last linear layer of training for encoder/experts 8 epochs. No additional search of the flank length or training wheels is allowed.

The external pMHC fractions can only be used as exploratory diagnostics and cannot be directly characteristic of the E26 main result unless there is no overlap between the training data and the current IEDB records.

### E27/E28: Zero initialization low H2 apperter

Aadapter is located after E3b peptide encoder before:

\[
h'=h+\alpha_{a}U_a\sigma(V_a h),\qquad a\in\{\mathrm{Kd},\mathrm{Kb}\}.
\]

ZXQ0QZ, \(\alpha_a=0\) Initialization, zero initialization of the apperter output layer, 10 times the main dry weight parameter, and 10 times the weight decay. Phase I freezes embedding, encoder, etc., training only on apperter and task heads 8 epochs; Phase II unfreezes the final linear layers 8 epochs and add L2-SP anchor to the initial weight of E3b. E27 only uses Kd; E28 simultaneously activate Kb/Kd. They are not hard grouping, nor add to the tabk private intervention parameters.

### E29/E30/E31: Combining, integration and cessation

E29 Only combines E26 and E27/E28, which have been independently adopted; structures, training sequences and hyper-parameters equal to the fixed combination of their father ' s candidates. E30 will only be implemented when E29 or aadapter is passed and its Kb/Kd OOF score is not fully connected to E15. It is E15 for Db/Kk, and Kb/Kd is fixed:

\[
s=0.75s_{\mathrm{E15}}+0.25s_{\mathrm{specialist}}.
\]

No weight, H2 range or task weight of E30 may be re-selected by the same OOF. E31 performs the five seeds and so on for the only winner; freezes the Phase 6. Fixed test only allows one-time confirmation of the frozen E31 winner and E15, and cannot be re-selected by the test structure,seed or weight.

## 5. Expectations and explanations

If Kb raises ZXQ0QZ, Kd increases `0.02` and the rest of the tasks remain unchanged, the macro is expected to increase on average:

\[
\frac{4\times0.03+5\times0.02}{24}=0.00917,
\]

ZXQ0QZ for E15 can reach about `0.8484`. This is a scenario calculation for quantitative targets, not a commitment to success. If E25 shows that Kb/Kd errors are mainly from tag conflict or the same seed stabilization error, priority should be given to collecting more high-quality tissues - H2 immunopeptidomics data, rather than expanding models.

## Products

- E25 runner:`scripts/run_mousepmhc_phase6_e25_kb_kd_audit.py`
- E25 results:`results/mousePMHC_phase6_e25_kb_kd_audit/`
- Follow-up to each experiment: `scripts/run_mousepmhc_page6_e<numbername>.py`
- Results catalogue: `resources/mousePHC_phase6_ename/ `
