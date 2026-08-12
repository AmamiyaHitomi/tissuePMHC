# TissuePMHC Phase 2 Experimental Road Map

Baseline model: E14a (auxiliary global branch + plain HLA-special branch + fixed probability average). The best is currently updated to E29 Multi-kernel CNN E14a 5-seed ensemble: mean AUROC 0.8373, mean AUPRC 0.8259, World-10 means AUROC 0.767. E15–E27 and E29 completed; E28 Negative Correlation Learning no longer defaulted and Standard split exploring ends with pre-registration commitments.

## Preparation steps (without reference to test number)

Keeps samples-by-species projections for each branch and aligns them with ZXQ0QZ, `target_tissue`, `mhc_restriction`. Integration can only take place on fully aligned projections. Experiments using valueation/OOF must ensure that test is used only for final evaluation.

## Experimental order

Order,  Experiment number,  Experiment,  Integration/training algorithm,
|---:|---|---|---|---|
1 E15 Fixed integration rules melt: average probability, logit average, track average rank score avelaging / logit averiging / rank integration [Dieterich, 2000] (ZXQ0QZ); [Bajor et al., 2019] (https://jmlr.csail.mit.edu/papers/v20/18-094.html)
2  E16  global and HLA branches, 5, 10, 20 times averaged after the MC Dropout projection, using E15 best rules to integrate  Monte Carlo Dropout [Gal & GHHARAmani, 2016] (https://proceedings.mlr.press/v48/gal16.html)
E17  Average of 3 independent seeds in the global branch and HLA branch, then expanded to 5 seed Deep Ensemble / prepositioning  [Lakshminayanan et al., 2017] (https://arxiv.org/abs/1612.01474)
E18  Select a global weight for a valuation/ OOF forecast; more fixed 0.5 and the global weight for the validation set selection  cross-valued weighting weight [van der Laan et al., 2007] (https://pubmed.ncbi.nlm.nih.gov/17910531/)
E19  Checkpoint Esmble, snapshot E14 former training process  Checkpoint Ensemble / Snapshot Ensemble  [Chen et al., 2017] (ZXQ0QZ); [Huang et al., 2017] (ZXQ1QZ)
6  E20  Comparison of SWA with original final checkpointStochasticWeight Avelaging [Izmailov et al., 2018] (https://arxiv.org/abs/1803.05407)
E21  In E14a global auxiliary branch, the main classification BCE mission is to control the ticue/ HLA arcial loss; only the global branch is retrenched and the saved HLA plain branch forecast is repeated Gradient-Similarity Auxiliary Gating [Du et al., 2018]
E22  Use Nash bargaining weights only between main categories, tissue auxiliary, HLA auxiliary; update weights every 10-20 catch, rest catch repeats recent solver of Periodic Nash-MTL [Navon et al., ICML 2022] (https://proceedings.mlr.press/v162/navon22a.html)
9 E23  For the E14a global auxiliary branch periodic fork different auxiliary-weight training path to select and merge valid updates with a pair-grouped error task, and compare the fixed auxiliary weights to the ForkMerge [Jiang et al., NeurIPS 2023] (https://proceedings.neurips.cc/paper_files/paper/2023/hash/60f9118a849e8e9a0c67e2a36ad80ebf-Abstract-Conference.html)
10  E24  Use the train-grouped valuation main task as meta-loss, dynamic learning tissue/ HLA auxiliary weight; main classification weight is fixed to 1 to avoid sacrificing the target task in exchange for [Liu et al., TMLR 2022] (https://openreview.net/forum?id=KKeCMim5VN)
11 E25  Construction HLA-Structured PLE: set a small number of global share experts and lightweight private expert by HLA route; gate sharing and corresponding HLA for each Tissue-HLA task; prohibit copying complete copies of 44 tabs per tablet  Progress Layed Extration [Tang et al., RecSys 2020] (https://doi.org/10.1145/3383313.3412236)
12  E26  Builds a candidate model library and selects members by using value/OF mean task AUROC; checkpoint esmble, snapshot essemble and SWA  Greeny Ensemble Selection [Caruana et al., 2004] (https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf)
13  E27  Train fixed L2 Rression 2-level models on the task-rank features of the E26 candidate library; use OOF fractional training, independent test evaluation only once  Super Learner/ stacked gender representation [Wolpert, 1992] (ZXQ0QZ) 80023-1); [van der Laan et al., 2007] (ZQ1QXZ)
14  E28  Joint training branch with HLA and add negative member-related error items to the main task loss; comparison with independent E14a  Negative Correlation Learning [Liu & Yao, 1999] (https://www.sciencedirect.com/science/article/pii/S0893608099000738)
15  E29  Global Auxiliary + HLA plain and task-rank Fusion for E14a, replacing Flatten-MLP peptide encoder only with a multi-kernel CNN for reserved location information; 1-/3-seed OF and pre-registered 5-Seed OF were adopted, 5-Seed test became final Standard-split best  Multi-kernel CNN Representation Diversity [Kim, 2014] (https://aclanthology.org/D14-1181/)

## Execute Order

```text
E15 → E16 → E17 → E18 → E19 → E20 → E21 → E22 → E23 → E24 → E25 → E26 → E27
                                                                            ↓
                                                     E29 1-seed OOF (through)  E29 3-seed OOF/test (through)  Pre-registration E29 5-seed OOF/test (through)
                                                                                                                ↓
                                                                                         Standard split end;turning to generalized validation
```

E15–E27 and E29 have been completed, and the number, code and result directory remains unchanged. E28 is not running, and E29 has been successfully provided as a sign of diversity and is no longer used as a default follow-up experiment. E295-seed is the last pre-registered standard split extension, which is now complete.

## E21-E25: completed secondary weights and structural routes

E21–E24 Modify and retrain E14a global auxiliary branch, fixed to repeat the same seed with the existing HLA plain branch forecast. This separates the global branch algorithm changes and avoids training the HLA branch. The three macro-objectives they use are defined as:

```text
Main task: 44 tissue-HLA headers overall BCE
Support Task 1: taskue
Support Task 2: HLA representation
```

E22 Nash-MTL only handles three macros, prohibiting extension to 44-task-to-task-to-task Nash bargaining. The weights are updated every 10-20 catch, the rest of the catch is recently released; if formal experiments are estimated to exceed the budget, the interval is increased.

E23 ForkMerge and E24 Auto-Lambda must use the inter-grouped valuation in the train, test not to participate in fork merge, meta-weight update or hyperparameter selection. Auto-Lambda 's meta-objective uses only the main classification loss, the main task weight is fixed at 1.

E25 is a structural experiment that trains a HLA-Structured PLE model without reusing E14a global level. To control costs, two narrow global experts and one HLA-private expert per sample are defaulted; no full 44 packages of privatists are allowed. E25 is compared to the E14a branch baseline as a single model, and integration with existing HLA/plan or E17 members is considered only if the single model is competitive.

The minimum conditions for entering the next stage are expected to be 3-bed.
|---:|---|---:|---|
1  E21 gravity-similarity auxiliary gaating  about 1.3–1.8 hours relative to seed E14a/E15 means AUROC does not fall; gating cannot fully or completely turn off for long
2  E22 periodic Nash-MTL approximately 1.7–2.3 hours  bargaining weights limited and stable; none of the three macro-targets have been permanently suppressed; there is no apparent deterioration of the relative fixed weight baseline
E23 ForkMrage  2.5–4 hours  valuation merge stabilization filter section auxiliary update; relative fixed weight E14a has positive pairing gain
E24 Auto-Lambda  2.5–4.5 hours  meta-weight limited and interpretable changes as training takes; main task valueation/test not below the fixed weight baseline
E25 HLA-Structured PLE  about 3-5 hours

The time estimate is based on the E14-E20 operating record and is not a guarantee given in the paper.

```text
1. Run 1 Seed smoke/screen and record actual extra time and diagnostics for algorithms.
2. Only if the corresponding seed baseline is not significantly degraded will it be able to run the full 3 seeds.
3. Compared first with E14a/E15 pair 3-seed results.
4. Only if a steady gain is achieved will it be extended to 5 seeds and challenge the 0.8263 AUROC of E17.
```

## E26-E27: Strict OOF integration completed

E26 has generated 3-old plain-grouped OOF forecasts and independent full-training test projections, with three lists of E14 final, three lists of E16 MC-20 seed, and two types of 3-seed mean. Of value, selection is only ZXQ0QZ. Its test means AUROC of 0.8246, slightly higher than E17 3-seed, but below E17 5-seed 0.8263.

E27 Reuse the same OOF/test pool to select a task-rank to enter L2 Logist Repression, which has a meaning AUROC of 0.8243, not exceeding E26 or E17. E14 final 3-seed means OOF task-rank integration, which means MC-20 3-seed, is 0.9987, which explains the failure mainly because of candidate homogeneity rather than the lack of a more complex blender.

## E29: Next priority route - Multi-kernel CNN E14a

E14a The current peptide encoder is amino-acid embedding, Platten and two layers of MLP. E29 Keeps the validated global auxiliary branch, plain HLA-special Branch and task-rank fasion, replacing only the peptide encoder to reduce the relevance of E14 members while controlling the scope of structural changes.

Fixed first round configuration: embedding dimension 16; kernel size 2, 3 and 5; 32 volumes per volume branch; recoupling and projecting to Hidden dimension 128; dropout 0.2; 25 epochs; AdamW; leaving ratter 0.001; weight decay 0.0001. It is prohibited to discard anchor information only on 9-mer with global max pouring.

E29 saw 202,60074, 3-old plain-grouped OOF screenen completed and unread test. The CNN single model OOF means AUROC 0.807, higher than the match E14 Seed 0.7915, and the match with E14 took-rank corration 0.8727. After integrating with E14 3-seed mean rank, OFOL means AUROC from 0.8042 to 0.8097, and World-10 means AUROC from 0.7257 to 0.7314, all four conditions were adopted.

Pre-established conditions of passage are:

```text
1. E14 OF mean AUROC relative to the ED saw no more than 0.005;
2. OOF task-rank corration with E14 is less than 0.97;
3. E14/E29 etc. task-rank integration relative to E14 OFF means AUROC at least up to 0.001;
4. World-10 means OOF AUROC is down to no more than 0.001.
```

E29 3-seed Phases have been completed. 3-seed CNN OOF means AUROC 0.8042 for 0.8138 above E14 3-seed; relevance 0.9428; etc. rink integration is equal to E14 OOF uplift 0.0090 AUROC, all four conditions are again adopted. Then a test evaluation of the fixed model is conducted, E29 3-seed meant the meaning AUROC 0.8341, mean AURPC 0.828, World-10 means AUROC 0.76334, exceeding E17 5-seed 0.8230.08139/03573.

Before the new training, the project pre-registered its last 5-side incremental expansion: training only for 2026007, 2026008, the first three seed; 5-seed OOF must be adopted relative to 3-seed to satisfy AUROC gains at least 0.010, work-10 AUROC gains at least 0.0000, AUPRC gains at no less than 0.0000, AUROC gains at no less than 0.000010, AUROC gains at no less than 0.0000.070, resulting in increases of 0.000311, World-10 + 0.00298, AUPRC+00242, all three. A fixed 5-seed test assessment to the mean AUROC 0.873, mean AUPRC at 0.8259, World-10 gains at 0.0070, compared to E29 3-seed increases of 00316, 00315, and 00359. E29 outputs follow the long form of ZXQ0XZ.

** Decision-making:** E29 5-seed means that the final standard standard outcome. E14/E29 inter alia integrates the AUROC on 3-seed test, which is slightly less than the E29 separate result and therefore does not replace E29; no weighting or selection of members may be made based on observed test.

As the last confirmed extension of the standard split, the E295-seed incremental experiment has been completed as pre-registered. Only new seeds 20607/2076008 has been added and the existing OOF/test forecast 20260704–20660706 has been reused; all three OOF date have been passed before generating the new test forecast. For full rules and results see `E29_5SEED_PREREGISTRATION.md`, the first key script is `scripts/run_tissuepmhc_e29_incremental_5seed.py`. This time, the standard split interface has been stopped.

## E28: Back-up route

E28 is no longer the default next step after E27. The non-introduction of new input information may create differences at the expense of the ranking of members themselves, and the weight of the coefficient involved requires additional options. Previously, E28 was considered only when E29 failed; now E29 has been successful on OOF and test, so E28 stops default execution and retains the 1-seed OFX supplemental experiment that fixes the individual relevance weights only when the paper requires complete method coverage.

## Reference method check

The following entries give the methodological source for E21-E29 in the road map. Uncertative Weighting and Alining-MTL have been removed from the current official route.

1. **E21 — Gradient-Similarity Auxiliary Gating**  
   Du, Y., Czarnecki, W. M., Jayakumar, S. M., Farajtabar, M., Pascanu, R., & Lakshminarayanan, B. *Adapting Auxiliary Losses Using Gradient Similarity*. arXiv:1812.02224, 2018.  
   https://arxiv.org/abs/1812.02224

2. **E22 — Nash-MTL**  
   Navon, A., Shamsian, A., Achituve, I., Maron, H., Kawaguchi, K., Chechik, G., & Fetaya, E. *Multi-Task Learning as a Bargaining Game*. Proceedings of the 39th International Conference on Machine Learning, PMLR 162, pp. 16428–16446, 2022.  
   https://proceedings.mlr.press/v162/navon22a.html

3. **E23 — ForkMerge**  
   Jiang, J., Chen, B., Pan, J., Wang, X., Liu, D., Jiang, J., & Long, M. *ForkMerge: Mitigating Negative Transfer in Auxiliary-Task Learning*. Advances in Neural Information Processing Systems 36, 2023.  
   https://proceedings.neurips.cc/paper_files/paper/2023/hash/60f9118a849e8e9a0c67e2a36ad80ebf-Abstract-Conference.html

4. **E24 — Auto-Lambda**  
   Liu, S., James, S., Davison, A. J., & Johns, E. *Auto-Lambda: Disentangling Dynamic Task Relationships*. Transactions on Machine Learning Research, 2022.  
   https://openreview.net/forum?id=KKeCMim5VN

5. **E25 — Progressive Layered Extraction (PLE)**  
   Tang, H., Liu, J., Zhao, M., & Gong, X. *Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations*. Proceedings of the 14th ACM Conference on Recommender Systems, pp. 269–278, 2020. DOI: 10.1145/3383313.3412236.  
   https://doi.org/10.1145/3383313.3412236

6. **E26 — Greedy Ensemble Selection**  
   Caruana, R., Niculescu-Mizil, A., Crew, G., & Ksikes, A. *Ensemble Selection from Libraries of Models*. Proceedings of ICML, 2004.  
   https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf

7. **E27 — Stacked Generalization / Super Learner**  
   Wolpert, D. H. *Stacked Generalization*. Neural Networks, 5(2), 241–259, 1992. DOI: 10.1016/S0893-6080(05)80023-1.  
   van der Laan, M. J., Polley, E. C., & Hubbard, A. E. *Super Learner*. Statistical Applications in Genetics and Molecular Biology, 6(1), 2007.  
   https://doi.org/10.1016/S0893-6080(05)80023-1  
   https://pubmed.ncbi.nlm.nih.gov/17910531/

8. **E28 — Negative Correlation Learning**  
   Liu, Y., & Yao, X. *Ensemble Learning via Negative Correlation*. Neural Networks, 12(10), 1399–1404, 1999.  
   https://www.sciencedirect.com/science/article/pii/S0893608099000738

9. **E29 — Multi-kernel CNN Peptide Encoder**  
   Kim, Y. *Convolental Neural Networks for Unity Transportation*. Issues of EMNLP, pp. 1746-1751, 2014.E29 uses a multi-dimensional volume to extract basic ideas for local sequence models, but retains post-volume position information for fixed 9-mer.
   https://aclanthology.org/D14-1181/
