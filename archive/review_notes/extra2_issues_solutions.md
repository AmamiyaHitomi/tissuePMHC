# Thesis problem solution

> Collapse against `issues_checklist.md` and the current `tissuePMHC_latex_v3`.
> State Meaning: ** Essentially resolved ** Current text is sufficient; ** Needed to enhance ** Existing ** may nevertheless trigger the same doubt; ** Unsolved ** = Need for additional text, graphs, verification or experimentation.

## I. Proposed first four changes

1. ** Add a unified table of assessment agreements** to replace the definitions currently dispersed in the body.
2. ** Addressing the unknown origin of 44-task**: Complete definition if building rules can be restored; not moving to supplementary material and clearly marked as historical exploratory results.
3. ** Add data overlap mapping or audit tables**, clearly distinguishing the separation of the four dimensions of pair, peptide, protein and task.
4. ** Positioning Table 7 as "Emergency Component Evidence", not as a strict decomposition experiment**; real decomposition can only be achieved using the same tasks, old, Seed and training budgets.

---

## II. Project-by-project solutions

### 1. Over-technicalization of the whole text

- Current status:** Need to be strengthened**.
- The current version has included a number of restrictive statements, but a large number of abbreviations and protocol names are still concentrated in Methods and Reults.
- Solutions:
  1. Each technical paragraph is organized in three sentences, "Definition - Purpose - Permissible Conclusions".
  2. For the first time, OOF, Pair-disjoint, macheted and conned component first wrote a popular explanation and then gave the technical name.
  3. To avoid merely writing "strict" "standard" and to spell out its biological meaning at the same time:
     - Standard: The same complete pair does not cross the assembly, but peptide can be reappeared through other pair/task;
     - peptide-disjoint: Test peptide does not appear in any training task;
     - Both test only those who have appeared in training.
  4. Add a sentence at the end of each RQ, "Biologic internationalisation" or natural paragraph of the equivalent, not just repeat the indicator.

Directly applicable:

> In plain terms, the standard split asks whether the model can rank new matched pairs within previously represented tissue--MHC tasks, even when an individual peptide may have been observed elsewhere in the training data. The peptide-disjoint split asks the harder question of whether the same tasks can be predicted for peptide sequences that are absent from all fitting folds.

### 2. Lack of organizational specificity

- Current status:** Partially resolved, suggested that a new paragraph** be added.
- Current Introduction has mentioned protein access, turnover, proteolysis, processing and tissue processing, but it can also describe the complete processing chain more clearly.
- Solutions:
  - Add a paragraph after paragraph 1 of Introduction that explicitly refers to protein enzyme cutting, TAP trans-shipment, ERAP trim, MHC loading and tissue differences in the associated genes.
  - The results of the model should not be interpreted as direct evidence of a molecular mechanism, as the model does not use expressions, protein groups or process route characteristics.
  - The most important paragraph in paper (map: antigen delivery organization-specificly preferred molecular mechanism)

Drafts in English that can be directly modified:

> Tissue-associated presentation can arise at several stages upstream of peptide--MHC binding. Differences in source-protein abundance and turnover determine which substrates are available, while proteasomal cleavage, TAP-mediated transport, aminopeptidase trimming, and peptide loading determine which fragments reach stable MHC-I complexes. The genes and cell types contributing to these processes vary among tissues, so the same protein may yield different presented peptides in different organs, and the same peptide may have different presentation evidence across tissue contexts. Characterizing these differences may improve tumor-antigen prioritization and help distinguish broadly presented targets from tissue-restricted candidates. Our sequence-only benchmark captures the resulting tissue-associated signal but does not identify its causal molecular source.

### 3. Lack of clarity regarding the overlap of the Mouse mixed test entity

- Current status:** Value audit resolved and presentation needs to be enhanced**.
- The current text is available as follows:
  - pair ID overlap:0;
  - unique peptide overlap:81.47%;
  - unique parent-protein overlap:88.88%;
  - row-level peptide/protein overlap:84.31%/93.04%.
- Solutions:
  - Replace the text with a four-line audit form:

Can you claim that you are not allowed to use the platinum?
|---|---|---:|---|
Pair ID  does not allow cross-group   can claim unseen pair
Peptide identity  Allows re-emerging by other pair/task  81.47% (unique)  Not claimed unseen peptide
Parent protein  Allows the re-emergence of 88.88% (unique)  Not claimed unseen protein
Tisse-H2 task  Training and testing have appeared 100% seen tasks  Unseen task

- Add a small diagram: the same test peptide can be used as a test sample in task A, as a training sample in task B, but the same pair ID cannot be assembled across the same group.

### 4. Assessment of confusion of the title of the agreement

- Current status:** The text is explained, but the summary table is lacking and still needs to be strengthened**.
- It is proposed to add the following protocol table to `Problem Formulation and Benchmark`:

Suggested uniform name  data source  division unit  test peptide for  Task state  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use  use
|---|---|---|---|---|---|
Standard fixed test  standalone  pair ID pre-freeted, via other pan/task
Standard plain-grouped OOF
Matched standard OOF plating pair pool pair ID; held-out size matching strict
Peptide-disjoint OOF  the same pair pool as matchd standid peptide Connected Conponent  seen task unseen-peptide

- Naming rules:
  - "frozen 5-seed" is a model/integral attribute and does not belong to the split name and should be placed in the Mode or Seeds columns.
  - "paired task-level standard-versus-standard statistics" is a statistical analysis and should not be listed alongside data disaggregation.
  - "stric" may be a acronym, but the full name must be written for the first appearance.

### The question of "three settings only, but with more names"

- Current status:** to be resolved by hierarchical naming**.
- Solutions:
  - The paper clearly contains only three conceptual aspects:
    1. standard fixed test;
    2. standard OOF;
    3. peptide-disjoint OOF.
  - Matched standard OOF is a matching version of the standard OOF.
  - Frozen, 3-seed/5-seed are the attributes of model assessment.
  - The analysis is the method.
  - Not all of these properties are called "evaluation settings".

It is suggested that the existing sentence should read:

> We distinguish three evaluation settings throughout the paper.

was replaced by:

> We use three split families: a frozen standard test, standard pair-grouped OOF, and globally peptide-disjoint OOF. "Matched" denotes a standard OOF control constructed on the same pair pool and with held-out sizes aligned to the peptide-disjoint folds; seed counts describe model ensembling rather than additional split types.

### 6. Matched fair-grouped OOF

- Current status:** largely resolved, with additional operational description proposed**.
- The current Methods has specified that match control uses the same pair pool and matches the old number, task coverage and task-wise help-out counts.
- It should also be added:
  - No stric test samples are screened to create performance differences;
  - (b) The test labels were not used to match;
  - The number of help-out plains for each task/old;
  - The only change in nature between standing and strict is whether the peptide can cross the old.
- Recommended formulation:

> The matched standard control does not modify labels or select examples based on model performance. It reassigns the same complete pair pool by pair ID, using the same number of folds and nearly identical task-by-fold held-out counts as the peptide-component split. The intended contrast therefore changes the peptide-separation constraint while holding the evaluated pair inventory and task coverage fixed.

### How and why did the match take place in Table 8

- Current status: **matched stated; position of species across species is still optimized**.
- Solutions:
  - The human and mouse matrix-vesus-stric is reasonably placed in RQ4 "Generation Boundarias", as RQ answers the same question about the recovery of species across species.
  - Before the table, state clearly:

    > Mouse is included here only to test whether the same peptide-overlap effect is reproduced under the corresponding mouse benchmark; this is not a direct comparison of human and mouse model quality.

  - The complete match-versus-stric in RQ5 is not repeated, citing RQ4 only and reporting centrally the results of the layer of mouse-specific fix, H2 and tissue.

### 8. Pair-disjoint, task-level peptide-disjoint and global peptide-disjoint

- Current status: ** Current v3 definition is substantially correct**.
- Final uniform use:
  - @Ambassy: @Ambassin: @Ambassin: @Ambassin: @Ambassin: @Ambassin: @Ambassin: @Ambassin: @Ambassin: @Ambassin: @Ambassin: @Ambam: @Ambambak: @Ambambak: @Ambak: @Ambsa: @Ambam: @Ambam: @Ambam: @Ambam: @Ambam_Abdam: @Ambambak: @Ambdam: @Abdbak: @Ambdam @AbdbH: @Abdd @Adddddddddd @Addddd: @Addddddddds
  - (b) task-local peptide-disjoint: peptide is not available in the current training centre in task, but may appear in other tabs;
  - Globally peptide-disjoint: peptide does not appear in any of the bits of the task;
  - The grouping algorithm is the global peptide-disjoint.
- If the task-local paper-disjoint is not actually calculated in the full text, it should not be reported separately for it, but only in the concept note.

### 9. Fixed test, matched OOF, peptide-disjoint OOF and train-pool OOF

- Current status: ** A diagram or table is required to be clear**.
- The following tiers are suggested:

```text
Original benchmark
├─ Frozen fixed test
for main closed-task communication only
└─ Complete training pair pool
   ├─ Matched standard pair-grouped OOF
   └─ Connected-component peptide-disjoint OOF
```

- Only the latter two come from the same pair pool, so only the difference between the two can be used to estimate the impact of the peptide classification.
- The difference between the cross-test and the stric OOF can only be described and cannot be attributed directly to the peptide overlap.

### 10. 44-task subset unknown source

- Current status:** Not really addressed, is one of the most clear gaps at present**.
- Solutions are prioritized:
  1. Find old splitmanifest, task list and data filter scripts, restore:
     - 44 complete list of tabs;
     - Data version and download date;
     - Minimum pair threshold;
     - split seed;
     - Number of trains/tests per task.
  2. Place the above information in a supplementary table and write a definition in the main text.
  3. If recovery is not reliable:
     - Mark 44-task results as histological exploratory screen;
     - Move to Summer;
     - Main text RQ1 is replaced by 157-task share-head/logistic baseline with full provance;
     - 44-task results are no longer considered core evidence or directly compared to 157-task results.
- Do not just write "original 44-task subset" which does not answer why it has 44 tasks.

### 11. What does Pair-grouped mean?

- Current status:** Essentially resolved, but suggested local definition**.
- Suggested definitions:

> Pair-grouped means that the two rows belonging to one matched positive–pseudo-negative pair share a persistent pair ID and are always assigned to the same fitting or held-out partition. This prevents direct pair splitting but does not prevent either peptide from reappearing through another pair or task.

- If the code is actually grouped by `pair_id`, it should be asserted in the supplementary submission that:
  - Two lines per pair;
  - (a) Label is positive or negative;
  - pair with the same tissue, MHC and parent UniProt;
  - The pair ID crosses the old overlap to zero.

### 12. H2 What is it?

- Current status:** Not fully explained where it first appeared.**
- Solutions:
  - For the first time, substitute

    > the mouse major histocompatibility complex (H-2; written as H2 in the processed task identifiers)

  - The title of RQ5 could read:

    > RQ5: Replication in Mouse Tissue–H2 Tasks

  - The data-processing component describes ZXQ0QZ, ZXQ1QZ, `H2-Kd` and `H2-Kk` as the four retained remissions.

### 13. How many tissues are there in Mouse, and should the report be complete?

- Current status: ** total number of tissue resolved and complete results need to be supplemented**
- The current benchmark table has given 13 tissues, 24 tasks, 4H2 remissions.
- Solutions:
  - The main text retains the high-low-end description table, but makes it clear that it is not the complete result.
  - Supplyary Add 13 complete tables showing:
    - number of tasks;
    - (a) the number of pairs;
    - mean/median AUROC;
    - mean AUPRC;
    - PairAcc;
    - Minimum/maximum task AUROC;
    - If available, task-bootstrap CI.
  - For only one tab tag marked as descriptive, do not calculate the difference over the tab.

### 14. Definition of PairAcc

- Current status:** largely resolved, but one needs harmonization**.
- The current benchmark section is defined as a positive score greater than a negative score; the Executive Setup also describes Tie's master analytical record 0 and reports on the half-credit representation.
- Solutions:
  - After the formula is added directly:

    > The primary analysis assigns zero credit to ties; a half-credit tie rule is reported as a sensitivity analysis.

  - Avoiding the writing of other reports as "0.5 points" without indicating that this is the sensitization rule.
  - Full text harmonization PairAcc case.

### 15. Meaning of the MHz-only and conflict label

- Current status:** Model input explained, but conflict label audit should still be visible**.
- The current v3 has been described as the MHC-only rating form ZXQ0QZ, m)\, and removes the Tissue Identity and Tissue-Specific Head.
- Solutions:
  1. Do not simply merge different data on the tissue and re-establish it as a single label; keep original rows and task-special labels.
  2. The model does not see the Tissue, but the same peptide-MHC can have different labels in different Tissuerows, which is precisely the difficulty of the negative contrast.
  3. Add an audit form:
     - Number of unique peptide-MHC query;
     - Whether the same query has a positive or negative label;
     - Conflicting query ratio;
     - Conflicting rows ratio.
  4. Clarify the role of MHz-only: Test the extent to which a universal peptide-MHC signal is available when it is not used; it is not an absolute presentation data set without noise.

### 16. Definition of Connected-Consent peptide-disjoint

- Current status:** As stated in principle, the details of the algorithm can still be strengthened**.
- Solutions:
  - Give a precise definition:
    1. Each matchd fair is a node;
    2. Connects to a side if two pair share any of the peptide division;
    3. No viewed components;
    4. The whole component is assigned to the same old;
    5. So any peptide and its passing connection will not cross the little/held-out.
  - If the current code is actually implemented using two maps, "peptide is node, pair is side " , the text should be adjusted to the true reality, but the same conditions are ultimately guaranteed.
  - Reported number of coponents, maximum coponents, number of old pairs, number of task help-outs and zero overlap audits.

### 17. The master results sheet approach is too few.

- Current status:** Partially resolved**.
- Solutions:
  - Do not mix different task inventory for the purpose of increasing the number of lines.
  - Only the same 157 tabs, the same split, and the full-sized help-out scenarios are shown in the master list.
  - 44-task history method has a separate table of histological baselines.
  - External method to display the general-pMHC control table.
  - The method of strict anarchitecture is the IDENTic-old list.
  - Equity is guaranteed through the tabulations, rather than placing all methods in a high ranking.
- To supplement the experiment, priority:
  1. 157-task one-hot logistic regression;
  2. 157-task BLOSUM62 random forest;
  3. 157-task shared heads;
  4. 157-task auxiliary dual branch;
  5. TissuePMHC;
  6. Recoverable external controls.

### 18. Figure 2 Not sufficiently sophisticated and too few methods

- Current status:** The original box chart problem still requires visual re-doing or replacement of the map**.
- Recommended programmes:
  1. If the methods are identical 157-task prections, use the rainclud/ violin + box + task dots.
  2. If there is no consistent task-by-task result, do not produce falsely comparable boxpl; switch to group point maps that mean, median, World-Tail.
  3. The contrast is more suitable for using task-wise painted forest maps or margin maps.
  4. Uniform:
     - (a) Colour-blind and friendly colouring;
     - Horizontal approach labels;
     - 95% task-bootstrap CI;
     - Same y-axis range;
     - The text and the chart are identical in the methodological order.
  5. The figure must indicate the meaning of red dots, boxes, whiskers and task aggregation.

### 19. Table 7 is a public test

- Current status:** Currently renamed Conponent event, but there is still a need to prevent overexplaining**
- Solutions:
  - The main text states clearly:

    > These comparisons summarize completed historical component evidence and are not a single factorial ablation study because task inventories and aggregation settings differ.

  - Table 7 does not use "Ablation" as a title.
  - If formalization is to be achieved, it should be fixed:
    - 157 tasks;
    - Same standard or stricold manifest;
    - The same three.
    - The same epoch/batch size/optimizer;
    - The same as the Esmble rule;
    - Only one component is changed at a time.
  - Minimum formal melting matrix:

Multi-kernel encoderAuxiliary lossDualbranchRank foundation
|---|---:|---:|---:|---:|---:|
Base MLP  No  No  No
Dualbranch No  No
auxiliary  No
multi-kernel
Probability Fuse is a  Yes No

### 20. Mouse tabular methods and indicators are too few

- Current status:** Indicators have improved and methodologies are still limited**.
- Solutions:
  - The standard table contains at least BLOSUM62 RF, shared headers, Factorized MMOE and completed H2-Kk apperter.
  - Only the method that is completed on the original list.
  - Indicators are harmonized: Mean AUROC, Median AUROC, AUPRC, PairAcc, World-6, World task.
  - Accuracy/MCC can be placed in the applicationary to avoid over-broadness of the master form.
  - Do not fill in the list of stact forms without running on the same flod.

### 21. Human does not match Mouse system

- Current status:** Problems exist objectively, but can be resolved by interpretation and by way of a second option from the supplementary experiment**.
- Reason:
  - The final structure of the Human and Mouse is different;
  - Data size, number of tasks and development history are also different;
  - So two strict tables are not multi-species model rankings.
- Minimum cost solutions:
  - They are clearly stated before both tables as "within-species controls", and do not require a matching method.
  - Keep the limit in Discussion that "not identifiable " species effect.
- More powerful solutions:
  - (a) Plain MLP dul brach, auxiliary dul brach;
  - The factorized MMOE for the human to make up for the mouse;
  - Use the same standard old manifest.
- If the calculation is limited, it is recommended that the lowest cost formula be used and that the value not completed or not be fairly added for the sake of form.

---

## III. ISSUES THAT DON'T RELATIONATE

The following changes can be accomplished directly:

- Introduction Biological Mechanisms Section;
- Plain-language rewrite;
- (a) An assessment of the agreement matrix;
- fixed/OOF/stract relationship diagrams;
- H2, PairAcc, Pair-grouped, Conned-component definitions;
- Mouse overlap audit form;
- Matched construction description;
- Table 7 name change and downgrade of conclusions;
- Interpretation of the human/mouse method is inconsistent;
- Moves the complete result to the Supplierary (provided that the existing check results are sufficient).

## IV. Which issues require first verification of codes or data

- Source of 44-task, filter rules and split manifest;
- Number of peptide-MHC consulting Labors in MHz;
- Old overlap for the air ID, peptide and protein;
- Real node/side realization of the conned-component map;
- Whether each table is mixed with the same test, standard OOF or stric OOF;
- Whether the methods in Figure 2 really have the same task forecast and task-by-task projection.

## V. ISSUES THAT REQUIRE ANOTHER EXPERIENCE

- (a) Formalization of the process under fully consistent 157-task/old/seed conditions;
- The method of matching the human and the mouse block artiecture table;
- Adding the 157-task baseline that has not been completed;
- shuffled-tissue-label control;
- Protein-disjoint, study-disjoint or external cohort authentication.

The priority is:

1. 44-task verification;
2. MHC-only conflict audit;
3. Harmonizing 157-task elimination;
4. Cross-species approach completed.

## VI. Recommended structure of the final paper

1. **Introduction**
   - Biological mechanisms;
   - definition of the tissue-MHC preference;
   - Clinical significance and non-causal limitations.
2. **Related Work**
   - binding/presentation;
   - method for context-aware representation;
   - Three points different from the present mandate.
3. **Benchmark**
   - pair construction;
   - human/mouse inventory;
   - (a) a split summary of agreements;
   - overlap audit.
4. **Methods**
   - Models;
   - MHC-only and external controls;
   - Indicators, PairAcc and statistical methods.
5. **Results**
   - Harmonize the primary result of task inventory;
   - matched standard vs peptide-disjoint;
   - mouse replication;
   - no-tissue/external controls;
   - identical-fold architecture controls.
6. **Supplementary**
   - 44-task historical results;
   - (a) a complete table of tissue/task;
   - fold manifests;
   - conflict/overlap audits;
   - History
