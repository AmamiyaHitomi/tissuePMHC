# List of issues for the revision of papers

> Based on 19 review screen maps in the `pic_issues` folder.
> Each contains a review of the original, a summary of the issues and a suggested treatment. The corresponding `[ ]` can be changed to `[x]` after the changes have been completed.

## I. Overall writing and biological background

$11. Reduced technicalization of the whole text and increased popular interpretation**

  - Screenshot: `Screen Screenshot 2026-07-26 130951.png '
  - Review of the original:

    > The current formulation is too technical, and the technical language is also used in parallel with the use of plain language, which is understood by those from cross-cutting areas.
    >
    > The text of the paper as a whole is modified with the following instructions:
    >
    > We plan to put in a periodical about biological or biographical issues. The current paper is very technical, with few biological implications and interpretations. The language is technical, and the way it is modified is that people in the biological field understand the biological significance of each set.

  - Summary of issues:
    - The full text uses a large number of computer and machine learning terms, but is not sufficiently friendly for non-calculation background readers.
    - The biological significance of the experimental setting and its impact on the organization ' s specific antigen presentation is not well explained.
  - Recommendations addressed:
    - A general definition is given when a professional term first appears.
    - Add "what biological questions should this set-up answer" after each technical set-up.
    - The results are reported in conjunction with the computational and biological interpretation.

$12. Supplementing antigen-advanced molecular mechanisms in Introduction**

  - Screenshot: `Scene Screenshot 2026-07-26 131015.png '
  - Review of the original:

    > Introduction Add an explanation of the molecular mechanisms for the transmission of antigens in different organizations, including enzymes, trans-shipment, trimping, transmission, etc., which are represented by the differences in genes in different tissues. Explain that the same protein is presented in different tissues, or that the same protein may have different delivery effects in different tissues.

  - Summary of issues:
    - The introduction lacks biological mechanisms for tissue-specific antigen processing and delivery.
    - The significance of the study for oncological immunotherapy has not been fully explained.
  - Recommendations addressed:
    - Supplements tissue differences in antigenase cutting, trans-shipment, trimping and transmission of the genes.
    - Descriptions of the same protein or the same epitope may produce different presentation results in different organizations.
    - Linking this background to oncological immunisation treatment, target selection and target removal risk.

## II. ASSESSMENT AGREEMENTS, DATA DATA DATA AND POSSIBLE SILLIES

$13. Show more clearly the overlap of entities in the mode mix test**

  - Screenshot: `Screen Screenshot 2026-07-26 131046.png '
  - Review of the original:

    > This one is not clear enough to see if you can draw a picture or list.
    >
    > In the test of the fit, the string may appear in other tasks, but not in the organizational-synchronous task we are looking at.

  - Summary of issues:
    - The test peptide and parent protein of Mouse benchmark is heavily present in training data, although it is classified by plain.
    - The current text does not provide a visual explanation of the difference between "unseen in the mandate" and "seen across the mandate".
  - Recommendations addressed:
    - Adds a map or table showing overlaps at the pair, peptide, protein and tabk levels.
    - Make clear whether the peptide is not seen in the current tissue-H2 task, but it is only found in other tabs.
    - The test is accurately described as internal closed-task communication and is not interpreted as global unseen-peptide or unseen-protein generalization.

Rewrite Values and Language Awards and harmonize naming**

  - Screenshot: `Scene Screenshot 2026-07-26 131105.png '
  - Review of the original:

    > Only three? What names are they?
    >
    > Wrong? Is it impossible to show up in the other pairs on the sub-mission?
    >
    > This is a very confusing one, with too vague a name to tell exactly how it was designed.
    >
    > It is proposed to add a table explaining in detail the difference and connection between so many setstings:
    >
    > fixed test
    >
    > pair-grouped OOF
    >
    > matched standard 5-seed OOF
    >
    > Frozen 5-seed fixed test
    >
    > Peptide-disjoint 5-seed OOF
    >
    > matched standard OOF
    >
    > peptide disjoint OOF
    >
    > connected-component peptide-disjoint
    >
    > Paired task-level standard-versus-strict statistics
    >
    > The current presentation and naming are too technical and coding and require a more precise purpose and supportive understanding of the following:
    >
    > For example,
    >
    > standard (testing peptide not seen before in sub-task),
    >
    > peptide-disjoint (testing peptide not seen before in any tasks)
    >
    > We plan to put in a periodical on biological or biographical issues. This part of the paper is very technical, but it is not sufficiently technical to describe objectives and meanings, is less sensitive to biological implications and interpretations, and people who do not compute hard cores may not be able to understand quickly. While the language remains technical, it is modified to allow people who do not compute hard cores in the biological field to understand quickly the meaning of each setup.

  - Summary of issues:
    - Three assessment settings are stated, but the names used are significantly more than three.
    - The names do not directly reflect the division of units, the rules for overlap of entities and the purpose of the assessment.
    - The expression "pair indentifier cannot cross the little/held-out boundary" may be inaccurate or misleading.
  - Recommendations addressed:
    - Recoupling the hierarchy of assessment agreements, distinguishing between core settings, derivative settings and statistical comparisons.
    - Add a summary of the agreement, at least with a breakdown of units, peptide/protein/task, OOF, matched, number of seeds and supported conclusions.
    - Harmonizes the names in the full text, tables and graphs.

$15. Explained fine meaning of the match-grouped OOF check**

  - Screenshot: `Screen Screenshot 2026-07-26 131133.png '
  - Review of the original:

    > What does that mean?

  - Original text:

    > A matched pair-grouped OOF comparison provides an additional check that the result is not unique to the final fixed test.

  - Summary of issues:
    - Readers cannot judge what factors were controlled by the matchd patch-grouped OOF and why it could verify the results.
  - Recommendations addressed:
    - Indicates the object and matching variables for the match, such as the number of folds, the number of samples and the positive and negative ratio.
    - Describes the similarities and differences between the comparison and the final fix test.
    - It is clear what alternative interpretation the analysis would or could not exclude.

$16. Explain how the match was achieved and why the mode appeared in the table**

  - Screenshot: `Scene Screenshot 2026-07-26 13109.png '
  - Review of the original:

    > How did you achieve the metted? Did you re-create the training data? Or did you choose the part of the tsetting data?
    >
    > Why did Mouse show up here?

  - Summary of issues:
    - Table 8 does not explain how the metted standard OOF is constructed.
    - Human, the result was a sudden pattern in the subsection, and the chapter was not organized and more intended.
  - Recommendations addressed:
    - It is clear that Matched is achieved by reconstructing the fold, adjusting the training set, screening test sets or otherwise.
    - Report specific matching conditions and sample sizes after matching.
    - Move the result of the mouse to the mouse correction subsection or explain the need for cross-species comparisons here.

$17. Verifying the pair-disjoint definition in the table of the mouse**

  - Screenshot: `Scene Screenshot 2026-07-26 131224.png '
  - Review of the original:

    > The phrase pair-disjoint may be wrong.
    >
    > It is estimated that the sub-mission is peptide-disjoint.
    >
    > And all the other tasks puptide-disjoint.

  - Summary of issues:
    - The pair-disjoint used for Table 9, Table 10 may not accurately reflect the actual division rules.
    - The sub-task is confused with the global peptide-disjoint.
  - Recommendations addressed:
    - Verify the actual rules used for each table against the data classification code.
    - Nams " Not seen in the current task" and " Not seen in all task " .
    - Check for consistency in text, tabulations, notes and methodology.

$18. Explains the relationship between the mixed test, met bandard OOF, peptide-disjoint OOF and train-pol OOF**

  - Screenshot: `Screen Screenshot 2026-07-26 13148.png '
  - Review of the original:

    > What's a cross-test?
    >
    > What's matched standard oof?
    >
    > What's a Peptide-disjoint OOF?
    >
    > What's it to do with the front of the train-pool OOF?

  - Summary of issues:
    - The main assessment agreement was not interpreted in situ near the results table.
    - The relationship between the train-pool OOF and the standard, matched, strict/peptide-disjoint OOF is unclear.
  - Recommendations addressed:
    - A single definition is given where it first appears and a uniform protocol statement is cited.
    - Indicate the data pools, training/test boundaries and main assessment objectives used for each agreement.
    - Avoid using different names in different tables for the same agreement.

## III. MANDATE, DATA AND DEFINITION OF INDICATORS

19. Explanation of origin of 44-task subset**

  - Screenshot: `Scene Screenshot 2026-07-26 131122.png '
  - Review of the original:

    > Why 44 task? Where's 44 task?

  - Summary of issues:
    - RQ1 suddenly uses the original 44-task subset, but the text does not specify how it is constructed.
  - Recommendations addressed:
    - Describes the definition of each tabk.
    - Gives 44 tabs of screening criteria, data sources and sample size thresholds.
    - Explains the relationship between 44-task subset and the subsequent 157-task benchmark.

$110. 44-task** for definition in Table 4

  - Screenshot: `Screen Screenshot 2026-07-26 131141.png '
  - Review of the original:

    > How?

  - Summary of issues:
    - The table in Table 4 is again used for the original 44-task station fair-disjoint benchmark, but there is still no definition.
  - Recommendations addressed:
    - Briefly describes the origin of 44-task in the table note and quotes the complete data-building section.
    - This is why the historical baseline cannot be directly matched to the 157-task master experiment.

$111. Explanation of meaning of pair-grouped**

  - Screenshot: `Scene Screenshot 2026-07-26 131146.png '
  - Review of the original:

    > What do you mean, "air-grouped"?

  - Summary of issues:
    - Table 6 does not describe which entities the pair consists of, and how the subgroups prevent leakage.
  - Recommendations addressed:
    - Is it a peptide-parent protein, tessue-MHC, peptide-MHC, or is it a combination?
    - Indicates whether all records of the same pair always enter the same fold.
    - Indicate which entities are permitted to repeat across the old rules.

$112. Interpretation H2**

  - Screenshot: `Scene Screenshot 2026-07-26 131218.png '
  - Review of the original:

    > H2 what does that mean?

  - Summary of issues:
    - Tissae-H2 in the RT5 heading is not sufficiently clear to readers of non-rat immunogenetic background.
  - Recommendations addressed:
    - The H-2 was described at the time of the first appearance as a compatibility complex system for the main tissues of mice.
    - Explains the H2 retribution correspondence with human HLA retribution.

$113. Report total number of use of tissue and expansion of the tissue-level**

  - Screenshot: `Screen Screenshot 2026-07-26 131232.png '
  - Review of the original:

    > How many of them were recommended for full reporting, with additional indicators and multiple lines.

  - Summary of issues:
    - Table 12 Only some low- and high-extreme organizations were reported, and no full number of organizations were reported.
    - The number of mandates and indicators per organization are small and it is difficult to judge stability.
  - Recommendations addressed:
    - Reported total number of tissue for use benchmark.
    - Provide complete results of all tissue in the body or in the supplementary material.
    - Adds information on tasks, AUROC, AUPRC, PairAcc, confidence intervals or variance.

$114. Definition of PairAcc and MHz-only**

  - Screenshot: `Scene Screenshot 2026-07-26 131239.png '
  - Review of the original:

    > What does PairAcc mean?
    >
    > What does MHC-only mean? Is it a mix of positive and negative samples of different tissues? Maybe a sign of contacting?

  - Summary of issues:
    - The calculation objects, formulae and tie policy for PairAcc are not clear.
    - The data construction of the MHz-only comparison is unclear and there may be the same problem of peptide-MHC conflicts of labels in different tissue.
  - Recommendations addressed:
    - Gives the formal definition, pairing rules and parallel value treatment for PairAcc.
    - Lists the input characteristics of the MHz-only model and the tissue information that it excludes.
    - Describes how duplicate records and conflict labels are treated when cross-tissue samples are merged.

$115. Defined-component peptide-disjoint**

  - Screenshot: `Scene Screenshot 2026-07-26 131258.png '
  - Review of the original:

    > What is connected-component peptide-disjoint?

  - Summary of issues:
    - The strict architecture experiment used a connected-compont peptide-disjoined rules without explaining the structure of the chart and the rules for its classification.
  - Recommendations addressed:
    - Description of what the nodes and edges represent in the graphs.
    - Describe how training and testing are distributed according to the agreed component.
    - Explaining what kind of information the rule would be additionally prevented by comparison with the general peptide-disjoint.

## IV. SCOPE OF EXPERIENCE, CORRECTION AND APPROACH

$116. Increase in the number of methods for the master results table**

  - Screenshot: `Scene Screenshot 2026-07-26 131146.png '
  - Review of the original:

    > The number of methods is too small to add more

  - Summary of issues:
    - The small number of models for Table 5, Table 6 makes it difficult to judge the advantages of TissePMHC vis-à-vis existing methods in a comprehensive manner.
  - Recommendations addressed:
    - Add classic models, external pMHC tools and more complete internal architecture comparisons.
    - A distinction is made between "direct matching comparisons" and "comparisons for information purposes only because of different data sets".

$117. Improve visual quality and methods of Figure2**

  - Screenshot: `Scene Screenshot 2026-07-26 131153.png '
  - Review of the original:

    > The pictures are not very precise, they need to be more beautiful. Why are there so few ways?

  - Summary of issues:
    - The colouring, layout and labeling of the box charts are not sufficiently suited for official papers.
    - The methodology is less extensive and the differences in the assignment-level pairing are not adequately demonstrated.
  - Recommendations addressed:
    - Harmonizes the colour, font, line width and label orientation of the papers.
    - Additional baseline methodologies.
    - Consider adding task-level scatterpoints, pairing lines, differential distribution or confidence intervals.

$118. Melting experiments to explicitly transform Table 7 into a norm**

  - Screenshot: `Screen Screenshot 2026-07-26 131202.png '
  - Review of the original:

    > - What does that mean?

  - Summary of issues:
    - Table 7 The evidence for the different components is shown together, but in part a comparison is made using different task numbers,seed or aggregation.
    - The current results cannot be interpreted directly as a strict component dissipation.
  - Recommendations addressed:
    - Make sure the table is descriptive and official.
    - If used as a melting experiment, the data division, training process, Seed, task collection and aggregation methods should be maintained.
    - Removes or replaces multi-kernel encoder, auxiliary branch, Fusion rule and sees avelaging.

119. Expanding mouse methods and indicators and reducing unnecessary table space**

  - Screenshot: `Scene Screenshot 2026-07-26 131224.png '
  - Review of the original:

    > Too few methods, too small a table, a few more indicators, a few more methods.

  - Summary of issues:
    - Mouse Table 9 and Table 10 have a limited number of methods and indicators.
    - Page space was not used very well, but experimental information was still insufficient.
  - Recommendations addressed:
    - Add methods to the human experiment.
    - Added MSC, capaciracy, PairAcc, World-task and uncertainty statistics.
    - Adjusting the table layout to increase information density and comparability.

120. Harmonized methodology for comparison of the strict structure of Human and Mouse**

  - Screenshot: `Screen Screenshot 2026-07-26 13104.png '
  - Review of the original:

    > Why isn't it the way it works?

  - Summary of issues:
    - The pool of candidate models used by Human Table 16 and Mouse Table 17 is not consistent.
    - Readers cannot judge whether the missing method is not functioning, performing poorly or not completing the experiment.
  - Recommendations addressed:
    - The same model aggregation is used to the extent possible for both species ' rigorous structures.
    - If some methods do not apply to mouse, the reasons need to be clearly stated.
    - Harmonized methodology name, ranking and indicator columns.

## V. Recommended priorities

### High priority: impact on the credibility of the findings

- [ ] Verify all data classification codes and representations, in particular the pair-disjoint, the task-level peptide-disjoint and the global peptide-disjoint.
- [ ] Explain the overlap of the peptide/protein/task in the cross-fixed test, limiting the scope of application of the corresponding conclusions.
- [ ] Defines and harmonizes the names of cross-test, standard OOF, match-distid-disjoint OOF, trade-pol OOF, etc.
- [ ] Describe the specific construction of the matchped data set or the fold.
- [ ] Check whether MHC-only cross-organizational conflict labels exist.

### Medium priority: affect the integrity of the experiment

- [ ] Add additional baselines that apply and can be replicated.
- [ ] Increase in the number of rigorous, equivalent-conditioned component-dismantling experiments.
- [ ] Expansion of methods, indicators and uncertainty reports for Human and Mouse.
- [ ] A methodology for harmonizing the Human versus the Mouse architecture.

### Expression and priority

- [ ] Supplementing the delivery mechanism and clinical significance of the tissue specific antigens in the Introduction.
- [ ] Increase the assessment agreement matrix and the data disaggregation diagram.
- [ ] Provide definitions for 44-task, H2, PairAcc, MHC-only, pair-grouped and connected-component peptide-disjoint.
- [ ] Improve the layout of Figure 2 and related tables.
- [ ] The full text reduces over-technical representation and complements biological interpretation.
