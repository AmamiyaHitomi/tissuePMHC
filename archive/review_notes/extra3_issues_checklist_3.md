# Third round of review of the list of issues

> Summarizes, de-heavy, based on 14 screenshots and `details summary.txt ' in `pic_issues_3`.
> The premise for this round is that all results should be recalculated using new data on the occurence-balanced; positive and negative samples should appear the same as when all data are concentrated as positionive. The current round should be treated as a reconstruction independent of the old software, and the old results cannot be used.

## P0: Data and experiments must be solved first

- [ ] **Run all experiments with new data and replace full-text results.** Update text figures, tables, graphs, appendices, graphs, table notes and cross-references to ensure that old data results are not left behind.
- [x] ** Verifying observation balancing.** Train, test and complete data have been verified by pair for Human/Mouse: the total number of positive occurrences in positive and negative samples is identical to the total frequency distribution, with micmatch 0; the reviewable results are available in `extra3/occurrence_balancing_audit.json`. Visible updates of Figure 2 are still being completed in the follow-up graphic task.
- [x] ** Elimination of the bias presentation from old data.** Validation of occurence-count Shortcut: the same distribution of occurence for each tabk, minimum-only AUROC, minimum, mean and maximum value of 0.500. The paper was changed to "the specific shortcut is controlled", with residual bias limits such as IEDB coverage, study/batch, donor, depth of detection, etc., avoiding claiming that all bias are missing.
- [x] ** Uniformly evaluated protocols and verified whether they were actually used.** Official-matched experiments use only fixed Train/test; no fixed/predation products of intellectual or peptide-separated cross-assessment.v7 protocol tables, flowcharts and methodological descriptions have been changed to cross-test-only, and old cross-checking appendices have been excluded from the main text.
- [x] ** Clear duplicate running settings for Human and Mouse.** Both species use dicts 20260704, 20260705, 20260706; default aggregations are numeric averages and sample standards for dicta and sample deviations, only when clearly marked ensemble indicates average line-level projections.
- [x] ** Survey and explain the asymmetrical size of interspecies data.** Human is 77 tasks/25,339 scales, Mouse 11 tasks/2, 639 scales; both are fixed at 50 tests per task, test size differences are from task inventory, training scale differences are from common screening and available records overlaying.

## P1: Overall logic and structure of the paper

- [ ] ** Redesign the syllabus of the paper.** Reordered in the order of general thesis: data and task definitions  data balance and leakage audit  evaluation agreements  Key findings  Strict generalization  digestion/mechanism analysis  tissue/allele heterogeneity  External control and supplementary results .
- [ ] ** Reorder all chart sequences and textual references.** The chart should appear at the first explained location; remove "results that do not refer to tables" from the body "results that appear first, define that and recurring problems with the same result in the body/appendix.
- [ ] ** Uniform chart number.** Solve the confusion in numbers and references resulting from the split of two Table 15, two Table A2 and "continued"; delete, renumber and update all cross-references.
- [ ] **Utility of numbers.** Performance values in the papers are in principle uniform after 4 decimal places are retained; percentages, counts and statistics are also harmonized according to pre-declared rules.

## P1: Main Text Table

> Completed: The original Table 1-18 problem was deleted, merged and re-configured and corresponds to the current continuous numberable Table 1-13; the old number was no longer mechanically retained.

- [x] **Table 1: deleted.** This table is no longer used to carry out the risk management function; necessary related comparison of work is changed to the main experimental baseline table described or regulated in the body.
- [x] **Table 2: Update with new data.** Synchronize textual references and conclusions.
- [x] **Table 3: Updates with new data.** Synchronize textual references and conclusions.
- [x] **Table 4: Update and add Human to the new data.** Current use of access-overlap audio is given only in the same table, with Human and Mouse listed in the same table, with clear distinctions between pair, peptide, parent protein and crrent-ZXQ0QZ overlap.
- [x] **Table 5: Merge two paragraphs into one table.** Only true evaluation mode is maintained, with over-abroad methodological terminology, model description, repeated training or forecast averages no longer included; and the unused entry " otherwise cross-valued " is deleted.
- [x] **Table 6: Update with new data and explain the relationship to Table 4.** Make clear whether the two tables answer the same combination-test overlap or cross-value overlap; if the information is substantively duplicated, one of them should be merged or deleted.
- [x] **Table 7: Redo label-confact audit.** At least one label peptide, more than one label peptide, and the number of peptide in which the label contract occurs; conflict ratio should be "more than 1 label peptide" as the direct relevant denominator, while maintaining the necessary aggregate information.
- [x] **Table 8: Expand to at least 10 methods.** Method pools refer to Table 12 and ensure that data disaggregation, task pools and indicator calibres are comparable.
- [x] **Table 9: Delete or prove necessity.** The current text is not quoted and it is unclear what conclusions the models can support; if retained, the text guidance, the purpose of the experiment and the difference from the main result must be filled.
- [x] **Table 10: Whether or not it is ablation study.** In the case of a melting experiment, the same data, old, Seed and indicator-by-section should be used to remove the component; if not, the two types of information that are mixed should be clearly listed. Also, a description of its relationship to Table A2 should be made, avoiding duplication of evidence.
- [x] **Table 12: Update and add indicator columns to new data.** Complement suitable task-level, robust or uncertain indicators, in addition to AUROC/AUPPC/Work-k; check data-volume differences in Human/Mouse.
- [x] **Table 13: Remove single-line tables.** First change to a higher volume of information figure; if the tables must be maintained, add multiple indicators and sufficient objects for comparison.
- [x] **Table 14: Update with new data.** Also, as a location for integration table A4/A5, harmonizes the results of the Tissue-blind/external-control.
- [x] ** Previous Table 15: Update and expand methods with new data.** The methodology is aligned with Table 12 and indicates which results are directly comparable.
- [x] ** The latter Table 15: Delete the stand-alone duplicate table and merge it into Table 8.** Avoid duplication after consolidation.
- [x] **Table 16: Update with new data and explain in text.** Current cross-assessment results are not covered in text; additional results need to be read, relate to other assessments and explain why methods are selected.
- [x] **Table 17: Update with new data.** Keep the number of tasks per tissue, recalculate all indicators and verify that the original bias under the new data are missing.
- [x] **Table 18: Change to polar overview.** Lists the top and bottom five four-digit splits (four-digit allele typeping) respectively, with a clear ranking of indicators, tabs and the necessary stability indicators.

## P1: Main Text

- [x] **Figure 1: Only one representative tisue.** Changed to a single Human lung route, listing the genes associated with source/cell-state, proteaome, TAP, EAP, ladding-complex and surface MHz-I; figure to indicate the 7-task of lung based, interlayered and non-caust range of selections.
- [x] **Figure 2: Update with new data.** Total value of total tsel 0/1 displays have been used for complete occurence-equal Human/Mouse train+test; 13 Human tissues and 4 Mouse tissues have been 0 for graphics, with requests and definitions and log axes.
- [x] **Figure 3: Verification of the official cross-assessment.** Figure 3 only retains the TissePMHC structure and no longer appears as the official CV; the three-fold reduction in training is only shown as a cross-reference step in the Figure 4 and the evaluation-protocol tables, with the only result agreement.
- [x] **Figure 4: Focus on re-engineering.** Re-programmed as a separate top-down trade-only and untouched-fixed-test stream of information, add phase number, boundary, arrow and consistent functional colour, clarifying the relationship between turning, contact locking, financial finding and reporting.
- [x] **Figure 5: delete B panel, only keep and expand A.** Changed to single panel 23-method Human Achitecture search, AUROC/AUPRC and seed SD are identical to the main results sheet, same task, same calibre.
- [x] **Figure 6: Update and encode the tissue information with new data.** All 77 Human and 11 Mouse have been shown for post-relocation tasks; point colours only encoded tisue, navy crosslines are only summarized as tissue mean and random reference lines are clearly non-data.
- [x] **Figure 7: Update with new data.** The AUROC difference between the re-referenced and the pre-register tasks has been used, the body and figure have been synchronized with the Human/Mouse average, 10,000 task bitstrap intervals, improved task numbers and the Wilcoxon results of the double-sided pair corrected by Holm.

## P2: Appendix Table

- [x] **Figure A1: deleted.** The old figures for image/peptide-separated genomicization and the appendix entry have been removed; the new draft does not report the strict generalization value of the data that did not run again on the occurrence-matched.
- [x] **Table A1: deleted.** The table has no independent information value and is not quoted in the unnecessary body of text and is not retained in v7.
- [x] **Table A2: Delete and weigh.** Both paragraphs repeat numbered and filtered AUROC tables with unclear criteria are removed; comparable complete system evidence is reported in the current main file Table 8 (companted-retraining documents) and complete Achitecture tables.
- [x] **Table A3: Delete and harmonize calibres.** The main text is reworded using the same three readings report as $ZXQ0QZ sample SD; the method part explicitly acquiesces to the average line level projection of the Seed, so that the duplicate and confusing table of table is no longer maintained.
- [x] **Table A4 and A5: Main text has been merged with new data.** Current Table 13 uses the same table header and display order as the same list of tables and displays under the same occurence-matched match-test protocol.

## P2: Description and consistency check

- [x] ** Answers "What is this table to indicate" for each reserved table."** The current 13 tables and 7 graphs are guided by text; the subject matter/charts distinguish between data audits, model structures, evaluation protocol, fixed test performance, heterogeneity and external control, and clearly identify the applicable data disaggregation, tasks, Seeds/determinate calculations, indicators and conclusions boundaries.
- [x] ** Cleans up the over-general glossary of terms.** The Evatation Protocol chapter has been independently set and the classification, optimization-only model definition, single-sched small, fixed-test evaluation, representational analysis and statistical hierarchy has been eliminated.
- [x] ** Checking the comparability of all tables.** Full version of complete data control; harmonized test, task invertory and three seeds, row-level aggregate with $ZXQ0QZ tags; tab/tassue/allele two-stage aggregation, determinative data audit, external-only indicators and non-direct interchangeability of Human World-10/Mouse World-5 have been indicated.
- [x] ** Complete examination of Human/Mouse symmetry.** Human/Mouse baseline statistics, balance/spill audits, master results and external controls have been harmonized in the definition, indicator and presentation sequence; Mouse master results have been completed with ACCURACE/MCC, H2 aggregated with task counts. The reasons for the methodological inventory of 23-vs-20, the scale of tasks of 77-vs-11, World-k and 29 HLAtyping displays only extremes, while the reasons for all four H2 representations have been clearly stated.

## Recommendation order of implementation

1. Freezes new data and the results of the Occurence-balancing validation.
2. Confirms the design of the evaluation modes, old and run/seed.
3. Rerun all experiments and output of the unified results matrix.
4. The structural reconstruction of Table 4, 5, 7, 8, 10, 12, 14, 16–18 is completed first.
5. Rewrite Figure 1-7, delete Figure A1.
6. Merge/delete duplicate tables in the appendix, and harmonize numbers with cross-references.
7. Reorder the text to the new outline and conclude with a consistent audit of values, terminology, chart references and four decimal places.
