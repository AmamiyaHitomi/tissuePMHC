# Professorial Predictable List (2026-07-23)

## Positioning benchmarks

- Paper Source: `paper.tex`
- Text page number in screenshot: pp. 18-23 and appendix 30
- The following line numbers correspond to the current workspace version of 2026-07-23; the subsequent editor `paper.tex` backline numbers may be moved.
- "Support position" means that the definition or result of the comment is available in the paper and may be used to explain it, and is not necessarily a page on which the professor has written.

## Screenshot 1: Page 18 (starting RQ1 with RQ2)

Professor's comment, direct position, link/support position, location description, location description, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location
|---|---:|---:|---|
"RQ1, this format is problematic."  ZXQ0QZ ZXQ1QZ RQ1 and its consistency with the formatting of other RQ headings.
"This table is too small to add a few different indicators, such as AUPRC, Sensitivity, Speciality, Media AUROC ..."  ZXQ0QZ  ZXQ1QZ  RQ1, three lines, list; only Model, Tasks, Mean AUROC are currently in the list. The definition of indicators is concentrated in the Evaluation Protocol.
"Table without number"  ZXQ0QZ `paper.tex:572–596,767–787` This table is only `center + tabular`, there are no `table`, `\caption`, `\label`; the latter two are the same as the existing numbered table.
ZXQ0QZ `paper.tex:384,570–600` RQ1 here is only one-hot comparison with shared encoder; a more complete traditional/neurological baseline is based on a strictly folded structure comparison.
"Adding lines: it is recommended that reference be made to earlier reports for the baseline" ZXQ0QZ ZXQ1QZ Data lines of the RQ1 form be expanded; other locations for the current paper already have candidate baselines such as BLOSUM62 RF, Dualbranch, MMoE etc.
"Model name suggests the removal of Net suffix, which is harmonized to TissePHC" First defined `paper.tex:11,20,41`; this page `paper.tex:444,455,470,478–485,488–500` All old model names in full This is a full-text level name change, not just page 18.
ZXQ0QZ `paper.tex:384,570–600` RQ2 master results sheet currently has only 4 models; the strict architecture table has a more complete list of models.

### Response to answer and process results

1. ** "RQ1, this format is a bit problematic."**
   The title RQ1 has been modified to question ** "Is Tissue-speciality Learable?"** and is consistent with the structure of research issues in RQ2-RQ5.

2. ** "The table is too small to add a few different indicators, such as AUPRC, Sensitivity, Speciality, Media AUROC ..."**
   The original three rows have been expanded to **Table 1**, with the addition of AUPPC, World-10 AUROC and Seeds, and the alignment of table fonts, row heights and captions. No additions have been made to Sensitivity and Space: Early 44-task experiments did not preserve the matrix of confusion under the harmonized decision threshold, and AUROC/AUPRC cannot reverse the two indicators. Media AUROC was added to the main results table 157-task (Table 2); 44-task history archive lacks all comparable mission-by-mission medians for baseline, and therefore the column is not constructed in Table 1.

3. ** "The table is not numbered."**
   This is manifested in the use of the official `table` environment, with cross-references to ZXQ1QZ, `\label` and text code **Table 1**.

4. ** "Is there a comparison with existing methods?"**
   The list of historical examples of actual completion and comparison in the current project is now aggregated. The following are the following:

5. ** "Adding lines: suggested reference to earlier reports; baseline."**
   Table 1 has been expanded to 11 historical lines and describes them in caption as original 44-task station plain-disjoined benchmark, avoiding confusion with subsequent 157-task benchmark.

6. ** "Model name suggested to remove Net suffix and to unify it to TissePMHC."**
   The full text has been revised. The name of the model in the title, summary, Methods, Revers, Discussion, table, graph and cross-references is harmonized to **TissePMHC**.

7. ** "Add more methods of comparison and more lines."**
   A verifiable method has been added to the history baseline table in RQ1; 157-tasktable 2 of RQ2 retains four major controlled models with the same task list and assessment configurations to avoid mixing results from different task ranges or from different help-out tools into the main ranking.

## Screenshot 2: Page 19 (Figure 1, Matched OOF Table, Start of RQ3)

Professor's comment, direct position, link/support position, location description, location description, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location
|---|---:|---:|---|
"The picture is complicated and not so simple." `paper.tex:460–486`   PGFPlots column, graphs and labels for Figure 1.
"Additional to the table above; additional methods; recommendations for boxplot/violin plot; if you want to show the range  data sheet ZXQ0XZ; figure ZXQ1QZ  single seed results ZXQ2 XZ; mission-level results source is shown in the column chart of the current map using individual aggregated values of each method, with no task distribution, seed distribution or error range.
ZXQ0QZ `paper.tex:670–704` Direct comment on Figuure 1; the papers currently available are Task Heterogeneity and Branch Complexity as the portal for refinement.
"Why is this a baseline??"  ZXQ0QZ; the name ZXQ1QZ  baseline definition in the table `paper.tex:384`  means "matched auxiliary dual-branch baseline / Matched auxiliarybranch?" The method partly describes the structure of the auxiliary MLP dualbranch, but there is no separate explanation here as to "why it was chosen as Matched base".
`paper.tex:505`  Detailed results `paper.tex:519–600`; full statistical table `paper.tex:767–787`  505 line reports only the start AUROC/AUPREC and points to RQ4, the full list results in RQ4.

### Response to answer and process results

1. ** "The picture is complicated, not so simple."**
   The Figure 1 has been redesigned. The new figure presents the Mean AUROC, Median AUROC and World-10 AUROC, which allows the graphic to display both overall performance and mission distribution centres and low end performance, rather than simply repeating three aggregation values.

2. ** "Reciprocal to the table above; additional methods added; recommendation boxplot/violin plot; if you want to show range."**
   Figure 1 is aligned with the main models and indicators of Table 2. The complete boxplatt/violin plot is not generated by force, as different historical methods do not preserve fully consistent task-by-task, seed-by-seed feeds; forced mergers mix 44-task, 157-task, cross-test and OOF sample pools. Currently, the Meran, Median and World-10 summaries are used to allow reliable review of available data.

3. ** "The results can be further refined according to the scale of the tissue/specify."**
   The human scale has been added to the following Analysis/RQ5: humans-level exceptions, HLA locus, mouse scale extremes and H2 restriction. The results of single tasks are clearly marked as descriptive results, avoiding the interpretation of very small sample groups as stabilizing effects.

4. ** "Why is this a baseline?"**
   The text has been supplemented by an explanation: Matched auxiliary dual-branch base shares two-brand surveillance and integration settings with the ultimate TissePMHC, the main difference being MLP encoder and position-positioning multi-kernel encoder, so it provides controlled comparisons for encoder replacements rather than randomly chosen baselines.

5. ** "Where did the results come from?"**
   RQ1/RQ2 duplicate reports on the same stric values have been deleted and the complete results of the global unseen-peptide have been assembled in RQ4. RQ2 only retains the bridge link description and points to RQ4, the official comparison table and the full match sheet through cross-references.

## Screenshot 3: Page 19 at the end of Q3 Compont Conventions

Professor's comment, direct position, link/support position, location description, location description, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location
|---|---:|---:|---|
"Reference to the corresponding table"  `paper.tex:509–515`  master result table ZXQ1QZ; mashed OOF table ZXQ2QZ; list structure ZXQ3QZ RQ3 with three elimination conclusions currently written directly and no `Table~\ref{...}`. The first two RQ2 tables are not cross-referenced.

### Response to answer and process results

1. ** "Reference to the corresponding table."**
   Formal tables of component events have been added and are cross-referenced in RQ3. The tables focus on multi-kernel encoder, branch complement, cross-transaction and seed observing four system-level evidence. The text also makes clear that existing experiments support system-level contributions on the standard Benchmark, but cannot claim full component ranking without the same strictolds being isolated.

## Screenshot 4: Page 20 (RQ4 start with standard/strict results table)

Professor's comment, direct position, link/support position, location description, location description, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location
|---|---:|---:|---|
"OOF What does it mean? There is no need to appear, Pair-disjoint?"  `paper.tex:523,528–531` Three assessment settings ZXQ1QZ; Evaluation Protocol ZXQ2QZ  OOF is out-of-fold. The paper is first introduced in full acronyms in line 195, but the RQ4 form uses direct acronyms; pair-disjoint is two separate compartments.
ZXQ0QZ  ZXQ1 XZ  human standard 0.765/2.7452 appears in RQ1, RQ2; RQ4 appears again for match of margin for matchted standard-vs-strict.
"The complete results on mice?"?  ZXQ0QZ  pattern table ZXQ1QZ; RT5 `paper.tex:602–664`  RQ4 opening table only listed AUROC; mouse AUROC/AUPREC/Work-6/PairAcc and standard results are spread over the strict architecture table and RQ5.

### Response to answer and process results

1. ** "What does that mean? It doesn't seem necessary, Pair-disjoint?"**
   Already defined in pre-methods and RQ4: OOF is **out-of-old**. Standard pair-disjoint OOF only prevents the same match-fair ID from appearing in the same matchping and help-out data; negotiated-compont peptide-disjoint OOF separates the whole picture identity.

2. ** "Repeal results?"**
   The duplicate 0.7652/0.7452 is deleted from RQ1/RQ2, and the full standard-versus-strict results and explanations are concentrated in RQ4; other chapters point to RQ4 only through cross-references.

3. ** "The complete results on mice?"**
   RQ4/RQ5 is now reporting the mouse standard OOF, fit test and peptide-disjoint OOF, and contains existing verifiable indicators such as AUROC, AUPRC, World-6, PairAcc; and adding H2 recovery and Tissuue stratification results.

## Screenshot 5: Page 20 (Figure 2 and RQ4 statistics)

Professor's comment, direct position, link/support position, location description, location description, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location
|---|---:|---:|---|
"Red dot for the diagonal line? Is this standard operation?"  Human `paper.tex:552`; Mouse ZXQ1QZ  Figure `paper.tex:558`  Source is actually painted with black dot ZXQ3QXXity line, not red dots; red squares in the screenshot are the rewriting/showing effects of the dotted line. This line is used to distinguish strits from or below strid.
"peptide-disjoint and pair-disjoint" are misleading; unseen in special task and unseen before in any task "  Axis/ Diagram ZXQ0XZ; explain `paper.tex:562–566` Protocol definition ZXQ2XZ; limit `paper.tex:740` current strit is global peptide intity and therefore "the strateg does not appear in any mission training discount", while the task itself is seen task; standard dai-disjoint only guarantees that there is no overlap.

### Response to answer and process results

1. ** "Diagonal dots? Is this standard? "**
   The figure shows clearly that it is a reference rather than a test data point. The line is used to determine whether the data line is rising or falling relative to the match.

2. ** The words "peptide-disjoint and pair-disjoint are misleading; need to discuss unseen before in any task."**
   The protocol definition, RQ4, the axis of coordinates and the graphs have been clarified: provisional peptide-disjoint indicates that peptide does not appear in the task 's little hands, i.e. **global unseen peptide**; but the assessed tesue-MHC task itself is still seen. It is not just missing from a particular task, nor unseen-task or protein-disjoint evaluation.

## Screenshot 6: Page 21 (RQ5 Mouse Master Table)

Professor's comment, direct position, link/support position, location description, location description, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location
|---|---:|---:|---|
"Form too small"ZXQ0QZ Rat rat-on-ly OOF table, currently 5 methods, 3 performance indicators and Evidence.
"More indicators, more methods" ZXQ0QZ  Full strict methodology/indicator `paper.tex:572–596`; indicator definition `paper.tex:390–408`  expandable Accuracy, MCC, PairAcc etc. or complete other completed mouse baselines; subject to the already available help-out prections.
"In accordance with the organization/ in accordance with the split" ZXQZ H2 group results ZXQ1QZ; Tsk analysis structure `paper.tex:670–694`  Professor wants the mouse results to be further displayed in the tissue and H2 distribution. Only at 660 are currently given to H2 group AUROC.

### Response to answer and process results

1. ** "The table is too small."**
   The caption space and table row height have been added uniformly; the table maintains readable characters and no more mandatory overall scaling.

2. ** "Mo more indicators, more methods."**
   The mouse baseline table has been expanded to BLOSUM62 grandester, share encoder, Factorized MMoE, H2-Kk residual apperter and five-said Fedorized MMoE, and reports AUROC, AUPRC, World-6 and Evidence. The other agreement matrix supplements the missing indicators by adding only those with ready-made, co-agreement-based help-out results.

3. ** By organization/by segment.**
   A table and a layer of H2-Db/H2-Kb/H2-Kd/H2-Kk have been added. The text clarifies that these subgroups result in descriptive analysis and does not interpret groups with very little task as controlled biological effects.

## Screenshot 7: Page 22 (Figure 3, OOF and cross-test table)

Professor's comment, direct position, link/support position, location description, location description, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location
|---|---:|---:|---|
"Why is this something?" comparison table ZXQ0QZ; Figure 3 ZXQ1QZ  Introduction `paper.tex:621`; Interpretation `paper.tex:660–664`  This table/figure is used to compare the one-time internatixed test of the train-only OOF model with the freeze model, with the intention of making a standard internal confirmation of the pair-disjoint; line 662 also recognizes the existence of significant overlaps of the cross-reference test.

### Response to answer and process results

1. ** "Why is this thing here?"**
   The original Figure 3 was deleted because it only repeated three summary values in the adjacent table, without providing distribution or uncertainty information. The protocol comparison table was retained and expanded to distinguish the one-time transaction test of the train-only OOF, freezing model, and the peptide-disjoint OOF; the text also states that the cross-test has a significant trade/test entry overlap, so only learningability under the internal standard protocol cannot be confirmed and cannot be demonstrated by strict physical generalization or model migration.

## Screenshot 8: Page 23 (HLA locus descriptive table)

Professor's comment, direct position, link/support position, location description, location description, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location
|---|---:|---:|---|
"Form too small"ZXQ0QZ Fuller task statistics `paper.tex:670–694` Tables only HLA-A/B/C line 3 and without numbering, caption, label; compared to orginal fixed with strict OOF, which is a descriptive comparison of non-matched sample pools.

### Response to answer and process results

1. ** "The table is too small."**
   The table has been changed to a formal numbering table, adding caption, label, text references, caption spacing and table height. The text further explains that the orginal mix test is different from the list OF uses of help-out tools, so that the table 's Diffence can only be compared descriptively and cannot be interpreted as a peptide-overlap effect; HLA-A/B/C also contains allele, tessue, training scale and peptide-component structure, nor can it be interpreted as controlled locus effects.

## Screenshot 9: Appendix Table 1 (paired statuses)

Professor's comment, direct position, link/support position, location description, location description, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location
|---|---:|---:|---|
"This table is numbered, but the format seems to be in trouble."  ZXQ0QZ  Text quote `paper.tex:564`  This is `tab:paired-stats` with caption/label; using the `\resizebox{0.98\linewidth}{!}` compressed column 8, resulting in word and readability problems. The table is re-numbered as Table 1.

### Response to answer and process results

1. ** "This table is numbered, but it seems to be in a different format."**
   ZXQ0QZ was cancelled, Human and Mouse were removed into two panels, using a single column range, row height and readable characters. The table is now officially numbered Table 12 and retains the values of Mean differenceence, Media differenceence, Hodges–Lehmann differenceence, W/T/L, 95% CI and BH-adjusted Wilcoxon ZXQ1QZ. Dismantling the panel avoided the forced scaling of the entire eight statistical tables.

## Screenshot 10: Appendix Figure 4

Professor's comment, direct position, link/support position, location description, location description, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location, location
|---|---:|---:|---|
(upshot) ZXQ0QZ  orientation paragraph ZXQ1QZ; figure ZXQ2QZ  is a summary of ZXQ3QZ average AUROC by HLA locus/H2 reduction; negative values indicate a decrease after strict split.
ZXQ0QZ  orientation paragraph ZXQ1QZ; note `paper.tex:832`  is the only single parent profile in the held-out collection with a little biting part; the ratio used to explain that peptide-disjoint is not the protein-disjoint.

### Response to answer and process results

1. ** "What does this mean?"**
   The diagrams are rewritten with the prefix, title, vertical axis and graph. The above graphs represent the average general deviations under HLA locus/H2 reduction:
   \[
   \Delta_{\mathrm{AUROC}}
   =
   \mathrm{AUROC}_{\text{peptide-disjoint OOF}}
   -
   \mathrm{AUROC}_{\text{matched standard OOF}}.
   \]
   The column is negative for AUROC to decline after the isolation of the whole world; it is quantified as peptide-separation development cap.

2. ** "What does this mean?"**
   It is clearly the only percentage of the head-out part of the equation that appears in the field. It is used to explain that pptide-disjoint is only isolated, not the entire partprotein, so peptide-disjoint is not protein-disjoint. The figure also indicates that ZXQ0QZ means matchted OF control, not or original match test.

## Summary of cross-page questions

1. ** The tabular system is not uniform**: RQ1, RQ2, two tables, RQ4, opening tables, RQ5, two tables and HLA list tables are naked `center + tabular`, with no uniform numbering, caption and label; only the strict achite table and the appendix statistical tables are used in the official `table` environment.
2. ** The results are repeated but the narrative is not clearly distinguished**: 0.7652/0.7452 appears in RQ1, RQ2, RQ4; if retained, the words "legality/bridge to object analysis/matched comparison" should be clearly identified.
3. ** Terminology need to be pre-defined and precise**: OOF, standard pair-disjoint, met standard OOF, conned-component peptide-disjoint, fixed test appears in Reults, but the clear definition is mainly in Methods.
4. ** The chart is not sufficiently dense or displaying a mismatch**: Figure 1 and Figure 3 show only aggregate column values; the professor would like to see more methods, more indicators, distribution/range, and Tissue/MHC stratification.
5. ** Complete baseline exists but is dispersed**: the most complete human/rats stric model is compared to the model list at ZXQ0QZ and Methods is located at `paper.tex:384`; it can be used as a source for restructuring the master results table.
6. ** Name change is a full-text task**: the old model is given a uniform name `TissuePMHC`, covering titles, summaries, methods, results, discussions, chart labels and graphs, and cannot be changed to just the page where the map is located.

## Final treatment of cross-page issues

1. ** The system of tables is not uniform**: the results tables for the new and modified tables have been harmonized into formal ZXQ0QZ, caption, label and cross-references for text; the caption is added to spacing and row height and `resizebox`, which would significantly reduce the font, is removed.
2. ** Repeated results**: provisional global unseen-peptide results are grouped in RQ4, RQ1/RQ2 only with necessary cross-references.
3. ** The terminology definition is delayed**: OOF, macheted stand, pair-disjoint, conned-component peptide-disjoint, fixed test and global unsenior peptide have been explained explicitly at the first use of Methods and Reults.
4. ** The density of graphs and figures is insufficient**: Figure 1 increases the Mean/Median/World-Tail information; Figuure 2 enhances the identity reference; former Figure 3 is deleted as a result of duplicate table data; new tisue, HLA/H2 and protocol stratification.
5. ** Baseline fragmentation**: 44-task History Baseline, 157-task Master Model, Mouse Baseline and Standard Protocol results have been reorganized by research issues and different task ranges and sample pools are no longer forcibly mixed into the same ranking.
6. ** Full-text name**: Old model name has been replaced with `TissuePMHC`.

## Current version review results

- Current Overleaf 2nd edition is 33 pages.
- LaTeX Errors:0.
- LaTeX Warnings:0.
- Overfull boxes:0.
- The remaining Underwood hints are automatically re-lined in narrow columns and do not cause transboundary, shield or loss of content.
