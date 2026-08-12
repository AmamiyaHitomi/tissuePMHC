# tissuePMHC Project Report

## 1. Project goal

This project studies tissue-specific presentation preference of human `HLA-I` peptides.

The final goal is to build a clean machine learning dataset named `tissuePMHC`.

The task is:

```text
For one tissue-HLA pair, predict whether a peptide is likely to be presented in this tissue under this HLA allele.
```

In this stage, we finished four main tasks:

1. Downloaded and processed the `IEDB MHC ligand` dataset.
2. Built paired positive and negative peptide samples.
3. Built the standard `tissuePMHC` training and testing datasets.
4. Tested several simple baseline models.

---

## 2. Raw data

The raw data came from `IEDB`.

Download link:

```text
https://www.iedb.org/downloader.php?file_name=doc/mhc_ligand_full_single_file.zip
```

Local file:

```text
data/raw/mhc_ligand_full_single_file.zip
```

The zip file contains:

```text
mhc_ligand_full.csv
```

This file is large. The uncompressed CSV is about `8.8 GB`.

The CSV has two header rows:

- Row 1: column group, such as `Epitope`, `Host`, `Assay`, `MHC Restriction`
- Row 2: column name

Important raw columns are:

| Column group | Column name | Use |
|---|---|---|
| `Epitope` | `Name` | peptide sequence |
| `Epitope` | `Object Type` | keep only `Linear peptide` |
| `Epitope` | `Modified residues` | remove modified peptides |
| `Epitope` | `Modifications` | remove modified peptides |
| `Epitope` | `Source Organism` | check if the peptide source is human |
| `Epitope` | `Species` | check if the peptide source is human |
| `Epitope` | `Source Molecule` | direct source molecule |
| `Epitope` | `Source Molecule IRI` | source molecule link, sometimes UniProt |
| `Epitope` | `Molecule Parent` | normalized parent protein |
| `Epitope` | `Molecule Parent IRI` | parent protein link, usually UniProt |
| `Host` | `Name` | host species |
| `Assay` | `Qualitative Measurement` | positive or negative result |
| `Antigen Presenting Cell` | `Source Tissue` | tissue information |
| `MHC Restriction` | `Name` | HLA allele |
| `MHC Restriction` | `Class` | MHC class |

---

## 3. Main data extraction script

Script:

```text
scripts/extract_iedb_human_mhci_ligands.py
```

This script reads the raw `IEDB` zip file and extracts clean human `HLA-I` peptide records.

### 3.1 Filtering rules

We used strict filters to make the dataset clean and easy to use for machine learning.

Current filters:

1. Keep only `MHC-I` records.
2. Keep only `Linear peptide` records.
3. Keep only peptides from human source.
4. Keep only records with human host.
5. Keep only positive assay records.
6. Keep only clear four-digit `HLA-I` alleles.
7. Keep only records with a valid `molecule_parent_uniprot_id`.
8. Keep only unmodified peptides.
9. Keep only standard amino acid sequences.
10. Keep only peptides with length `9`.

### 3.2 Why keep only MHC-I records?

This project studies `HLA-I` peptide presentation.

So we keep only:

```text
MHC Restriction / Class == I
```

This removes `MHC-II` records and other unrelated records.

### 3.3 Why keep only human peptides and human hosts?

The long-term goal is related to tumor vaccines in humans.

So we only want peptides from human proteins and presented in human samples.

This avoids mixing viral, bacterial, mouse, or other species peptides into the dataset.

### 3.4 Why keep only positive records?

We want peptides that were reported as presented or bound.

So we keep assay results where:

```text
Qualitative Measurement starts with Positive
```

This gives us reported positive presentation evidence.

### 3.5 Why keep only four-digit HLA alleles?

Some IEDB records have unclear HLA names, for example:

```text
HLA-A2
HLA class I
```

These are not precise enough.

We keep only alleles such as:

```text
HLA-A*02:01
HLA-B*07:02
HLA-C*03:04
```

Reason:

- Different four-digit alleles can have different peptide motifs.
- Machine learning models need clear allele labels.
- Unclear HLA labels may add noise.

### 3.6 Why keep only records with `molecule_parent_uniprot_id`?

We later build negative samples from the same protein as the positive sample.

For this, we need a stable protein ID.

We use:

```text
molecule_parent_uniprot_id
```

because it is more normalized and has better coverage than `source_molecule_uniprot_id`.

### 3.7 Difference between `source_molecule` and `molecule_parent`

`source_molecule` is the direct molecule annotation in IEDB.

`molecule_parent` is the parent or normalized protein.

Simple example:

```text
peptide
  comes from source_molecule
    belongs to molecule_parent
```

Recommended use:

| Field | Meaning | Use |
|---|---|---|
| `source_molecule` | direct source molecule | useful for tracing original annotation |
| `source_molecule_uniprot_id` | UniProt ID from source molecule link | may be missing |
| `molecule_parent` | normalized parent protein | better for protein-level analysis |
| `molecule_parent_uniprot_id` | UniProt ID of parent protein | main protein ID used in this project |

### 3.8 Why keep only unmodified 9-mer peptides?

Most `HLA-I` ligands are short peptides, often 8-11 amino acids.

We decided to make the first standard dataset simple and clean.

So we keep only unmodified `9-mer` peptides.

This has several advantages:

1. All peptides have the same length.
2. Feature encoding is simple.
3. Machine learning models are easier to compare.
4. Modified peptides do not add extra complexity.
5. The dataset is easier for other students to use.

Removed examples:

```text
FTDPRTMGY + PHOS(T6)
HEIFTDPRTMGY + OX(M10)
```

---

## 4. Processed IEDB output files

Output folder:

```text
data/processed/
```

Important files:

| File | Description |
|---|---|
| `iedb_human_mhci_ligands.csv.gz` | evidence-level records; one row is one IEDB assay record |
| `iedb_human_mhci_ligands_unique_peptide_mhc_tissue.csv.gz` | unique `peptide-HLA-tissue` table |
| `iedb_human_mhci_ligands_unique_peptide_mhc_tissue_protein.csv.gz` | unique table with protein information; main input for pairing |
| `iedb_human_mhci_ligands_summary.json` | extraction summary |

Final extraction summary:

```text
rows_total: 5,571,809
rows_mhci: 3,643,999
rows_unmodified_standard_peptide: 3,445,971
rows_peptide_length_9: 1,990,718
rows_human_peptide_source: 1,685,478
rows_human_host: 1,681,733
rows_positive: 1,678,574
rows_four_digit_hla: 671,925
rows_molecule_parent_uniprot: 669,674
rows_written_evidence: 669,674
rows_written_unique_peptide_mhc_tissue: 514,017
rows_written_unique_peptide_mhc_tissue_protein: 549,453
```

The final processed table has:

- only human `HLA-I` records
- only clear four-digit HLA alleles
- only unmodified 9-mer peptides
- only records with `molecule_parent_uniprot_id`

---

## 5. Tissue and HLA summary

Script:

```text
scripts/summarize_tissue_hla_uniprot.py
```

Output files:

```text
data/processed/iedb_tissue_summary.csv
data/processed/iedb_tissue_hla_uniprot_summary.csv
```

Current summary:

```text
tissues: 48
tissue_hla_pairs: 1280
```

`iedb_tissue_summary.csv` columns:

| Column | Meaning |
|---|---|
| `source_tissue` | tissue name |
| `n_hla_alleles` | number of HLA alleles in this tissue |
| `n_molecule_parent_uniprot_ids` | number of source proteins |
| `n_peptides` | number of peptides |

Example:

```text
source_tissue,n_hla_alleles,n_molecule_parent_uniprot_ids,n_peptides
lymphoid,113,11664,81156
blood,101,13570,104559
NA,94,958,2238
lung,53,9949,33772
kidney,45,4958,8678
```

Tissue can be `NA` if IEDB did not provide clear tissue information.

---

## 6. Positive and negative sample construction

Script:

```text
scripts/build_tissue_specificity_pairs.py
```

Input:

```text
data/processed/iedb_human_mhci_ligands_unique_peptide_mhc_tissue_protein.csv.gz
```

Output:

```text
data/processed/iedb_tissue_specificity_pairs.csv.gz
data/processed/iedb_tissue_specificity_pairs_summary_9mer.csv
```

### 6.1 Positive sample definition

For one `tissue-HLA-UniProt` group, positive peptides are:

```text
peptides reported in this target tissue under this HLA allele, from this UniProt protein
```

### 6.2 Negative sample definition

For the same `tissue-HLA-UniProt` group, negative peptides are selected by this logic:

1. Find all positive peptides in the target `tissue-HLA-UniProt` group.
2. Find all peptides from the same `UniProt` protein and same `HLA`, but reported in other tissues.
3. Remove peptide sequences that were already reported in the target `tissue-HLA` group.
4. The remaining peptides are possible negative peptides.
5. Compare the number of positive peptides and possible negative peptides.
6. Keep the smaller group.
7. Randomly sample the same number of peptides from the larger group.
8. Randomly pair positive and negative peptides one-to-one.

### 6.3 Why use this negative sampling strategy?

This is important.

The negative sample is not a random peptide.

It must be:

- from the same protein as the positive peptide
- under the same HLA allele
- reported in other tissues
- not reported in the target tissue-HLA group

This design reduces some simple biases.

For example, the model should not only learn that one protein is common in one tissue.

Because positive and negative samples come from the same `UniProt` protein, the model is forced to learn more tissue-specific peptide preference.

### 6.4 Paired sample size

Current paired data:

```text
input_rows: 549,453
pairs: 125,649
paired_rows: 251,298
tissue_hla_pairs_with_pairs: 1,091
```

This means:

```text
positive samples: 125,649
negative samples: 125,649
total rows: 251,298
```

### 6.5 Pair validation

The script automatically checked the paired data.

Validation results:

```text
validation_invalid_pair_ids: 0
validation_mismatched_uniprot_pair_ids: 0
validation_negative_reported_in_target_tissue_hla: 0
validation_tissue_hla_label_count_mismatches: 0
validation_tissue_hla_duplicate_label_peptides: 0
validation_tissue_hla_unique_peptide_mismatches: 0
```

Meaning:

- each `pair_id` has exactly one positive and one negative sample
- each pair uses the same `molecule_parent_uniprot_id`
- negative peptides are not reported in the target `tissue-HLA`
- for each `tissue-HLA`, positive and negative row counts are equal
- for each `tissue-HLA`, positive peptide sequences are unique
- for each `tissue-HLA`, negative peptide sequences are unique
- for each `tissue-HLA`, positive and negative unique peptide counts are equal

---

## 7. Peptide length distribution

Script:

```text
scripts/summarize_pair_peptide_lengths.py
```

Output:

```text
data/processed/iedb_tissue_specificity_pair_length_distribution.csv
```

Because we already kept only unmodified 9-mer peptides, the final paired dataset has only length 9 peptides:

```text
peptide_length,positive_count,negative_count,positive_fraction,negative_fraction
9,125649,125649,1.0,1.0
```

---

## 8. tissuePMHC standard dataset

Script:

```text
scripts/build_tissuepmhc_dataset.py
```

Input:

```text
data/processed/iedb_tissue_specificity_pairs.csv.gz
```

Output folder:

```text
data/tissuePMHC/
```

Output files:

| File | Description |
|---|---|
| `tissuePMHC_train.csv.gz` | standard training dataset |
| `tissuePMHC_test.csv.gz` | standard testing dataset |
| `tissuePMHC_summary.csv` | summary for each tissue-HLA task |
| `tissuePMHC_metadata.json` | metadata |

### 8.1 Dataset name

The dataset is named:

```text
tissuePMHC
```

### 8.2 Machine learning task

We chose task type A:

```text
Train one binary classifier for each tissue-HLA pair.
```

For example:

```text
blood + HLA-A*02:01 -> one binary classifier
lung + HLA-A*02:01 -> one binary classifier
lymph node + HLA-C*05:01 -> one binary classifier
```

Input:

```text
peptide_sequence
```

Output:

```text
label = 1 or 0
```

where:

```text
label = 1: reported in target tissue-HLA
label = 0: same UniProt and HLA, reported in other tissue, not target tissue-HLA
```

### 8.3 Why remove small tissue-HLA tasks?

Some tissue-HLA pairs have very few peptide pairs.

If a task has too few samples:

- the model cannot learn stable patterns
- test performance will be noisy
- results are not reliable

So we keep only tissue-HLA tasks with:

```text
n_pairs > 500
```

This gives each task enough training samples.

### 8.4 Train and test split

For each selected tissue-HLA task:

```text
100 positive-negative pairs are randomly selected as testing data.
All remaining pairs are used as training data.
```

Random seed:

```text
20260704
```

Why use exactly 100 test pairs per task?

- It makes all tasks have the same test size.
- Each test set has 100 positive and 100 negative samples.
- Performance can be compared fairly across tasks.
- The test set is large enough for a simple first benchmark.

### 8.5 tissuePMHC data size

Final selected tasks:

```text
selected_tissue_hla_groups: 44
```

Training data:

```text
pairs: 48,486
rows: 96,972
positive rows: 48,486
negative rows: 48,486
```

Testing data:

```text
pairs: 4,400
rows: 8,800
positive rows: 4,400
negative rows: 4,400
```

### 8.6 Split validation

Validation results:

```text
train_pairs: 48,486
test_pairs: 4,400
pair_overlap: 0

train positive: 48,486
train negative: 48,486
test positive: 4,400
test negative: 4,400

test_group_label_count_not_100: 0
train_group_label_imbalance: 0
```

Meaning:

- no `pair_id` appears in both training and testing
- training data is balanced
- testing data is balanced
- every selected tissue-HLA task has exactly 100 positive and 100 negative test samples

### 8.7 Columns in the tissuePMHC dataset

Both training and testing files have the same columns:

| Column | Meaning |
|---|---|
| `dataset` | dataset name, always `tissuePMHC` |
| `split` | `train` or `test` |
| `sample_id` | sample ID |
| `pair_id` | paired positive-negative ID |
| `label` | `1` positive, `0` negative |
| `target_tissue` | target tissue |
| `mhc_restriction` | HLA-I allele |
| `peptide_sequence` | unmodified 9-mer peptide |
| `molecule_parent_uniprot_id` | UniProt protein ID |
| `source_molecule` | direct source molecule from IEDB |
| `source_molecule_uniprot_id` | UniProt ID of source molecule, may be `NA` |
| `molecule_parent` | normalized parent protein |
| `reported_tissues_same_hla_uniprot` | tissues where this peptide was reported under the same HLA and UniProt |

### 8.8 Example selected tasks

From `data/tissuePMHC/tissuePMHC_summary.csv`:

```text
target_tissue,mhc_restriction,total_pairs_before_filter,train_pairs,test_pairs
lymph node,HLA-A*02:01,6565,6465,100
blood,HLA-A*02:01,4497,4397,100
bone,HLA-A*02:01,4354,4254,100
lymphoid,HLA-A*02:01,4016,3916,100
uterine cervix,HLA-A*02:01,1718,1618,100
lung,HLA-A*02:01,1574,1474,100
lymphoid,HLA-B*07:02,1518,1418,100
lymphoid,HLA-B*27:05,1257,1157,100
ovary,HLA-A*02:01,1233,1133,100
brain,HLA-A*02:01,1229,1129,100
```

---

## 9. Baseline machine learning models

Script:

```text
scripts/run_tissuepmhc_baselines.py
```

Input:

```text
data/tissuePMHC/tissuePMHC_train.csv.gz
data/tissuePMHC/tissuePMHC_test.csv.gz
```

Output:

```text
results/tissuePMHC_baselines/per_task_metrics.csv
results/tissuePMHC_baselines/summary_metrics.csv
results/tissuePMHC_baselines/metadata.json
```

### 9.1 Why run baseline models?

Baseline models are simple models.

They help us answer:

1. Is there a learnable signal in this dataset?
2. How hard is each tissue-HLA task?
3. What performance should future models compare against?

### 9.2 Models tested

We tested five simple models.

| Model name | Feature | Classifier |
|---|---|---|
| `onehot_logistic_regression` | 9 positions x 20 amino acid one-hot | Logistic Regression |
| `blosum62_logistic_regression` | 9 positions x 20 BLOSUM62 scores | Logistic Regression |
| `blosum62_random_forest` | BLOSUM62 | Random Forest |
| `blosum62_extra_trees` | BLOSUM62 | Extra Trees |
| `blosum62_hist_gradient_boosting` | BLOSUM62 | Histogram Gradient Boosting |

### 9.3 Why use one-hot encoding?

One-hot encoding is the simplest sequence encoding.

For a 9-mer peptide:

```text
9 positions x 20 amino acids = 180 features
```

It does not use biological similarity between amino acids.

It is a good simple baseline.

### 9.4 Why use BLOSUM62 encoding?

`BLOSUM62` describes amino acid substitution similarity.

It gives the model some biological information about amino acid similarity.

For each amino acid, we use a 20-dimensional BLOSUM62 score vector.

For a 9-mer peptide:

```text
9 positions x 20 scores = 180 features
```

### 9.5 Why test random forest and gradient boosting?

Logistic regression is linear.

Random forest and gradient boosting can model non-linear patterns.

They are still simple and easy to run.

So they are useful baseline models before using deep learning.

### 9.6 Overall baseline results

The results below are averaged over 44 tissue-HLA binary tasks.

| model | mean AUROC | median AUROC | mean AUPRC | mean ACC | mean MCC |
|---|---:|---:|---:|---:|---:|
| `onehot_logistic_regression` | `0.7558` | `0.7442` | `0.7384` | `0.6909` | `0.3841` |
| `blosum62_logistic_regression` | `0.7554` | `0.7434` | `0.7358` | `0.6893` | `0.3807` |
| `blosum62_hist_gradient_boosting` | `0.7536` | `0.7526` | `0.7431` | `0.6818` | `0.3661` |
| `blosum62_extra_trees` | `0.7450` | `0.7396` | `0.7348` | `0.6782` | `0.3606` |
| `blosum62_random_forest` | `0.7446` | `0.7379` | `0.7357` | `0.6798` | `0.3644` |

### 9.7 Interpretation

Main observations:

1. The best simple baseline is `onehot_logistic_regression`.
2. `BLOSUM62 + logistic regression` is almost the same as one-hot logistic regression.
3. Random forest does not improve performance in this first test.
4. Gradient boosting is competitive, especially for AUPRC.
5. The dataset has a real signal because AUROC is clearly higher than 0.5.
6. Different tissue-HLA tasks have different difficulty.

Good-performing examples:

| model | tissue | HLA | AUROC | AUPRC | ACC | MCC |
|---|---|---|---:|---:|---:|---:|
| `blosum62_logistic_regression` | `umbilical cord blood` | `HLA-A*02:01` | `0.8958` | `0.8939` | `0.785` | `0.571` |
| `onehot_logistic_regression` | `umbilical cord blood` | `HLA-A*02:01` | `0.8946` | `0.8994` | `0.760` | `0.520` |
| `blosum62_hist_gradient_boosting` | `lymph node` | `HLA-C*05:01` | `0.8794` | `0.8074` | `0.825` | `0.657` |

Poor-performing examples:

| model | tissue | HLA | AUROC | AUPRC | ACC | MCC |
|---|---|---|---:|---:|---:|---:|
| `onehot_logistic_regression` | `blood` | `HLA-B*07:02` | `0.6121` | `0.6168` | `0.560` | `0.120` |
| `blosum62_logistic_regression` | `blood` | `HLA-B*07:02` | `0.6161` | `0.6532` | `0.550` | `0.100` |
| `blosum62_hist_gradient_boosting` | `blood` | `HLA-B*07:02` | `0.6168` | `0.6456` | `0.570` | `0.140` |

---

## 10. Current project files

Important files and folders:

```text
data/
  raw/
    mhc_ligand_full_single_file.zip
  processed/
    iedb_human_mhci_ligands.csv.gz
    iedb_human_mhci_ligands_unique_peptide_mhc_tissue.csv.gz
    iedb_human_mhci_ligands_unique_peptide_mhc_tissue_protein.csv.gz
    iedb_human_mhci_ligands_summary.json
    iedb_tissue_summary.csv
    iedb_tissue_hla_uniprot_summary.csv
    iedb_tissue_specificity_pairs.csv.gz
    iedb_tissue_specificity_pairs_summary_9mer.csv
    iedb_tissue_specificity_pair_length_distribution.csv
  tissuePMHC/
    tissuePMHC_train.csv.gz
    tissuePMHC_test.csv.gz
    tissuePMHC_summary.csv
    tissuePMHC_metadata.json

scripts/
  extract_iedb_human_mhci_ligands.py
  summarize_tissue_hla_uniprot.py
  build_tissue_specificity_pairs.py
  summarize_pair_peptide_lengths.py
  build_tissuepmhc_dataset.py
  run_tissuepmhc_baselines.py

results/
  tissuePMHC_baselines/
    per_task_metrics.csv
    summary_metrics.csv
    metadata.json

REPORT.md
```

---

## 11. How to reproduce the results

Run the scripts in this order:

```bash
python scripts/extract_iedb_human_mhci_ligands.py
python scripts/summarize_tissue_hla_uniprot.py
python scripts/build_tissue_specificity_pairs.py --summary-output data/processed/iedb_tissue_specificity_pairs_summary_9mer.csv
python scripts/summarize_pair_peptide_lengths.py
python scripts/build_tissuepmhc_dataset.py
python scripts/run_tissuepmhc_baselines.py
```

If this file is not open in another program:

```text
data/processed/iedb_tissue_specificity_pairs_summary.csv
```

then this command can also be used:

```bash
python scripts/build_tissue_specificity_pairs.py
```

---

## 12. Important notes

### 12.1 Negative samples are not true biological negatives

A negative sample means:

```text
not reported in the target tissue-HLA
```

It does not mean:

```text
never presented in that tissue-HLA
```

This is because IEDB is based on reported experiments. A peptide may be truly presented but not yet measured or reported.

So the task should be described as:

```text
tissue-specific presentation preference prediction
```

not as absolute presentation vs non-presentation.

### 12.2 IEDB has study bias

Some tissues and HLA alleles are studied more often than others.

So the dataset may contain bias from:

- experimental design
- tissue availability
- disease focus
- HLA allele popularity
- mass spectrometry depth

This should be mentioned when using the dataset in a paper.

### 12.3 This is a clean first version

This version only uses:

- human peptides
- human host
- `HLA-I`
- four-digit HLA alleles
- positive records
- available `molecule_parent_uniprot_id`
- unmodified standard amino acid peptides
- 9-mer peptides

This makes the dataset clean and easy to use.

Future versions can include:

- 8-mer, 10-mer, 11-mer peptides
- modified peptides
- more relaxed HLA labels
- external validation data

---

## 13. Future work

### 13.1 Better single-task models

The current baseline models are simple.

Future students can try stronger models, such as:

1. `XGBoost` or `LightGBM`
2. small `CNN` models
3. amino acid physicochemical features
4. peptide embedding models
5. pretrained protein language model embeddings, such as `ESM`

These models should be compared with the current simple baselines.

### 13.2 More strict test sets

The current split uses random test pairs inside each tissue-HLA task.

Future work can make harder test sets:

1. Split by `UniProt ID`.
2. Split by peptide sequence.
3. Use external datasets.

This can test whether the model can generalize to new proteins or new peptides.

### 13.3 Multi-task binary prediction

This is the most important future direction.

Currently, we train one model for each tissue-HLA task.

This is simple, but it has a weakness:

```text
Each model can only use data from one tissue-HLA pair.
```

Many tissue-HLA pairs have limited data.

Also, different HLA alleles and tissues may share useful information.

For example:

- two HLA alleles may have similar peptide motifs
- two tissues may have similar presentation preference
- one tissue-specific pattern may appear across multiple HLA alleles

So we hope to build a multi-task binary model.

A possible model input is:

```text
peptide_sequence + tissue + HLA allele
```

The output is:

```text
probability of tissue-HLA-specific presentation preference
```

A possible neural network design is:

```text
shared peptide encoder
+ tissue embedding
+ HLA embedding
+ task-specific or shared output layer
```

The key idea is:

```text
Use information from other tissues and other HLA alleles to improve prediction for one target tissue-HLA task.
```

This may improve performance, especially for tissue-HLA tasks with fewer samples.

Important research questions:

1. Does multi-task learning perform better than single-task models?
2. Which tissue-HLA tasks benefit most from shared learning?
3. Which HLA alleles share similar tissue-specific signals?
4. Which tissues share similar presentation preference?
5. Can the model help select better tumor vaccine peptide candidates?

### 13.4 Connection to tumor vaccine design

The long-term goal is to support tumor vaccine research.

In the future, the model can be combined with:

- tumor antigen peptides
- neoantigens
- tumor expression data
- normal tissue expression data
- HLA binding prediction
- immunogenicity prediction

A possible final use case is:

```text
Given a patient HLA type and candidate tumor peptides,
predict which peptides are more likely to be presented in the target tumor tissue,
and less likely to be presented in important normal tissues.
```

This could help prioritize safer and more effective tumor vaccine candidates.

---

## 14. Summary

In this project stage, we built a clean and reproducible dataset for studying tissue-specific `HLA-I` peptide presentation.

Main outputs:

1. Clean processed IEDB human `HLA-I` 9-mer peptide data.
2. Paired positive and negative tissue-specific samples.
3. Standard `tissuePMHC` training and testing datasets.
4. Baseline model results for 44 tissue-HLA binary tasks.
5. A full set of scripts for reproduction.

Best simple baseline:

```text
model: onehot_logistic_regression
mean AUROC: 0.7558
mean AUPRC: 0.7384
mean accuracy: 0.6909
mean MCC: 0.3841
```

This result suggests that peptide sequence contains useful signal for tissue-HLA-specific presentation preference.

The next major step is to build a multi-task binary model that can share information across tissues and HLA alleles, and improve prediction for each specific tissue-HLA task.
