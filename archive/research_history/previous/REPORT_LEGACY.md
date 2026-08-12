# TissuePMHC Project Report

## 1. Project objectives

This project examines the organizational preferences of human ZXQ0QZ peptide.

The ultimate goal is to construct a clean machine learning data set, named `tissuePMHC`.

The mandate is defined as:

```text
For a given tissue-HLA pair, the possibility of a pyre being presented in the organization under the HLA equivalent gene is predicted.
```

At this stage, we have accomplished four main tasks:

1. Downloads and processes the `IEDB MHC ligand` data set.
2. The first is a pair of positive and negative pygmies.
3. Builds standard `tissuePMHC` training set and test set.
4. A number of simple baseline models were tested.

---

## Original data

Source data from `IEDB`.

Download links:

```text
https://www.iedb.org/downloader.php?file_name=doc/mhc_ligand_full_single_file.zip
```

Local file:

```text
data/raw/mhc_ligand_full_single_file.zip
```

The Zip file contains:

```text
mhc_ligand_full.csv
```

The file is large and the released CSV is about `8.8 GB`.

CSV has two rows of tableheads:

- Line 1: Columns, e.g. ZXQ0QZ, ZXQ1QZ, ZXQ2QZ, `MHC Restriction`
- Line 2: column name

The following are the important original examples:

Column group  column name  use
|---|---|---|
ZXQ0QZ  `Name`
ZXQ0QZZXQ1QZ Retain ZXQ2QZ
ZXQ0QZ  `Modified residues`  Defiguring
ZXQ0QZ  `Modifications`  Defiguring
ZXQ0QZ  `Source Organism`  Checking for  origin as human
ZXQ0QZ  `Species`  Checking for  origin as human
ZXQ0QZ  `Source Molecule`  Direct source molecule
ZXQ0QZ`Source Molecule IRI`  Source Molecular Link, sometimes UniProt
ZXQ0QZ  `Molecule Parent`  Standardized protein
ZXQ0QZ  ZXQ1QZ teleton link, usually UniProt
ZXQ0QZ  `Name`  Host species
ZXQ0QZ  `Qualitative Measurement`  Positive or negative result
ZXQ0QZ`Source Tissue` Organisational Information
ZXQ0QZ  ZXQ1QZ  HLA
ZXQ0QZZXQ1QZMHC class

---

## 3. Scripts for primary data extraction

Script:

```text
scripts/extract_iedb_human_mhci_ligands.py
```

The script reads the original ZXQ0QZ zip file and extracts the clean human ZXQ1QZ platinum record.

### 3.1 Filtering rules

To make the data sets clean and machine-friendly, we have introduced rigorous filtering.

Current filter rules:

1. Only `MHC-I` records are kept.
2. Only `Linear peptide` records are kept.
3. Only the pelicans of human origin are retained.
4. Only records of the human host are kept.
5. Only positive laboratory records are kept.
6. Only the four-digit `HLA-I` equivalent gene is retained.
7. Only records with valid `molecule_parent_uniprot_id` are maintained.
8. Only unmodified pylons are retained.
9. Only standard amino acid sequences are retained.
10. Only the peptide with a length of `9` is retained.

### 3.2 Why do you keep only the MHC-I records?

The project study ZXQ0QZ XZ X submission, therefore only:

```text
MHC Restriction / Class == I
```

This removes the `MHC-II` records and other unrelated records.

### 3.3 Why do we keep only the pebbles and the hosts?

The long-term goal is related to human oncology vaccines, so we want to retain only peptide originating from human proteins and presented in human samples.

This would avoid mixing viruses, bacteria, mice or other species with pyrophorics in data sets.

### 3.4 Why do we keep only positive records?

We want to use pelicans that have been reported as having been presented or integrated, and therefore retain the results of the experiment that meet the following conditions:

```text
Qualitative Measurement starts with Positive
```

This provides reported evidence of positive transmission.

### 3.5 Why do you keep only four-digit HLA equivalent genes?

Some IEDB records have unclear HLA names, such as:

```text
HLA-A2
HLA class I
```

These names are not precise. We keep the following genogenics:

```text
HLA-A*02:01
HLA-B*07:02
HLA-C*03:04
```

Reason:

- The four-digit equivalent gene may have different thiorium motifs.
- Machine learning models need clear genolabels.
- Undefined HLA labels may introduce noise.

### 3.6 Why do you keep records only with ZXQ0QZ?

The subsequent construction of negative samples from the same proteins as the positive ones requires stable protein IDs.

We use:

```text
molecule_parent_uniprot_id
```

Because it is more standardized and better covered than `source_molecule_uniprot_id`.

### 3.7 The distinction between ZXQ0QZ and `molecule_parent`

`source_molecule` is a direct molecular comment in IEDB.

ZXQ0QZ is a protein of the body or standardized protein.

Simple example:

```text
peptide
  comes from source_molecule
    belongs to molecule_parent
```

Recommended use:

Fields,  Meaning,  Purpose,
|---|---|---|
ZXQ0QZ  Direct source molecule  for retrospective original comment
`source_molecule_uniprot_id`  Source Molecular Link for UniProt ID  may be missing
ZXQ0QZ  Standardized protein, , more suitable for protein level analysis,
UniProt ID   ZXQ0QZ

### 3.8 Why do you keep only unmodified 9-mer peptide?

Most ZXQ0QZ formulations are short peptide, usually 8-11 amino acids.

We decided to keep the first standard data set simple and clean, so only unmodified ZXQ0QZ permunctuation is retained.

The advantages are as follows:

1. All peptide has the same length.
2. Characteristics are simple.
3. Machine learning models are easier to compare.
4. The modifier will not add additional complexity.
5. Data sets are more easily used by other students.

Example removed:

```text
FTDPRTMGY + PHOS(T6)
HEIFTDPRTMGY + OX(M10)
```

---

## 4. Processed IEDB output files

Output directory:

```text
data/processed/
```

Important documents:

Documentation
|---|---|
ZXQ0QZ  Evidence level record; one IEDB experimental record for each line
`iedb_human_mhci_ligands_unique_peptide_mhc_tissue.csv.gz`  Only `peptide-HLA-tissue` Table
ZXQ0QZ  Only table containing protein information; pairing of main input
`iedb_human_mhci_ligands_summary.json`  extract summary

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

The final processing table has the following characteristics:

- Only human `HLA-I` records
- Only four-digit HLA equivalent.
- Only unmodified 9-mer pelican
- Only records with `molecule_parent_uniprot_id`

---

## 5. Organization and HLA summary

Script:

```text
scripts/summarize_tissue_hla_uniprot.py
```

Output file:

```text
data/processed/iedb_tissue_summary.csv
data/processed/iedb_tissue_hla_uniprot_summary.csv
```

Current summary:

```text
tissues: 48
tissue_hla_pairs: 1280
```

Columns for `iedb_tissue_summary.csv`:

Columns, meanings,
|---|---|
`source_tissue`  Organization Name
`n_hla_alleles`  HLA equivalent gene count in the organization
ZXQ0QZ  Source protein count
ZXQ0QZ

Example:

```text
source_tissue,n_hla_alleles,n_molecule_parent_uniprot_ids,n_peptides
lymphoid,113,11664,81156
blood,101,13570,104559
NA,94,958,2238
lung,53,9949,33772
kidney,45,4958,8678
```

If IEDB does not provide a clear organizational information, the organization can be `NA`.

---

## 6. Positive and negative sample construction

Script:

```text
scripts/build_tissue_specificity_pairs.py
```

Enter:

```text
data/processed/iedb_human_mhci_ligands_unique_peptide_mhc_tissue_protein.csv.gz
```

Output:

```text
data/processed/iedb_tissue_specificity_pairs.csv.gz
data/processed/iedb_tissue_specificity_pairs_summary_9mer.csv
```

### 6.1 Positive sample definitions

For a `tissue-HLA-UniProt` group, the positive sample is:

```text
The platinum from the UniProt protein in target tissue, under HLA-like genetic conditions, reported
```

### 6.2 Negative sample definition

For the same `tissue-HLA-UniProt` group, negative samples are selected according to the following logic:

1. Found all positive samples in the target `tissue-HLA-UniProt` group.
2. Found all peptide from the same ZXQ0QZ protein, the same `HLA`, but reported in other organizations.
3. Removes the thong sequences already reported in the target `tissue-HLA` group.
4. The remaining pelican is the candidate negative sample.
5. Compares the number of positive and negative samples.
6. Keep a smaller group.
7. The same amount of peptide is taken from a larger group of random samples.
8. The Pyramid is a random pair.

### 6.3 Why is this negative sampling strategy used?

It's important.

Negative samples are not random peptide. It must satisfy:

- The same protein as the positive sample.
- Reported under the same HLA equivalent.
- Reported in other organizations
- Not reported in target Tissue-HLA group

This design reduced some simple deviations.

For example, models should not simply learn that a protein is common in an organization. Because positive and negative samples come from the same ZXQ0QZ protein, models are forced to learn more organizationally specific antigen preferences.

### 6.4 Sampling volume

Current pairing data:

```text
input_rows: 549,453
pairs: 125,649
paired_rows: 251,298
tissue_hla_pairs_with_pairs: 1,091
```

That is:

```text
positive samples: 125,649
negative samples: 125,649
total rows: 251,298
```

### 6.5 Twinning certification

Scripts automatically check pairing data.

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

- Every ZXQ0XZ is exactly the same as a positive and negative sample.
- Use the same ZXQ0QZ for each pair
- Negative peptide not reported in target `tissue-HLA`
- Equal number of positive and negative lines for each `tissue-HLA`
- For each ZXQ0XZ, the sample is unique
- For each ZXQ0XZ, the only negative sample sequence
- For each `tissue-HLA`, the number of platinum is equal.

---

## 7. Distribution of thong length

Script:

```text
scripts/summarize_pair_peptide_lengths.py
```

Output:

```text
data/processed/iedb_tissue_specificity_pair_length_distribution.csv
```

Since previously only 9-mer peptum had been left unmodified, the final pairing data were only peptide with a length of 9:

```text
peptide_length,positive_count,negative_count,positive_fraction,negative_fraction
9,125649,125649,1.0,1.0
```

---

## 8. TissuePMHC standard data set

Script:

```text
scripts/build_tissuepmhc_dataset.py
```

Enter:

```text
data/processed/iedb_tissue_specificity_pairs.csv.gz
```

Output directory:

```text
data/tissuePMHC/
```

Output file:

Documentation
|---|---|
`tissuePMHC_train.csv.gz`  Standard training data set
`tissuePMHC_test.csv.gz`  Standard Test Data Set
`tissuePMHC_summary.csv`  Summary of each Tissue-HLA task
`tissuePMHC_metadata.json`  Metadata

### 8.1 Name of data set

The data set is named as:

```text
tissuePMHC
```

### 8.2 Machine learning missions

Select Task Type A:

```text
Trains a second classifier for each Tissue-HLA pair.
```

For example:

```text
♪ BLood + HLA-A*02:01-> a second classifier
lung + HLA-A*02:01-> a second classifier
lymph node + HLA-C*05:01-> a second classifier
```

Enter:

```text
peptide_sequence
```

Output:

```text
label = 1 or 0
```

Of which:

```text
Label = 1: reported in target tissue-HLA
Labor = 0: From the same UniProt and HLA, reported in other organisations but not in target target
```

### 8.3 Why remove smaller samples?

Some tissue-HLA pairs are rarely paired. If the tabk sample is too few:

- Models cannot learn to stabilize.
- There's a lot of noise going on in the tests.
- The results are unreliable.

Therefore, only the tissue-HLA task that meets the following conditions is retained:

```text
n_pairs > 500
```

This gives each task enough training samples.

### 8.4 Division of training and testing sets

For each selected Tissue-HLA task:

```text
Random selection of 100 positive and negative samples as test data.
All remaining pairs are used as training data.
```

Random torrent:

```text
20260704
```

Why do each task just use 100 tests?

- All tasks have the same test set size.
- Each test set contains 100 positive and 100 negative samples.
- It's fair to compare performance.
- For a simple first benchmark, the test set is large enough.

### 8.5 tissuuePMHC Data Size

Final selection task:

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

Test data:

```text
pairs: 4,400
rows: 8,800
positive rows: 4,400
negative rows: 4,400
```

### 8.6 Classification of Validation

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

- No `pair_id` at the same time as the training and testing set.
- Training data balance
- Test Data Balance
- 100 positive and 100 negative samples per selected Tissue-HLA task

### Columns in 8.7 tissuuePMHC datasets

The training and testing documents have the same columns:

Columns, meanings,
|---|---|
ZXQ0QZ  Dataset name, constant ZXQ1QZ
ZXQ0QZZXQ1QZ or `test`
`sample_id` Specific ID
`pair_id`  Positive-negative ID
ZXQ0QZ  ZXQ1QZ for positive samples and `0` for negative samples
`target_tissue`  Target organization
ZXQ0QZ  HLA-I gene
ZXQ0QZ  unmodified 9-mer
ZXQ0QZUniProt Protein ID
Direct-source molecule in ZXQ0QZ  IEDB
UniProt ID of the ZXQ0QZ  Source molecule, possibly ZXQ1QZ
`molecule_parent`  Standardized protein
ZXQ0QZ  The tissue reported under the same HLA and UniProt conditions

### 8.8 Example of the selected task

From ZXQ0QZ:

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

## 9. Baseline machine learning model

Script:

```text
scripts/run_tissuepmhc_baselines.py
```

Enter:

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

### 9.1 Why are baseline models in operation?

Baseline models are simple models that can help to answer:

1. Are there learning signs for data concentration?
2. How difficult are the different tissue-HLA task?
3. What performance should future models compare?

### 9.2 Models tested

Five simple models were tested:

Model name  Characteristics  Classifier
|---|---|---|
ZXQ0QZ  9 positions x one-hot Ligistic Restatement  20 amino acids
ZXQ0QZ  9 positions x 20 BLOSUM62 scores Logistic Repression
| `blosum62_random_forest` | BLOSUM62 | Random Forest |
| `blosum62_extra_trees` | BLOSUM62 | Extra Trees |
| `blosum62_hist_gradient_boosting` | BLOSUM62 | Histogram Gradient Boosting |

### 9.3 Why use one-hot encoding?

One-hot is the simplest sequence encoding. For a 9-mer pelican:

```text
9 positions x 20 amino acids = 180 features
```

It does not use biological similarities between amino acids, and is a good simple baseline.

### 9.4 Why BLOSUM62 encoding?

ZXQ0QZ describes amino acid substitution similarities that can provide some information on amino acid biological similarities to models.

For each amino acid, use a 20-dimensional BLOSUM62 fraction vector. For a 9-mer pelican:

```text
9 positions x 20 scores = 180 features
```

### 9.5 Why test random forest and gradient?

Logistic restatement is a linear model, a model of non-linear models, of a grandom forest and a gradient boosting.

They remain simple and easy to operate and are therefore valuable baseline models before using in-depth learning.

### 9.6 Overall baseline results

The following table shows the average of 44 tissue-HLA II categories.

means AUROC  median AUROC  AUPRC  means ACC  means MCC
|---|---:|---:|---:|---:|---:|
| `onehot_logistic_regression` | `0.7558` | `0.7442` | `0.7384` | `0.6909` | `0.3841` |
| `blosum62_logistic_regression` | `0.7554` | `0.7434` | `0.7358` | `0.6893` | `0.3807` |
| `blosum62_hist_gradient_boosting` | `0.7536` | `0.7526` | `0.7431` | `0.6818` | `0.3661` |
| `blosum62_extra_trees` | `0.7450` | `0.7396` | `0.7348` | `0.6782` | `0.3606` |
| `blosum62_random_forest` | `0.7446` | `0.7379` | `0.7357` | `0.6798` | `0.3644` |

### 9.7 Explanation of the results

Main observations:

1. The best simple baseline is `onehot_logistic_regression`.
2. ZXQ0QZ is almost identical to one-hot blog review.
3. In this preliminary test, the Random forest did not enhance performance.
4. Gradient boosting is competitive, especially on AUPRC.
5. The data set has a real signal because AUROC is significantly higher than 0.5.
6. Different tissue-HLA task is different.

For example, better performance:

Models, tissues, HLAAUROCAUPPCACCMCC
|---|---|---|---:|---:|---:|---:|
| `blosum62_logistic_regression` | `umbilical cord blood` | `HLA-A*02:01` | `0.8958` | `0.8939` | `0.785` | `0.571` |
| `onehot_logistic_regression` | `umbilical cord blood` | `HLA-A*02:01` | `0.8946` | `0.8994` | `0.760` | `0.520` |
| `blosum62_hist_gradient_boosting` | `lymph node` | `HLA-C*05:01` | `0.8794` | `0.8074` | `0.825` | `0.657` |

Examples of poor performance:

Models, tissues, HLAAUROCAUPPCACCMCC
|---|---|---|---:|---:|---:|---:|
| `onehot_logistic_regression` | `blood` | `HLA-B*07:02` | `0.6121` | `0.6168` | `0.560` | `0.120` |
| `blosum62_logistic_regression` | `blood` | `HLA-B*07:02` | `0.6161` | `0.6532` | `0.550` | `0.100` |
| `blosum62_hist_gradient_boosting` | `blood` | `HLA-B*07:02` | `0.6168` | `0.6456` | `0.570` | `0.140` |

---

## 10. Current project document

Important files and directories:

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

## 11. How to replicate results

Run scripts in the following order:

```bash
python scripts/extract_iedb_human_mhci_ligands.py
python scripts/summarize_tissue_hla_uniprot.py
python scripts/build_tissue_specificity_pairs.py --summary-output data/processed/iedb_tissue_specificity_pairs_summary_9mer.csv
python scripts/summarize_pair_peptide_lengths.py
python scripts/build_tissuepmhc_dataset.py
python scripts/run_tissuepmhc_baselines.py
```

If the following documents are not opened in other proceedings:

```text
data/processed/iedb_tissue_specificity_pairs_summary.csv
```

You can also use the following command:

```bash
python scripts/build_tissue_specificity_pairs.py
```

---

## 12. Key notes

### 12.1 Negative samples are not really biological negative samples

Negative samples indicate:

```text
Not reported in target Tissue-HLA
```

It does not mean:

```text
Never filed in the Tissue-HLA
```

This is because IEDB is based on reported experiments. A pyromium may actually occur but has not yet been measured or reported.

The mandate should therefore be described as:

```text
Organisational specificity proxies
```

The government has been trying to make a difference between the two.

### 12.2 IEDB Study deviations

Some organizations and HLA-like genes are studied more frequently, so data sets may contain deviations from:

- Experimental design
- Organizational accessibility
- Focus of disease research
- The extent of the hotness of HLA equivalent genes
- Mass spectrometry depth

This should be stated when using the data set in the paper.

### 12.3 This is a clean initial version.

This version only uses:

- Human pea.
- The human host.
- `HLA-I`
- Four-digit HLA equivalent gene
- Positive records
- Available `molecule_parent_uniprot_id`
- Unmodified standard amino acid pylon
- 9-mer peptide

This keeps the data sets clean and easy to use.

Future versions could include:

- 8-mer, 10-mer, 11-mer peptide
- Modified Permium
- A lighter HLA tag
- External validation of data

---

## 13. Follow-up

### 13.1 Better single-mission model

The baseline model is currently simpler. The following can be tried for stronger models, for example:

1. ZXQ0QZ or ZXQ1QZ
2. Small `CNN` model
3. Amino-acidated properties
4. Pyramid Emplacement Model
5. Pre-training protein language model embedded, for example, ZXQ0QZ

These models should be compared with the current simple baseline.

### 13.2 More stringent test sets

The current division of the test randomly sampled in each Tissue-HLA task. The following can build a more difficult test set:

1. Distribute by `UniProt ID`.
2. Distribute by sequence of thongs.
3. Use external data sets.

This would test the size of the model to new proteins or pyres.

### 13.3 Multi-mission II Classification Projections

That is the most important follow-up direction.

Currently, a model is being trained for each Tissue-HLA task. This is simple, but there is one weakness:

```text
Each model can only use one Tissue-HLA pair of data.
```

Many of the tissue-HLA pairs have limited data; useful information may also be shared between different HLA-like genes and tissues.

For example:

- Two HLA equivalent genes may have similar pelican motifs
- The two organizations may have similar submissions preferences
- Some tissue-specific model may occur across multiple HLA equivalent genes.

It is therefore hoped that the multi-task II classification model will be constructed.

One possible model input is:

```text
peptide_sequence + tissue + HLA allele
```

Output is:

```text
Organisation-HLA Probability of Prevailing Speciality
```

A possible neural network is designed to:

```text
Shared pylon encoder
+ Organisation
+ HLA Embedded
+ tab-specific or shared output layer
```

Core idea:

```text
Improved prediction of a target Tissue-HLA pair using information from other organizations and other HLA-like genes.
```

Key research issues:

1. Is multitasking better than single-task models?
2. Which of the Tissue-HLA task benefits most from shared learning?
3. Which HLA equivalent genes share similar organospecific signals?
4. Which organizations share similar submissions preferences?
5. Can models help to select better candidate for acne vaccine?

### 13.4 Links to Oncology Design

The long-term objective is to support oncological vaccine research.

In the future, models can be combined with the following information:

- Tumor antigenium.
- New antigens.
- Oncological expression data
- Organisation
- HLA combined with projections
- Immunization projection

A possible end-use scenario:

```text
The patient is given HLA specs and candidate for tumor pelicans.
The project predicts which thongs are more likely to be present in the target oncology tissue.
The government has been working on the issue of the right to education, and it is less likely to be presented in important normal organizations.
```

This has helped to prioritize the selection of safer and more effective candidate for oncology vaccine.

---

## Summary

During the project phase, we constructed a clean, repossible data set for the study of organizational specificity ZXQ0QZ platinum.

Main outputs:

1. Clean, processed human ZXQ0QZ 9-mer IEDB data.
2. Positive and negative samples of the specific properties of paired tissues.
3. Standard `tissuePMHC` training and testing data sets.
4. 44 baseline model results for Tissue-HLA II categories.
5. A complete repossible script.

Best simple baseline:

```text
model: onehot_logistic_regression
mean AUROC: 0.7558
mean AUPRC: 0.7384
mean accuracy: 0.6909
mean MCC: 0.3841
```

The results showed that the thong sequence contained a useful signal for the tissue-HLA specificity.

The next major task is to construct multitask II classification models that allow information to be shared between tissues and HLA-like genes and to enhance the predictive effect of each specific Tissue-HLA task.
