# tissuePMHC

**tissuePMHC** is a research codebase for studying tissue-conditioned MHC class I
peptide presentation. Given a target tissue and an MHC restriction, the project
tests whether peptide sequence contains information for ranking a
recorded-positive peptide above a biologically matched comparison peptide.

This repository contains the frozen Human and Mouse benchmarks used in the
manuscript, data-processing and audit scripts, traditional and neural models,
external-predictor controls, interpretation analyses, statistical outputs, and
the complete LaTeX manuscript source.

> **Primary manuscript dataset:** the paper reports the occurrence-equal
> (`occurrence_equal`, also called occurrence-matched) benchmark described
> below. The larger Phase 7 min200 and standard Mouse datasets are earlier
> research benchmarks and must not be mixed with the manuscript results.

## Manuscript benchmark at a glance

| Species | Tasks | Tissues | MHC restrictions | Train pairs | Fixed-test pairs |
|---|---:|---:|---:|---:|---:|
| Human | 77 | 13 | 29 HLA alleles | 21,489 | 3,850 |
| Mouse | 11 | 4 | 4 H-2 restrictions | 2,089 | 550 |

Every task is a target tissue-MHC combination. Each task contributes exactly
50 complete positive-pseudo-negative pairs to the fixed test set; all remaining
pairs form the training set. The split is frozen with random state `20260704`,
and final neural-model runs use seeds `20260704`, `20260705`, and `20260706`.

The primary data files are:

```text
data/humanPMHC_occurence_equal_dataset/
|-- humanPMHC_train.csv.gz
`-- humanPMHC_test.csv.gz

data/mousePMHC_occurence_equal_dataset/
|-- mousePMHC_train.csv.gz
`-- mousePMHC_test.csv.gz
```

The historical directory spelling `occurence` is intentional and is retained
because existing scripts depend on it.

## Why occurrence equality matters

For a peptide `p`, MHC restriction `m`, and parent protein `u`, define
`c(p, m, u)` as the number of tissues with positive source-record evidence for
that peptide under the same MHC restriction and parent protein.

Each `pair_id` contains two rows:

- a recorded-positive peptide (`label = 1`) in the target tissue;
- a pseudo-negative peptide (`label = 0`) with positive evidence elsewhere but
  not in the target tissue.

The two peptides are matched on all of the following:

```text
target tissue
MHC restriction
parent UniProt protein
positive tissue-occurrence count, c(p, m, u)
```

Equivalently, the pair must satisfy:

```text
target_tissue+  = target_tissue-
MHC+            = MHC-
parent_protein+ = parent_protein-
c(peptide+, MHC, parent_protein) = c(peptide-, MHC, parent_protein)
```

Without the final equality, a model could exploit a reporting-frequency
shortcut: peptides observed in more tissues overall might be easier to label as
positive without learning tissue-conditioned sequence preferences. Occurrence
matching removes this specific shortcut.

It does **not** eliminate every source of observational bias. Study, laboratory,
donor, batch, detection depth, tissue coverage, and MHC coverage may still
affect the data. A pseudo-negative means that the peptide was not reported in
the target context; it is not a confirmed biological non-presentation event.

## Quick start

Python 3.10 or newer is recommended. Install the PyTorch build appropriate for
your CPU or CUDA environment, then install the project dependencies.

### Windows PowerShell

```powershell
git clone https://github.com/AmamiyaHitomi/tissuePMHC.git
cd tissuePMHC

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional testing and external-tool dependencies can be installed with:

```powershell
python -m pip install -r requirements-optional.txt
```

Run all commands below from the repository root unless a section explicitly
changes directory. Project scripts resolve paths from their own locations and
do not require a manually configured `PYTHONPATH`.

## Verify the manuscript datasets

Run the occurrence-balancing audit before training or evaluating models:

```powershell
python scripts/audit_occurrence_balancing.py
```

The audit verifies:

- exactly two rows and one row per label in every pair;
- identical tissue, MHC restriction, and parent protein within each pair;
- identical `presentation_tissue_count` within each pair;
- identical occurrence-count distributions between labels in every task;
- equal label counts and occurrence sums; and
- exactly 50 fixed-test pairs per retained task.

The committed machine-readable report is
[`extra3/occurrence_balancing_audit.json`](extra3/occurrence_balancing_audit.json).
It passes for both species: every mismatch count is zero, and an
occurrence-count-only score has task AUROC `0.5000` in the train, test, and
combined partitions.

## Reproduce the manuscript experiments

Human and Mouse workflows are kept in separate namespaces and refuse legacy
benchmark inputs.

### Human occurrence-equal experiments

Preview the planned runs without training:

```powershell
python extra_occurrence_equal_dataset/run_v7_remaining_experiments.py --dry-run --device auto
```

Run or resume the missing experiments:

```powershell
python extra_occurrence_equal_dataset/run_v7_remaining_experiments.py --device auto
```

Aggregate the completed manuscript results:

```powershell
python extra_occurrence_equal_dataset/aggregate_v7_paper_results.py
```

See
[`extra_occurrence_equal_dataset/README.md`](extra_occurrence_equal_dataset/README.md)
for the frozen protocol, individual experiment entry points, and output layout.

### Mouse occurrence-equal experiments

Preview, run, and aggregate the Mouse experiments with:

```powershell
python extra_mouse_occurrence_equal_dataset/run_v7_remaining_experiments.py --dry-run --device auto
python extra_mouse_occurrence_equal_dataset/run_v7_remaining_experiments.py --device auto
python extra_mouse_occurrence_equal_dataset/aggregate_v7_paper_results.py
```

See
[`extra_mouse_occurrence_equal_dataset/README.md`](extra_mouse_occurrence_equal_dataset/README.md)
for the Mouse-specific protocol and provenance notes.

Training programs report per-epoch, per-seed, and total runtime information.
Generated result directories and model checkpoints are excluded from Git by
default.

## Data provenance

Compact benchmark and processed files are included under `data/`. The following
large IEDB source files are intentionally omitted from GitHub:

```text
data/raw/mhc_ligand_full_single_file.zip
data/raw/mhc_ligand_full.csv
```

Download the official IEDB MHC ligand export with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_iedb_mouse_source.ps1
```

The official export page, direct-download URL, manuscript snapshot date,
SHA-256 checksum, license, attribution, and manual verification commands are
documented in [`data/README.md`](data/README.md).

Important data locations are:

| Directory | Purpose |
|---|---|
| `data/humanPMHC_occurence_equal_dataset/` | Primary Human manuscript benchmark |
| `data/mousePMHC_occurence_equal_dataset/` | Primary Mouse manuscript benchmark |
| `data/processed/` | Processed IEDB-derived evidence, pairing, and MHC sequence tables |
| `data/expression/` | Expression resources and provenance metadata |
| `data/tissuePMHC_phase7_min200/` | Earlier 157-task Human benchmark |
| `data/mousePMHC/` | Earlier 24-task Mouse benchmark |
| `data/tissuePMHC/` | Earlier 44-task Human benchmark |
| `data/humanPMHC_premium/` | Human premium-data extension |

## Repository structure

```text
tissuePMHC/
|-- data/                                  Frozen and processed datasets
|-- scripts/                               Data, training, evaluation, and plotting tools
|-- protocols/                             Frozen experiment protocols
|-- extra_occurrence_equal_dataset/        Human manuscript experiments
|-- extra_mouse_occurrence_equal_dataset/  Mouse manuscript experiments
|-- extra3/                                Occurrence audit and related analyses
|-- final_phase/                           Final statistics and reproducibility audits
|-- paper/tissuePMHC_latex_v9/             Current manuscript and supplement
|-- extra1/                                External and generalization research workflows
|-- extra2/                                Earlier Human and cross-species workflows
|-- extra_premium/                         Human premium-data experiments
|-- archive/                               Research history and manuscript versions v3-v8
|-- requirements.txt                       Core Python dependencies
`-- requirements-optional.txt              Optional dependencies
```

## Manuscript source

The current manuscript is located in `paper/tissuePMHC_latex_v9/`:

- `main.tex`: main manuscript;
- `supplementary_main.tex`: supplementary material;
- `sections/`: manuscript and supplementary sections;
- `references/`: bibliography files;
- `figures/`: current figures;
- `scripts/`: manuscript figure-generation scripts.

With a complete TeX Live or MiKTeX installation:

```powershell
cd paper/tissuePMHC_latex_v9
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary_main.tex
```

The Data and Code Availability statement points to this repository:
<https://github.com/AmamiyaHitomi/tissuePMHC>.

## Additional research workflows

The workflows in this section belong to earlier or supplementary benchmarks;
they are retained for research history and must not be used as sources for the
primary occurrence-equal manuscript results.

### Earlier Human benchmarks

Build the Phase 7 min200 benchmark and inspect representative model entry
points with:

```powershell
python scripts/build_human_dataset_min200.py --help
python scripts/run_e2_shared_heads.py --help
python scripts/run_e14_auxiliary_soft_ensemble.py --help
python scripts/run_e17_seed_ensemble.py --help
python scripts/run_e26_all_in_one.py --help
python scripts/run_e29_multikernel_cnn_oof.py --help
python scripts/run_e31_peptide_disjoint_oof.py --help
```

### Earlier Mouse benchmarks

Representative entry points are:

```powershell
python scripts/run_mousepmhc_phase3_e0_oof.py --help
python scripts/run_mousepmhc_phase4_e15_five_seed_confirmation.py --help
python scripts/run_mousepmhc_phase6_e33_peptide_disjoint_oof.py --help
```

### External predictors

MHCflurry and NetMHCpan are used only for their corresponding control
experiments and are not required for core tissuePMHC training. Local paths can
be configured with environment variables:

```powershell
$env:MHCFLURRY_EXECUTABLE = "C:\path\to\mhcflurry-predict.exe"
$env:MHCFLURRY_MODELS_DIR = "C:\path\to\models_class1_presentation\models"
$env:NETMHCPAN_EXECUTABLE = "/path/inside/wsl/netMHCpan"
```

NetMHCpan and its model assets are governed by their official license and are
not redistributed in this repository.

### Legacy final-phase audit

```powershell
python final_phase/run_all.py
```

This non-training workflow consumes existing frozen predictions and performs
PairAcc calculation, fold and provenance checks, statistical tests, figure
generation, and dataset-card collection. It reports missing input paths
explicitly.

## Validation and repository hygiene

Run the automated tests and a non-training syntax check with:

```powershell
python -m pytest
python -m compileall -q scripts final_phase extra1 extra2 extra_premium `
  extra_occurrence_equal_dataset extra_mouse_occurrence_equal_dataset
```

Before publishing changes, inspect tracked and ignored files:

```powershell
git status --short
git status --ignored --short
```

Do not commit credentials, local environments, generated results, model
checkpoints, third-party runtimes, or the omitted IEDB source archive.

## Reproducibility notes

- Keep benchmark identifiers and frozen splits consistent across training,
  validation, and testing.
- Never mix standard, min200, premium, and occurrence-equal datasets or results.
- Preserve pair membership when splitting, evaluating, and aggregating data.
- Record parameters, seeds, folds, epoch/seed/total runtime, software versions,
  hardware information, and input-file checksums for formal experiments.
- Repeated GPU runs may not be bitwise deterministic even with fixed seeds;
  report CUDA, cuDNN, and hardware configuration.
- Human and Mouse differ in task inventory and coverage; raw performance
  differences do not establish a species effect.
- Material under `archive/` documents project development and is not guaranteed
  to run as a current experiment entry point.

## Citation

The manuscript is currently under submission. Citation metadata and a paper
DOI will be added when available.

## License

TissuePMHC source code and software are available for academic and
non-commercial use under the PolyForm Noncommercial License 1.0.0. For
commercial licensing, please contact Jia Meng at jia.meng@xjtlu.edu.cn.
Third-party data and software remain subject to their original licenses.
