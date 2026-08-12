# tissuePMHC

**tissuePMHC** is a research codebase for studying tissue-conditioned MHC class I
peptide presentation. It combines peptide sequence, tissue context, and MHC
information to evaluate whether a peptide reported in one tissue can be ranked
above a matched peptide observed under the same source-protein and MHC context
in another tissue.

The repository contains data-processing pipelines, paired benchmark datasets,
traditional and neural baselines, multitask models, strict-generalization
evaluations, external-predictor comparisons, model-interpretation analyses,
statistical audits, and the LaTeX source for the accompanying manuscript.

## Current benchmarks

| Benchmark | Species | Tasks | Training rows | Test rows |
|---|---|---:|---:|---:|
| `tissuePMHC_phase7_min200` | Human (tissue-HLA) | 157 | 147,596 | 31,400 |
| `mousePMHC` | Mouse (tissue-H2) | 24 | 13,532 | 4,800 |

Each positive-negative pair shares its source protein and MHC condition. The
positive peptide was reported under the target tissue-MHC condition; its paired
negative was reported in another tissue under the same source-protein and MHC
context but not in the target tissue. Primary evaluations include AUROC, AUPRC,
MCC, accuracy, and within-pair ordering accuracy (PairAcc).

The benchmark labels are observational and should not be interpreted as
confirmed biological presentation or non-presentation in every context.

## Occurrence-equal (`occurrence_equal`) benchmark (primary manuscript analysis)

The accompanying manuscript uses the **occurrence-equal** (also described as
**occurrence-matched**) Human and Mouse benchmarks for its primary analyses.
This is an important additional control beyond matching the target tissue, MHC
restriction, and parent protein.

For a peptide `p`, MHC restriction `m`, and parent protein `u`, let
`c(p, m, u)` denote the number of tissues in which the source records contain
positive presentation evidence for that peptide under the same MHC and parent
protein. Every `pair_id` in the occurrence-equal benchmark contains:

- one recorded-positive peptide (`label = 1`) in the target tissue;
- one pseudo-negative peptide (`label = 0`) that has positive evidence in one
  or more other tissues but not in the target tissue;
- the same target tissue, MHC restriction, and parent UniProt protein for both
  peptides; and
- exactly the same positive tissue-occurrence count,
  `presentation_tissue_count`, for both peptides.

In compact form, each pair satisfies

```text
target_tissue+ = target_tissue-
MHC+           = MHC-
parent_protein+ = parent_protein-
c(peptide+, MHC, parent_protein) = c(peptide-, MHC, parent_protein)
```

### Why occurrence matching matters

Without the final equality, a model could exploit a simple reporting-frequency
shortcut: peptides observed in more tissues overall could be easier to classify
as positives, even without learning tissue-conditioned sequence preferences.
Occurrence matching removes this specific shortcut. It does **not** remove all
observational bias; study, laboratory, donor, batch, detection-depth, tissue,
and MHC-coverage effects may remain. A pseudo-negative is therefore an
unreported peptide in the target context, not a confirmed biological negative.

### Frozen datasets used in the manuscript

| Species | Dataset files | Tasks | Tissues | MHC restrictions | Train pairs | Fixed-test pairs |
|---|---|---:|---:|---:|---:|---:|
| Human | `data/humanPMHC_occurence_equal_dataset/` | 77 | 13 | 29 | 21,489 | 3,850 |
| Mouse | `data/mousePMHC_occurence_equal_dataset/` | 11 | 4 | 4 | 2,089 | 550 |

Each retained tissue-MHC task contributes exactly 50 complete pairs to the
fixed test set; all remaining pairs form the training set. The fixed split uses
random state `20260704`, and final neural-model runs use seeds `20260704`,
`20260705`, and `20260706`. Human and Mouse differ in task inventory and data
coverage, so their raw results should not be interpreted as a controlled
species comparison.

### Verify the occurrence-equal invariant

Run the repository audit from the project root:

```powershell
python scripts/audit_occurrence_balancing.py
```

The audit checks pair size, one row per label, task/MHC/parent agreement,
pairwise occurrence-count equality, label-wise occurrence distributions, and
the 50-pair fixed-test allocation. Its machine-readable report is
[`extra3/occurrence_balancing_audit.json`](extra3/occurrence_balancing_audit.json).
The committed audit passes for both species: every mismatch count is zero, and
an occurrence-count-only score has task AUROC `0.5000` for every train, test,
and combined partition.

### Reproduce the occurrence-equal experiments

The species-specific experiment code and detailed commands are documented in:

- [`extra_occurrence_equal_dataset/README_zh.md`](extra_occurrence_equal_dataset/README_zh.md)
  for Human;
- [`extra_mouse_occurrence_equal_dataset/README_zh.md`](extra_mouse_occurrence_equal_dataset/README_zh.md)
  for Mouse.

These runners are deliberately isolated from legacy benchmark results. Do not
mix standard, min200, premium, or occurrence-equal data, predictions, or model
summaries in the same analysis.

## Repository structure

```text
tissuePMHC/
|-- data/                                  Compact and processed datasets
|-- scripts/                               Data, training, evaluation, and plotting entry points
|-- protocols/                             Frozen experiment protocols
|-- final_phase/                           Final statistics and reproducibility audits
|-- extra1/                                External-predictor and strict-generalization workflows
|-- extra2/                                Human 157-task and cross-species experiments
|-- extra_premium/                         Human premium-data experiments
|-- extra_occurrence_equal_dataset/        Human occurrence-matched experiments
|-- extra_mouse_occurrence_equal_dataset/  Mouse occurrence-matched experiments
|-- paper/tissuePMHC_latex_v9/             Current manuscript and supplement
|-- archive/                               Research history and manuscript versions v3-v8
|-- requirements.txt                       Core Python dependencies
`-- requirements-optional.txt              Optional testing and external-tool dependencies
```

Generated `results/`, model checkpoints, local environments, and third-party
runtimes are excluded from Git by default.

## Installation

Python 3.10 or newer is recommended. Install the PyTorch build appropriate for
your CPU/CUDA environment, then install the project dependencies.

### Windows PowerShell

```powershell
git clone https://github.com/AmamiyaHitomi/tissuePMHC.git
cd tissuePMHC

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional dependencies used by tests and selected external tools can be
installed with:

```powershell
python -m pip install -r requirements-optional.txt
```

Run commands from the repository root. The scripts resolve project paths from
their own locations and do not require a manually configured `PYTHONPATH`.

## Data

Compact benchmark and processed data files are included under `data/`. Two
large IEDB source files are intentionally omitted:

```text
data/raw/mhc_ligand_full_single_file.zip
data/raw/mhc_ligand_full.csv
```

Download the official IEDB MHC ligand export with the bundled script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_iedb_mouse_source.ps1
```

The official export page and direct-download URL, the manuscript snapshot date,
SHA-256 checksum, license, and verification instructions are documented in
[`data/README.md`](data/README.md).
Important dataset locations:

| Directory | Purpose |
|---|---|
| `data/tissuePMHC_phase7_min200/` | Current 157-task Human benchmark |
| `data/mousePMHC/` | Current 24-task Mouse benchmark |
| `data/tissuePMHC/` | Earlier 44-task Human benchmark |
| `data/humanPMHC_premium/` | Human premium-data extension |
| `data/humanPMHC_occurence_equal_dataset/` | Human occurrence-matched dataset |
| `data/mousePMHC_occurence_equal_dataset/` | Mouse occurrence-matched dataset |
| `data/processed/` | Processed IEDB-derived evidence and pairing tables |
| `data/expression/` | Expression resources and provenance metadata |

The historical directory spelling `occurence` is retained because existing
scripts depend on it. Do not rename those directories without updating every
dependent path.

## Main Human workflow

### Build the Phase 7 min200 benchmark

```powershell
python scripts/build_human_dataset_min200.py --help
python scripts/build_human_dataset_min200.py
```

This entry point applies the `total_pairs > 200` inclusion rule and writes the
Human benchmark to `data/tissuePMHC_phase7_min200/`.

### Inspect and run model entry points

```powershell
python scripts/run_e2_shared_heads.py --help
python scripts/run_e14_auxiliary_soft_ensemble.py --help
python scripts/run_e17_seed_ensemble.py --help
python scripts/run_e26_all_in_one.py --help
python scripts/run_e29_multikernel_cnn_oof.py --help
```

Strict unseen-peptide evaluation is available through:

```powershell
python scripts/run_e31_peptide_disjoint_oof.py --help
```

Training programs record epoch-, seed-, and total-runtime information for
formal runs. Result directories are not version-controlled by default.

## Mouse workflow

Mouse experiments are implemented by the `scripts/run_mousepmhc_*.py` entry
points. Representative commands are:

```powershell
python scripts/run_mousepmhc_phase3_e0_oof.py --help
python scripts/run_mousepmhc_phase4_e15_five_seed_confirmation.py --help
python scripts/run_mousepmhc_phase6_e33_peptide_disjoint_oof.py --help
```

Human and Mouse predictions, fold assignments, and candidate configurations
must remain in their species- and benchmark-specific namespaces.

## External predictors

MHCflurry and NetMHCpan are used only in the corresponding comparison
experiments; they are not required for the core tissuePMHC training workflow.
Local paths can be configured with environment variables:

```powershell
$env:MHCFLURRY_EXECUTABLE = "C:\path\to\mhcflurry-predict.exe"
$env:MHCFLURRY_MODELS_DIR = "C:\path\to\models_class1_presentation\models"
$env:NETMHCPAN_EXECUTABLE = "/path/inside/wsl/netMHCpan"
```

NetMHCpan and its model assets are governed by their official license and are
not redistributed in this repository.

## Final analysis and audits

The final-phase workflow consumes existing frozen predictions and performs
PairAcc calculation, matched-fold auditing, parent-protein overlap analysis,
provenance checks, statistical tests, supplementary-table generation,
visualization, reproducibility collection, and dataset-card generation:

```powershell
python final_phase/run_all.py
```

This workflow does not train models. It requires the expected prediction files
and reports any missing input path explicitly.

## Manuscript

The current manuscript is located in `paper/tissuePMHC_latex_v9/`:

- `main.tex`: main manuscript;
- `supplementary_main.tex`: supplementary material;
- `sections/`: manuscript and supplementary sections;
- `references/`: reference files;
- `figures/`: current figures;
- `scripts/`: manuscript figure-generation scripts.

With a complete TeX Live or MiKTeX installation:

```powershell
cd paper/tissuePMHC_latex_v9
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary_main.tex
```

The manuscript's Data and Code Availability statement points to this repository:
<https://github.com/AmamiyaHitomi/tissuePMHC>.

## Validation

Run the available automated tests:

```powershell
python -m pytest
```

Run a non-training syntax check across the principal Python modules:

```powershell
python -m compileall -q scripts final_phase extra1 extra2 extra_premium `
  extra_occurrence_equal_dataset extra_mouse_occurrence_equal_dataset
```

Before publishing changes, inspect both tracked and ignored files:

```powershell
git status --short
git status --ignored --short
```

Do not commit credentials, local environments, generated result trees,
third-party runtimes, or model checkpoints.

## Reproducibility notes

- Keep benchmark identifiers and frozen splits consistent across training,
  validation, and testing.
- Do not mix results from the standard, min200, premium, and
  occurrence-matched benchmarks.
- Record parameters, seeds, folds, epoch/seed/total runtime, software versions,
  hardware information, and input-file checksums for formal experiments.
- Repeated GPU runs may not be bitwise deterministic even with fixed seeds;
  report CUDA/cuDNN and hardware configuration.
- The historical material in `archive/` documents project development and is
  not guaranteed to run as a current experiment entry point.

## Citation and license

The manuscript is currently under submission. Citation metadata and a paper
DOI will be added after they become available.

This repository does not yet include a project-wide software license. Until a
license is added, copyright law reserves reuse and redistribution rights to the
authors. Third-party datasets and tools remain subject to their original
licenses, including the attribution requirements documented in
[`data/README.md`](data/README.md).
