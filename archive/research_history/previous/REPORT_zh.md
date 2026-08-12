# tissuePMHC 项目报告

## 1. 项目目标

本项目研究人类 `HLA-I` 肽的组织特异性呈递偏好。

最终目标是构建一个干净的机器学习数据集，命名为 `tissuePMHC`。

任务定义为：

```text
对于一个给定的组织-HLA 配对，预测某个肽是否可能在该 HLA 等位基因条件下呈递于该组织中。
```

在本阶段，我们完成了四项主要工作：

1. 下载并处理 `IEDB MHC ligand` 数据集。
2. 构建成对的正、负肽样本。
3. 构建标准 `tissuePMHC` 训练集与测试集。
4. 测试若干简单的基线模型。

---

## 2. 原始数据

原始数据来自 `IEDB`。

下载链接：

```text
https://www.iedb.org/downloader.php?file_name=doc/mhc_ligand_full_single_file.zip
```

本地文件：

```text
data/raw/mhc_ligand_full_single_file.zip
```

该 zip 文件包含：

```text
mhc_ligand_full.csv
```

该文件很大，解压后的 CSV 约为 `8.8 GB`。

CSV 有两行表头：

- 第 1 行：列分组，例如 `Epitope`、`Host`、`Assay`、`MHC Restriction`
- 第 2 行：列名称

重要的原始列如下：

| 列分组 | 列名称 | 用途 |
|---|---|---|
| `Epitope` | `Name` | 肽序列 |
| `Epitope` | `Object Type` | 仅保留 `Linear peptide` |
| `Epitope` | `Modified residues` | 去除修饰肽 |
| `Epitope` | `Modifications` | 去除修饰肽 |
| `Epitope` | `Source Organism` | 检查肽来源是否为人类 |
| `Epitope` | `Species` | 检查肽来源是否为人类 |
| `Epitope` | `Source Molecule` | 直接来源分子 |
| `Epitope` | `Source Molecule IRI` | 来源分子链接，有时为 UniProt |
| `Epitope` | `Molecule Parent` | 标准化后的母体蛋白 |
| `Epitope` | `Molecule Parent IRI` | 母体蛋白链接，通常为 UniProt |
| `Host` | `Name` | 宿主物种 |
| `Assay` | `Qualitative Measurement` | 阳性或阴性结果 |
| `Antigen Presenting Cell` | `Source Tissue` | 组织信息 |
| `MHC Restriction` | `Name` | HLA 等位基因 |
| `MHC Restriction` | `Class` | MHC 类别 |

---

## 3. 主要数据提取脚本

脚本：

```text
scripts/extract_iedb_human_mhci_ligands.py
```

该脚本读取原始 `IEDB` zip 文件，并提取干净的人类 `HLA-I` 肽记录。

### 3.1 过滤规则

为使数据集干净且便于机器学习使用，我们采用了严格过滤。

当前过滤规则：

1. 仅保留 `MHC-I` 记录。
2. 仅保留 `Linear peptide` 记录。
3. 仅保留人类来源的肽。
4. 仅保留人类宿主的记录。
5. 仅保留阳性实验记录。
6. 仅保留明确的四位数字 `HLA-I` 等位基因。
7. 仅保留具有有效 `molecule_parent_uniprot_id` 的记录。
8. 仅保留未修饰肽。
9. 仅保留标准氨基酸序列。
10. 仅保留长度为 `9` 的肽。

### 3.2 为什么只保留 MHC-I 记录？

本项目研究 `HLA-I` 肽呈递，因此只保留：

```text
MHC Restriction / Class == I
```

这会移除 `MHC-II` 记录及其他无关记录。

### 3.3 为什么只保留人类肽和人类宿主？

长期目标与人类肿瘤疫苗有关，因此我们只希望保留来源于人类蛋白、并在人类样本中呈递的肽。

这样可以避免将病毒、细菌、小鼠或其他物种的肽混入数据集。

### 3.4 为什么只保留阳性记录？

我们希望使用已被报道为发生呈递或结合的肽，因此保留满足以下条件的实验结果：

```text
Qualitative Measurement starts with Positive
```

这提供了已报道的阳性呈递证据。

### 3.5 为什么只保留四位数字 HLA 等位基因？

部分 IEDB 记录具有不明确的 HLA 名称，例如：

```text
HLA-A2
HLA class I
```

这些名称不够精确。我们只保留如下等位基因：

```text
HLA-A*02:01
HLA-B*07:02
HLA-C*03:04
```

原因：

- 不同四位数字等位基因可能具有不同的肽 motif。
- 机器学习模型需要明确的等位基因标签。
- 不明确的 HLA 标签可能引入噪声。

### 3.6 为什么只保留具有 `molecule_parent_uniprot_id` 的记录？

后续会从与正样本相同的蛋白中构建负样本，因此需要稳定的蛋白 ID。

我们使用：

```text
molecule_parent_uniprot_id
```

因为它比 `source_molecule_uniprot_id` 更标准化，且覆盖率更好。

### 3.7 `source_molecule` 与 `molecule_parent` 的区别

`source_molecule` 是 IEDB 中的直接分子注释。

`molecule_parent` 是母体蛋白或标准化后的蛋白。

简单示例：

```text
peptide
  comes from source_molecule
    belongs to molecule_parent
```

推荐用法：

| 字段 | 含义 | 用途 |
|---|---|---|
| `source_molecule` | 直接来源分子 | 用于追溯原始注释 |
| `source_molecule_uniprot_id` | 来源分子链接中的 UniProt ID | 可能缺失 |
| `molecule_parent` | 标准化母体蛋白 | 更适合蛋白层面分析 |
| `molecule_parent_uniprot_id` | 母体蛋白的 UniProt ID | 本项目使用的主要蛋白 ID |

### 3.8 为什么只保留未修饰的 9-mer 肽？

大多数 `HLA-I` 配体是短肽，通常为 8–11 个氨基酸。

我们决定先使第一个标准数据集保持简单、干净，因此仅保留未修饰的 `9-mer` 肽。

优点如下：

1. 所有肽具有相同长度。
2. 特征编码简单。
3. 机器学习模型更容易比较。
4. 修饰肽不会增加额外复杂度。
5. 数据集更容易被其他学生使用。

被移除的示例：

```text
FTDPRTMGY + PHOS(T6)
HEIFTDPRTMGY + OX(M10)
```

---

## 4. 处理后的 IEDB 输出文件

输出目录：

```text
data/processed/
```

重要文件：

| 文件 | 说明 |
|---|---|
| `iedb_human_mhci_ligands.csv.gz` | 证据层面记录；每行对应一条 IEDB 实验记录 |
| `iedb_human_mhci_ligands_unique_peptide_mhc_tissue.csv.gz` | 唯一的 `peptide-HLA-tissue` 表 |
| `iedb_human_mhci_ligands_unique_peptide_mhc_tissue_protein.csv.gz` | 含蛋白信息的唯一表；配对的主要输入 |
| `iedb_human_mhci_ligands_summary.json` | 提取汇总 |

最终提取汇总：

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

最终处理表具有以下特征：

- 仅包含人类 `HLA-I` 记录
- 仅包含明确的四位数字 HLA 等位基因
- 仅包含未修饰的 9-mer 肽
- 仅包含具有 `molecule_parent_uniprot_id` 的记录

---

## 5. 组织和 HLA 汇总

脚本：

```text
scripts/summarize_tissue_hla_uniprot.py
```

输出文件：

```text
data/processed/iedb_tissue_summary.csv
data/processed/iedb_tissue_hla_uniprot_summary.csv
```

当前汇总：

```text
tissues: 48
tissue_hla_pairs: 1280
```

`iedb_tissue_summary.csv` 的列：

| 列 | 含义 |
|---|---|
| `source_tissue` | 组织名称 |
| `n_hla_alleles` | 该组织中的 HLA 等位基因数量 |
| `n_molecule_parent_uniprot_ids` | 来源蛋白数量 |
| `n_peptides` | 肽数量 |

示例：

```text
source_tissue,n_hla_alleles,n_molecule_parent_uniprot_ids,n_peptides
lymphoid,113,11664,81156
blood,101,13570,104559
NA,94,958,2238
lung,53,9949,33772
kidney,45,4958,8678
```

如果 IEDB 未提供明确组织信息，组织可以为 `NA`。

---

## 6. 正负样本构建

脚本：

```text
scripts/build_tissue_specificity_pairs.py
```

输入：

```text
data/processed/iedb_human_mhci_ligands_unique_peptide_mhc_tissue_protein.csv.gz
```

输出：

```text
data/processed/iedb_tissue_specificity_pairs.csv.gz
data/processed/iedb_tissue_specificity_pairs_summary_9mer.csv
```

### 6.1 正样本定义

对于一个 `tissue-HLA-UniProt` 组，正样本肽为：

```text
在目标组织中、该 HLA 等位基因条件下、来自该 UniProt 蛋白且被报道的肽
```

### 6.2 负样本定义

对于相同的 `tissue-HLA-UniProt` 组，负样本按以下逻辑选择：

1. 找到目标 `tissue-HLA-UniProt` 组中的所有正样本肽。
2. 找到来自相同 `UniProt` 蛋白、相同 `HLA`，但在其他组织中被报道的所有肽。
3. 去除已经在目标 `tissue-HLA` 组中被报道的肽序列。
4. 剩余肽即为候选负样本肽。
5. 比较正样本肽与候选负样本肽的数量。
6. 保留较小的一组。
7. 从较大的一组随机采样相同数量的肽。
8. 将正负肽一一随机配对。

### 6.3 为什么使用这种负采样策略？

这很重要。

负样本不是随机肽。它必须满足：

- 与正样本肽来自相同蛋白
- 在相同 HLA 等位基因条件下被报道
- 在其他组织中被报道
- 未在目标 tissue-HLA 组中被报道

这一设计降低了一些简单偏差。

例如，模型不应仅学习某个蛋白在某个组织中常见。由于正负样本来自相同的 `UniProt` 蛋白，模型被迫学习更具组织特异性的肽偏好。

### 6.4 配对样本量

当前配对数据：

```text
input_rows: 549,453
pairs: 125,649
paired_rows: 251,298
tissue_hla_pairs_with_pairs: 1,091
```

即：

```text
positive samples: 125,649
negative samples: 125,649
total rows: 251,298
```

### 6.5 配对验证

脚本会自动检查配对数据。

验证结果：

```text
validation_invalid_pair_ids: 0
validation_mismatched_uniprot_pair_ids: 0
validation_negative_reported_in_target_tissue_hla: 0
validation_tissue_hla_label_count_mismatches: 0
validation_tissue_hla_duplicate_label_peptides: 0
validation_tissue_hla_unique_peptide_mismatches: 0
```

含义：

- 每个 `pair_id` 恰好包含一个正样本和一个负样本
- 每个配对使用相同的 `molecule_parent_uniprot_id`
- 负肽未在目标 `tissue-HLA` 中被报道
- 对每个 `tissue-HLA`，正负行数相等
- 对每个 `tissue-HLA`，正样本肽序列唯一
- 对每个 `tissue-HLA`，负样本肽序列唯一
- 对每个 `tissue-HLA`，正负唯一肽数量相等

---

## 7. 肽长度分布

脚本：

```text
scripts/summarize_pair_peptide_lengths.py
```

输出：

```text
data/processed/iedb_tissue_specificity_pair_length_distribution.csv
```

由于此前已经只保留未修饰 9-mer 肽，最终配对数据仅有长度为 9 的肽：

```text
peptide_length,positive_count,negative_count,positive_fraction,negative_fraction
9,125649,125649,1.0,1.0
```

---

## 8. tissuePMHC 标准数据集

脚本：

```text
scripts/build_tissuepmhc_dataset.py
```

输入：

```text
data/processed/iedb_tissue_specificity_pairs.csv.gz
```

输出目录：

```text
data/tissuePMHC/
```

输出文件：

| 文件 | 说明 |
|---|---|
| `tissuePMHC_train.csv.gz` | 标准训练数据集 |
| `tissuePMHC_test.csv.gz` | 标准测试数据集 |
| `tissuePMHC_summary.csv` | 每个 tissue-HLA task 的汇总 |
| `tissuePMHC_metadata.json` | 元数据 |

### 8.1 数据集名称

数据集命名为：

```text
tissuePMHC
```

### 8.2 机器学习任务

选择任务类型 A：

```text
为每个 tissue-HLA 配对训练一个二分类器。
```

例如：

```text
blood + HLA-A*02:01 -> 一个二分类器
lung + HLA-A*02:01 -> 一个二分类器
lymph node + HLA-C*05:01 -> 一个二分类器
```

输入：

```text
peptide_sequence
```

输出：

```text
label = 1 or 0
```

其中：

```text
label = 1：在目标 tissue-HLA 中被报道
label = 0：来自相同 UniProt 和 HLA、在其他组织中被报道、但不在目标 tissue-HLA 中
```

### 8.3 为什么移除样本量较小的 tissue-HLA task？

一些 tissue-HLA 配对的肽对数量很少。若 task 样本过少：

- 模型无法学习稳定模式
- 测试性能会有较大噪声
- 结果不可靠

因此仅保留满足以下条件的 tissue-HLA task：

```text
n_pairs > 500
```

这使每个 task 都具有足够的训练样本。

### 8.4 训练集与测试集划分

对每个选定的 tissue-HLA task：

```text
随机选择 100 对正负样本作为测试数据。
其余所有配对作为训练数据。
```

随机种子：

```text
20260704
```

为什么每个 task 恰好使用 100 个测试 pair？

- 所有 task 具有相同测试集大小。
- 每个测试集包含 100 个正样本与 100 个负样本。
- 可以公平比较性能。
- 对简单的首个基准而言，测试集足够大。

### 8.5 tissuePMHC 数据规模

最终选定 task：

```text
selected_tissue_hla_groups: 44
```

训练数据：

```text
pairs: 48,486
rows: 96,972
positive rows: 48,486
negative rows: 48,486
```

测试数据：

```text
pairs: 4,400
rows: 8,800
positive rows: 4,400
negative rows: 4,400
```

### 8.6 划分验证

验证结果：

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

含义：

- 没有任何 `pair_id` 同时出现在训练集和测试集
- 训练数据平衡
- 测试数据平衡
- 每个选定 tissue-HLA task 恰好有 100 个正样本和 100 个负样本

### 8.7 tissuePMHC 数据集中的列

训练和测试文件具有相同列：

| 列 | 含义 |
|---|---|
| `dataset` | 数据集名称，恒为 `tissuePMHC` |
| `split` | `train` 或 `test` |
| `sample_id` | 样本 ID |
| `pair_id` | 正负配对 ID |
| `label` | `1` 为正样本，`0` 为负样本 |
| `target_tissue` | 目标组织 |
| `mhc_restriction` | HLA-I 等位基因 |
| `peptide_sequence` | 未修饰 9-mer 肽序列 |
| `molecule_parent_uniprot_id` | UniProt 蛋白 ID |
| `source_molecule` | IEDB 中的直接来源分子 |
| `source_molecule_uniprot_id` | 来源分子的 UniProt ID，可能为 `NA` |
| `molecule_parent` | 标准化母体蛋白 |
| `reported_tissues_same_hla_uniprot` | 该肽在相同 HLA 和 UniProt 条件下被报道的组织 |

### 8.8 选定 task 示例

来自 `data/tissuePMHC/tissuePMHC_summary.csv`：

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

## 9. 基线机器学习模型

脚本：

```text
scripts/run_tissuepmhc_baselines.py
```

输入：

```text
data/tissuePMHC/tissuePMHC_train.csv.gz
data/tissuePMHC/tissuePMHC_test.csv.gz
```

输出：

```text
results/tissuePMHC_baselines/per_task_metrics.csv
results/tissuePMHC_baselines/summary_metrics.csv
results/tissuePMHC_baselines/metadata.json
```

### 9.1 为什么运行基线模型？

基线模型是简单模型，它们可以帮助回答：

1. 数据集中是否存在可学习信号？
2. 各个 tissue-HLA task 的难度如何？
3. 未来模型应与怎样的性能进行比较？

### 9.2 测试的模型

测试了五种简单模型：

| 模型名称 | 特征 | 分类器 |
|---|---|---|
| `onehot_logistic_regression` | 9 个位置 × 20 个氨基酸的 one-hot | Logistic Regression |
| `blosum62_logistic_regression` | 9 个位置 × 20 个 BLOSUM62 分数 | Logistic Regression |
| `blosum62_random_forest` | BLOSUM62 | Random Forest |
| `blosum62_extra_trees` | BLOSUM62 | Extra Trees |
| `blosum62_hist_gradient_boosting` | BLOSUM62 | Histogram Gradient Boosting |

### 9.3 为什么使用 one-hot 编码？

One-hot 编码是最简单的序列编码。对一个 9-mer 肽：

```text
9 positions x 20 amino acids = 180 features
```

它不使用氨基酸之间的生物学相似性，是一个良好的简单基线。

### 9.4 为什么使用 BLOSUM62 编码？

`BLOSUM62` 描述氨基酸替换相似性，可为模型提供部分氨基酸生物学相似性信息。

对每个氨基酸，使用一个 20 维 BLOSUM62 分数向量。对一个 9-mer 肽：

```text
9 positions x 20 scores = 180 features
```

### 9.5 为什么测试随机森林和梯度提升？

Logistic regression 是线性模型，random forest 和 gradient boosting 能建模非线性模式。

它们仍然简单且易于运行，因此在使用深度学习之前是有价值的基线模型。

### 9.6 总体基线结果

下表结果为 44 个 tissue-HLA 二分类 task 的平均值。

| 模型 | mean AUROC | median AUROC | mean AUPRC | mean ACC | mean MCC |
|---|---:|---:|---:|---:|---:|
| `onehot_logistic_regression` | `0.7558` | `0.7442` | `0.7384` | `0.6909` | `0.3841` |
| `blosum62_logistic_regression` | `0.7554` | `0.7434` | `0.7358` | `0.6893` | `0.3807` |
| `blosum62_hist_gradient_boosting` | `0.7536` | `0.7526` | `0.7431` | `0.6818` | `0.3661` |
| `blosum62_extra_trees` | `0.7450` | `0.7396` | `0.7348` | `0.6782` | `0.3606` |
| `blosum62_random_forest` | `0.7446` | `0.7379` | `0.7357` | `0.6798` | `0.3644` |

### 9.7 结果解释

主要观察：

1. 最佳简单基线是 `onehot_logistic_regression`。
2. `BLOSUM62 + logistic regression` 与 one-hot logistic regression 几乎相同。
3. 在这次初步测试中，random forest 未提升性能。
4. Gradient boosting 具有竞争力，尤其是在 AUPRC 上。
5. 数据集存在真实信号，因为 AUROC 明显高于 0.5。
6. 不同 tissue-HLA task 的难度不同。

表现较好的示例：

| 模型 | 组织 | HLA | AUROC | AUPRC | ACC | MCC |
|---|---|---|---:|---:|---:|---:|
| `blosum62_logistic_regression` | `umbilical cord blood` | `HLA-A*02:01` | `0.8958` | `0.8939` | `0.785` | `0.571` |
| `onehot_logistic_regression` | `umbilical cord blood` | `HLA-A*02:01` | `0.8946` | `0.8994` | `0.760` | `0.520` |
| `blosum62_hist_gradient_boosting` | `lymph node` | `HLA-C*05:01` | `0.8794` | `0.8074` | `0.825` | `0.657` |

表现较差的示例：

| 模型 | 组织 | HLA | AUROC | AUPRC | ACC | MCC |
|---|---|---|---:|---:|---:|---:|
| `onehot_logistic_regression` | `blood` | `HLA-B*07:02` | `0.6121` | `0.6168` | `0.560` | `0.120` |
| `blosum62_logistic_regression` | `blood` | `HLA-B*07:02` | `0.6161` | `0.6532` | `0.550` | `0.100` |
| `blosum62_hist_gradient_boosting` | `blood` | `HLA-B*07:02` | `0.6168` | `0.6456` | `0.570` | `0.140` |

---

## 10. 当前项目文件

重要文件与目录：

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

## 11. 如何复现结果

按以下顺序运行脚本：

```bash
python scripts/extract_iedb_human_mhci_ligands.py
python scripts/summarize_tissue_hla_uniprot.py
python scripts/build_tissue_specificity_pairs.py --summary-output data/processed/iedb_tissue_specificity_pairs_summary_9mer.csv
python scripts/summarize_pair_peptide_lengths.py
python scripts/build_tissuepmhc_dataset.py
python scripts/run_tissuepmhc_baselines.py
```

如果以下文件未在其他程序中打开：

```text
data/processed/iedb_tissue_specificity_pairs_summary.csv
```

也可以使用以下命令：

```bash
python scripts/build_tissue_specificity_pairs.py
```

---

## 12. 重要说明

### 12.1 负样本不是真正的生物学阴性样本

负样本表示：

```text
未在目标 tissue-HLA 中被报道
```

它并不表示：

```text
从未在该 tissue-HLA 中呈递
```

这是因为 IEDB 基于已报道的实验。某个肽可能确实发生呈递，但尚未被测量或报道。

因此，该任务应描述为：

```text
组织特异性呈递偏好预测
```

而非绝对的呈递与非呈递预测。

### 12.2 IEDB 存在研究偏差

一些组织和 HLA 等位基因被研究得更频繁，因此数据集可能包含以下方面的偏差：

- 实验设计
- 组织可获得性
- 疾病研究重点
- HLA 等位基因的热门程度
- 质谱检测深度

在论文中使用该数据集时应说明这一点。

### 12.3 这是一个干净的初始版本

此版本仅使用：

- 人类肽
- 人类宿主
- `HLA-I`
- 四位数字 HLA 等位基因
- 阳性记录
- 可获得的 `molecule_parent_uniprot_id`
- 未修饰的标准氨基酸肽
- 9-mer 肽

这使数据集保持干净且易于使用。

未来版本可包括：

- 8-mer、10-mer、11-mer 肽
- 修饰肽
- 更宽松的 HLA 标签
- 外部验证数据

---

## 13. 后续工作

### 13.1 更好的单任务模型

当前基线模型较为简单。后续可以尝试更强模型，例如：

1. `XGBoost` 或 `LightGBM`
2. 小型 `CNN` 模型
3. 氨基酸理化特征
4. 肽嵌入模型
5. 预训练蛋白语言模型嵌入，例如 `ESM`

这些模型应与当前简单基线进行比较。

### 13.2 更严格的测试集

当前划分在每个 tissue-HLA task 内随机抽取测试 pair。后续可构建更困难的测试集：

1. 按 `UniProt ID` 划分。
2. 按肽序列划分。
3. 使用外部数据集。

这样可以检验模型能否泛化至新蛋白或新肽。

### 13.3 多任务二分类预测

这是最重要的后续方向。

当前为每个 tissue-HLA task 单独训练一个模型。这种方式简单，但有一个弱点：

```text
每个模型只能使用一个 tissue-HLA 配对的数据。
```

许多 tissue-HLA 配对的数据有限；不同 HLA 等位基因和组织之间也可能共享有用信息。

例如：

- 两个 HLA 等位基因可能具有相似的肽 motif
- 两个组织可能具有相似的呈递偏好
- 某种组织特异性模式可能跨多个 HLA 等位基因出现

因此，希望构建多任务二分类模型。

一种可能的模型输入为：

```text
peptide_sequence + tissue + HLA allele
```

输出为：

```text
组织-HLA 特异性呈递偏好的概率
```

一种可能的神经网络设计为：

```text
共享肽编码器
+ 组织嵌入
+ HLA 嵌入
+ task-specific 或共享输出层
```

核心思想：

```text
利用其他组织和其他 HLA 等位基因的信息，改善对一个目标 tissue-HLA 配对的预测。
```

重要研究问题：

1. 多任务学习是否优于单任务模型？
2. 哪些 tissue-HLA task 从共享学习中获益最多？
3. 哪些 HLA 等位基因共享相似的组织特异性信号？
4. 哪些组织共享相似的呈递偏好？
5. 模型能否帮助选择更好的肿瘤疫苗候选肽？

### 13.4 与肿瘤疫苗设计的联系

长期目标是为肿瘤疫苗研究提供支持。

未来，模型可以与以下信息结合：

- 肿瘤抗原肽
- 新抗原
- 肿瘤表达数据
- 正常组织表达数据
- HLA 结合预测
- 免疫原性预测

一种可能的最终使用场景：

```text
给定患者 HLA 分型和候选肿瘤肽，
预测哪些肽更可能在目标肿瘤组织中呈递，
且更不可能在重要正常组织中呈递。
```

这有助于优先选择更安全、更有效的肿瘤疫苗候选肽。

---

## 14. 总结

在本项目阶段，我们构建了一个用于研究组织特异性 `HLA-I` 肽呈递的干净、可复现数据集。

主要产出：

1. 干净的、处理后的人类 `HLA-I` 9-mer 肽 IEDB 数据。
2. 成对的组织特异性正负样本。
3. 标准 `tissuePMHC` 训练和测试数据集。
4. 44 个 tissue-HLA 二分类 task 的基线模型结果。
5. 一套完整的可复现脚本。

最佳简单基线：

```text
model: onehot_logistic_regression
mean AUROC: 0.7558
mean AUPRC: 0.7384
mean accuracy: 0.6909
mean MCC: 0.3841
```

该结果表明，肽序列包含对 tissue-HLA 特异性呈递偏好有用的信号。

下一项主要工作是构建能在组织和 HLA 等位基因之间共享信息的多任务二分类模型，并提升每个特定 tissue-HLA task 的预测效果。
