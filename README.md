# tissuePMHC

tissuePMHC 是一个研究组织条件如何影响 MHC-I 肽呈递的科研代码库。项目将肽序列、组织环境和 MHC 条件联合建模，覆盖人类 tissue–HLA 与小鼠 tissue–H2 基准，包括数据构建、传统基线、多任务神经网络、严格泛化评估、外部预测器比较、模型解释和论文结果整理。

当前正式主线是：

- 人类 `tissuePMHC_phase7_min200`：157 个 tissue–HLA 任务，147,596 条训练样本和 31,400 条测试样本；
- 小鼠 `mousePMHC`：24 个 tissue–H2 任务，13,532 条训练样本和 4,800 条测试样本；
- 当前论文源文件：`paper/tissuePMHC_latex_v9/`；
- 历史阶段、修改记录和论文 v3–v8：`archive/`。

数据集和完整实验结果保留在本地，但默认不纳入 Git。公开发布时应通过具备版本号、许可证说明和校验值的独立数据存储提供。

## 研究任务

每条正负样本共享肽来源蛋白和 MHC 条件：正样本表示该肽在目标组织–MHC 条件下被报告，配对负样本表示同一来源和 MHC 条件下的肽在其他组织出现、但未在目标组织报告。主要评估包括 AUROC、AUPRC、MCC、分类准确率和配对准确率（PairAcc）。

仓库同时保留两类泛化协议：

1. 标准分层/交叉验证实验，用于比较不同模型结构；
2. peptide-disjoint 等严格划分，用于评估未见肽上的泛化能力并审计数据泄漏。

## 目录结构

```text
tissuePMHC/
├── scripts/                              # 数据构建、训练、评估、解释与作图入口
│   └── scripts/_runner.py                # 人类 min200 主线的统一路径配置
├── final_phase/                          # 基于已有预测的最终统计与审计（不训练）
├── protocols/                            # 冻结的实验协议
├── extra_premium/                        # 人类 premium 数据实验
├── extra_occurrence_equal_dataset/       # 人类 occurrence-equal 实验代码
├── extra_mouse_occurrence_equal_dataset/ # 小鼠 occurrence-equal 实验代码
├── extra2/
│   ├── human_157tasks_experiments/       # 人类 157-task 扩展实验
│   └── mouse_2human_experiments/         # 小鼠到人类的迁移/对照实验
├── extra1/
│   ├── issue5/                           # NetMHCpan/MHCflurry 外部预测器公共实现
│   └── issue9/                           # 严格泛化与跨物种实验公共实现及测试
├── paper/tissuePMHC_latex_v9/            # 当前论文、补充材料、图和出图脚本
├── archive/                              # 历史阶段、评审记录和论文 v3–v8
├── tools/                                # 文档和历史论文辅助工具
├── data/                                 # 本地数据（默认不提交）
└── results/                              # 主实验输出（默认不提交）
```

扩展实验可能将结果写入各自目录下的 `results/`。所有 `data/`、`results/`、`external/`、模型检查点、缓存和本地环境均由 [.gitignore](.gitignore) 默认排除。

## 环境安装

建议使用 Python 3.10 或更高版本，并按照本机 CUDA/CPU 条件先安装匹配的 PyTorch。基础环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

需要运行测试、MHCflurry 或翻译工具时安装可选依赖：

```powershell
python -m pip install -r requirements-optional.txt
```

本项目已使用以下环境完成整理后的语法、入口和轻量模型验证：Python 3.13.11、PyTorch 2.10.0、NumPy 2.4.1、pandas 3.0.0、scikit-learn 1.8.0、SciPy 1.17.0 和 Matplotlib 3.10.8。

## 数据约定

代码使用 `Path(__file__).resolve()` 从脚本位置定位项目根目录。所有命令建议从项目根目录执行，不要依赖用户目录或手工修改 `PYTHONPATH`。

| 数据目录 | 用途 | 当前元数据中的任务数 |
|---|---|---:|
| `data/tissuePMHC_phase7_min200/` | 当前人类 min200 主基准 | 157 |
| `data/mousePMHC/` | 当前小鼠主基准 | 24 |
| `data/tissuePMHC/` | 早期人类基准 | 依元数据为准 |
| `data/humanPMHC_premium/` | 人类 premium 扩展 | 依数据版本为准 |
| `data/humanPMHC_occurence_equal_dataset/` | 人类 occurrence-equal 数据 | 依数据版本为准 |
| `data/mousePMHC_occurence_equal_dataset/` | 小鼠 occurrence-equal 数据 | 依数据版本为准 |
| `data/processed/`、`data/raw/` | 中间数据和原始来源 | 不作为模型入口 |

注意：两个本地数据目录沿用历史拼写 `occurence`；对应代码目录使用正确拼写 `occurrence`。请勿擅自重命名数据目录，否则需要同步调整所有读取路径。

仓库不会自动下载受许可证限制的数据或第三方模型。公开复现时应为每个数据版本提供来源、许可证、生成命令、随机种子和 SHA-256 校验值。

## 人类主实验流程

### 1. 构建 Phase 7 min200 数据

```powershell
python scripts/build_human_dataset_min200.py --help
python scripts/build_human_dataset_min200.py
```

该入口固定使用 `total_pairs > 200` 的 tissue–HLA 纳入规则，并将文件写入 `data/tissuePMHC_phase7_min200/`。如果本地已经保存正式数据，运行前应先备份并核对元数据；数据构建会写入 `data/`。

### 2. 运行主要模型链

```powershell
python scripts/run_e2_shared_heads.py --help
python scripts/run_e14_auxiliary_soft_ensemble.py --help
python scripts/run_e17_seed_ensemble.py --help
python scripts/run_e26_all_in_one.py --help
python scripts/run_e29_multikernel_cnn_oof.py --help
```

这些包装入口将数据和输出路径固定到 Phase 7 min200 命名空间，避免与早期实验结果混用。E29 默认只执行 OOF 流程；只有完成匹配基线并确认评估协议后，才应显式启用 test 预测。

严格未见肽评估入口：

```powershell
python scripts/run_e31_peptide_disjoint_oof.py --help
```

所有训练入口都应记录随机种子、每个 epoch/seed 和总耗时；正式运行时还应保存 Python、PyTorch、CUDA 与 GPU 信息。

## 小鼠实验

小鼠实验入口位于 `scripts/run_mousepmhc_*.py`，覆盖传统 OOF、多任务模型、集成、梯度审计、H2 结构建模和 peptide-disjoint 评估。建议从帮助信息和阶段报告确认依赖顺序：

```powershell
python scripts/run_mousepmhc_phase3_e0_oof.py --help
python scripts/run_mousepmhc_phase4_e15_five_seed_confirmation.py --help
python scripts/run_mousepmhc_phase6_e33_peptide_disjoint_oof.py --help
```

不要将人类和小鼠的预测文件、fold assignment 或 candidate 名称交叉复用。

## 扩展实验

各扩展实验的具体数据契约和运行顺序见：

- [人类 premium 实验](extra_premium/README_zh.md)
- [premium 机制实验](extra_premium/experiments/README_zh.md)
- [人类 occurrence-equal 实验](extra_occurrence_equal_dataset/README_zh.md)
- [小鼠 occurrence-equal 实验](extra_mouse_occurrence_equal_dataset/README_zh.md)
- [小鼠 occurrence-equal 调参](extra_mouse_occurrence_equal_dataset/adjusting/README_zh.md)
- [人类 157-task 实验](extra2/human_157tasks_experiments/README_zh.md)
- [mouse-to-human 实验](extra2/mouse_2human_experiments/README_zh.md)

## 外部预测器

NetMHCpan 与 MHCflurry 只用于相应对照实验，不是核心训练流程的必需依赖。路径可通过环境变量设置：

```powershell
$env:MHCFLURRY_EXECUTABLE = "C:\path\to\mhcflurry-predict.exe"
$env:MHCFLURRY_MODELS_DIR = "C:\path\to\models_class1_presentation\models"
$env:NETMHCPAN_EXECUTABLE = "/path/inside/wsl/netMHCpan"
```

对应脚本也支持 `--executable`、`--model-dir` 或 `--wsl-distro`。NetMHCpan 及其模型文件受官方许可证约束，不应随本仓库重新分发。

## 最终统计与审计

[final_phase/](final_phase/README_zh.md) 包含 9 个按依赖顺序运行的只读分析入口，覆盖 PairAcc、matched-fold 审计、parent-protein overlap、来源可行性、统计检验、补充表、可视化、可复现性材料和数据卡：

```powershell
python final_phase/run_all.py
```

该流程不会训练模型，但会读取既有 human/mouse 标准与 peptide-disjoint 预测，并将新分析写入 `results/final_phase/`。缺少任一输入时会报告具体路径。

## 论文与图表

当前论文位于 `paper/tissuePMHC_latex_v9/`：

- `main.tex`：主文；
- `supplementary_main.tex`：补充材料；
- `sections/`：正文和补充章节；
- `references/`：参考文献；
- `figures/`：当前图文件；
- `scripts/`：论文出图脚本。

Figure 5 的复现入口为：

```powershell
python scripts/make_figure5_human_benchmark.py
```

它读取已有 occurrence-equal 指标并将 PDF/PNG 写入 v9 的 `figures/`。论文历史版本不参与当前构建，统一保存在 [archive/paper_versions/](archive/README.md)。

在完整 TeX Live/MiKTeX 环境中，可在论文目录执行：

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary_main.tex
```

## 测试与代码检查

运行现有自动化测试：

```powershell
python -m pytest
```

执行不训练模型的语法编译检查：

```powershell
python -m compileall -q scripts final_phase extra1 extra2 extra_premium `
  extra_occurrence_equal_dataset extra_mouse_occurrence_equal_dataset
```

整理后的验证状态：270 个 Python 文件通过 AST 语法解析，现有 pytest 测试为 4 项全部通过；代表性传统模型拟合、PyTorch 前向/反向传播、真实数据表头和主要 CLI 入口均已完成抽查。该状态不等同于重新执行所有耗时训练实验。

## 结果与可复现性要求

- 训练、验证和测试必须使用相同的数据集标识与冻结 split；
- 不得混用 standard、min200、premium 和 occurrence-equal 的结果；
- 每次正式训练应保存参数、seed、fold、epoch/seed/总耗时、软件环境和输入文件校验值；
- GPU 训练即使固定随机种子也未必逐位确定，应报告硬件和 CUDA/cuDNN 配置；
- 完整结果和模型权重应发布到独立制品仓库，Git 中只保留复现必需的小型汇总和 provenance；
- `final_phase/` 依赖冻结预测文件，不应以不同训练版本的同名文件覆盖。

## 上传前检查

```powershell
python -m pytest
git status --short
git status --ignored --short
```

确认暂存内容不包含 `data/`、任何 `results/`、`external/`、模型检查点、虚拟环境、缓存或第三方运行时。当前 `archive/` 是研究历史的一部分；是否随公开仓库发布，可根据期刊补充材料和审计要求决定。

## 归档

[archive/](archive/README.md) 仅用于追溯：

- `research_history/`：Phase 1–7 报告、路线图和 previous 总结；
- `review_notes/`：历史问题清单与修改记录；
- `paper_versions/`：论文 v3–v8。

归档内容可能包含旧命令、旧环境路径和旧输出约定，不保证能作为当前实验入口直接运行。

## 发布前仍需补充

正式公开前建议补充 `LICENSE`、`CITATION.cff`、作者与联系方式、论文 DOI/预印本链接，以及数据/模型制品的永久下载地址和许可证说明。
