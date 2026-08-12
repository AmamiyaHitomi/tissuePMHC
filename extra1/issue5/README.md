# Issue 5：通用 pMHC predictor 与 MHC-only 对照

本目录实现 `extra/remaining_issues_5_9.md` 的第 5 项。代码不会使用本文标签微调
NetMHCpan 或 MHCflurry，并把所有外部分数统一为“越高越强”。

## 实验组成

外部冻结基线：

1. NetMHCpan-4.1 BA：主分数 `-%Rank_BA`；
2. NetMHCpan-4.1 EL：主分数 `-%Rank_EL`；
3. MHCflurry presentation：主分数 `presentation_score`，正式运行禁用 flank；
4. MHCflurry affinity percentile 可作为额外 binding 敏感性结果。

内部对照：

- human：E29-compatible position-preserving multi-kernel CNN encoder + HLA heads；
- mouse：E15-compatible peptide MLP/expert structure + H2-only gate/heads；
- 两者都不输入 tissue，不使用 tissue embedding、tissue auxiliary loss、tissue–MHC
  task head或 task-balanced loss。

## 环境

项目 Python：

```powershell
E:\ancd\envs\my_pytorch\python.exe
```

NetMHCpan-4.1 需要用户按照 DTU 的许可取得 standalone UNIX 包。建议在 Linux/WSL
中运行。MHCflurry 应固定正式稳定版和模型 bundle；不要在同一次正式实验中自动升级
软件或模型文件。

## 1. 校验并建立唯一查询

```powershell
E:\ancd\envs\my_pytorch\python.exe extra\issue5\run_tests.py
E:\ancd\envs\my_pytorch\python.exe extra\issue5\build_queries.py
```

预期生成：

```text
results/issue5_general_pmhc/queries/
├── human_unique_peptide_mhc.csv.gz
├── mouse_unique_peptide_mhc.csv.gz
├── human_mhcflurry_input.csv
├── mouse_mhcflurry_input.csv
├── human_netmhcpan_manifest.csv
├── mouse_netmhcpan_manifest.csv
└── netmhcpan/{human,mouse}/*.pep
```

正式数据预期为 human `79,759`、mouse `6,663` 个唯一 peptide–MHC 查询。

## 2. 运行 MHCflurry

先在独立环境安装并固定版本、下载 presentation bundle，例如：

```powershell
python -m pip install mhcflurry==2.2.1
mhcflurry-downloads fetch models_class1_presentation
mhcflurry-downloads path models_class1_presentation
```

若使用 Python 3.13 且 `mhcflurry-downloads` 报
`ModuleNotFoundError: No module named 'pipes'`，使用项目内兼容启动器：

```powershell
$env:PYTHONUTF8 = "1"
E:\ancd\envs\my_pytorch\python.exe extra\issue5\mhcflurry_downloads_py313.py fetch models_class1_presentation
E:\ancd\envs\my_pytorch\python.exe extra\issue5\mhcflurry_downloads_py313.py path models_class1_presentation
```

该启动器只把已从 Python 3.13 删除的 `pipes.quote` 映射到等价的
`shlex.quote`，不修改 MHCflurry 模型或预测代码。

把最后一条命令得到的 bundle 下的 `models` 目录显式传给 runner。稳定版兼容命令示例：

```powershell
python extra\issue5\run_predictors.py mhcflurry --legacy-cli `
  --executable mhcflurry-predict `
  --input results\issue5_general_pmhc\queries\human_mhcflurry_input.csv `
  --output results\issue5_general_pmhc\raw_outputs\human_mhcflurry.csv `
  --metadata results\issue5_general_pmhc\raw_outputs\human_mhcflurry.metadata.json `
  --model-dir <models_class1_presentation模型目录>\models
```

mouse 使用对应 input/output 重复一次。若所冻结版本明确提供新版统一 CLI，可省略
`--legacy-cli` 并把 executable 改为 `mhcflurry`。runner 固定使用：

```text
--no-throw --no-flanking
```

MHCflurry 2.2.1 的 legacy predict CLI 不接受 `--num-jobs`，因此 runner 默认不传该
参数。只有确认所用版本支持时才显式设置 `--num-jobs`。

正式运行前还应保存：

```powershell
mhcflurry-predict --version
mhcflurry-predict --list-supported-alleles
mhcflurry-downloads info
```

## 3. 运行 NetMHCpan-4.1

在能够执行 standalone NetMHCpan 的 Linux/WSL 环境中运行：

```bash
python extra/issue5/run_predictors.py netmhcpan \
  --executable /absolute/path/to/netMHCpan \
  --manifest results/issue5_general_pmhc/queries/human_netmhcpan_manifest.csv \
  --metadata results/issue5_general_pmhc/raw_outputs/human_netmhcpan.metadata.json
```

mouse manifest 重复一次。每个 allele 独立运行，已存在的完整输出默认复用；只有明确
需要覆盖时才传 `--force`。正式运行前必须用工具的 allele listing 核对全部
35 个 HLA 与 4 个 H2 映射，特别是：

```text
H2-Db -> H-2-Db
H2-Kb -> H-2-Kb
H2-Kd -> H-2-Kd
H2-Kk -> H-2-Kk
```

如果 WSL 无法读取 manifest 中的 Windows 路径，可在 WSL 内重新运行
`build_queries.py --output-dir`，让 manifest 保存 WSL 可见的绝对路径。

## 4. 导入统一 score cache

MHCflurry：

```powershell
E:\ancd\envs\my_pytorch\python.exe extra\issue5\import_scores.py mhcflurry `
  --queries results\issue5_general_pmhc\queries\human_unique_peptide_mhc.csv.gz `
  --predictions results\issue5_general_pmhc\raw_outputs\human_mhcflurry.csv `
  --version 2.2.1 `
  --output results\issue5_general_pmhc\score_cache\human_mhcflurry.csv.gz
```

NetMHCpan：

```powershell
E:\ancd\envs\my_pytorch\python.exe extra\issue5\import_scores.py netmhcpan `
  --queries results\issue5_general_pmhc\queries\human_unique_peptide_mhc.csv.gz `
  --manifest results\issue5_general_pmhc\queries\human_netmhcpan_manifest.csv `
  --version 4.1 `
  --output results\issue5_general_pmhc\score_cache\human_netmhcpan.csv.gz
```

mouse 重复相同操作。导入器不会用极端值填补缺失预测；unsupported/missing 组合保留
为 NaN 并进入覆盖率审计。

## 5. 外部 predictor 评估

```powershell
E:\ancd\envs\my_pytorch\python.exe extra\issue5\evaluate_external.py `
  --score-cache results\issue5_general_pmhc\score_cache\human_netmhcpan.csv.gz `
  --score-cache results\issue5_general_pmhc\score_cache\human_mhcflurry.csv.gz `
  --score-cache results\issue5_general_pmhc\score_cache\mouse_netmhcpan.csv.gz `
  --score-cache results\issue5_general_pmhc\score_cache\mouse_mhcflurry.csv.gz
```

输出：

```text
results/issue5_general_pmhc/external_evaluation/
├── row_predictions.csv.gz
├── per_task_metrics.csv
├── summary_metrics.csv
├── coverage_audit.csv
├── paired_statistics.csv
├── per_task_differences.csv
└── metadata.json
```

AUROC、AUPRC 和 PairAcc 只使用正负两行都成功评分的完整 pair。主模型指标会在相同
完整 pair 集合上重新计算。覆盖率同时报告 row、complete-pair、task 和 full-task。

matched-standard OOF 与 peptide-disjoint OOF 使用同一个 train row pool，因此冻结外部
predictor 的独立指标应相同；两栏的区别是配对比较所用的完整模型预测不同。

## 6. 控制通用分数后的增量

```powershell
E:\ancd\envs\my_pytorch\python.exe extra\issue5\stack_increment.py `
  --row-predictions results\issue5_general_pmhc\external_evaluation\row_predictions.csv.gz
```

该分析分别拟合：

```text
external score
external score + TissuePMHC score
```

standard/strict OOF 始终用另外两个 outer folds 拟合 logistic stacker，再预测 held fold。
fixed test 的 stacker 只在 matched-standard OOF 行上拟合。stacker 不输入 tissue、task
或 task intercept。这是控制通用分数后的次要增量分析，不替代原始外部分数基线。

## 7. MHC-only 内部对照

正式运行：

```powershell
E:\ancd\envs\my_pytorch\python.exe extra\issue5\run_mhc_only.py --species human --device cuda
E:\ancd\envs\my_pytorch\python.exe extra\issue5\run_mhc_only.py --species mouse --device cuda
```

runner 覆盖：

- deterministic standard pair-grouped OOF，split seed `20260711`；
- 冻结 connected-component peptide-disjoint folds；
- full-train → fixed test；
- human 三个预声明 seed；
- mouse 五个预声明 seed；
- 每个 protocol × seed × fold checkpoint，可安全续跑。

分析：

```powershell
E:\ancd\envs\my_pytorch\python.exe extra\issue5\analyze_mhc_only.py `
  --predictions results\issue5_general_pmhc\human_mhc_only\ensemble_predictions.csv.gz `
  --predictions results\issue5_general_pmhc\mouse_mhc_only\ensemble_predictions.csv.gz
```

## 8. 统计口径

每个 species × protocol × metric 报告：

- task-macro mean/median；
- main minus baseline 的 mean/median difference；
- Hodges–Lehmann difference；
- win/tie/loss；
- 10,000 次 task bootstrap mean-difference interval；
- Wilcoxon signed-rank；
- BH-FDR。

这些检验是 nominal task-level inference。任务共享 tissue、MHC、peptide 和 parent
protein，不能解释为独立外部队列证据。

## 9. 正式完成前检查

- 记录工具精确版本、运行日期、完整命令；
- 记录 MHCflurry 模型 bundle 路径及文件 SHA256；
- 记录 NetMHCpan data package 版本，但不要违反许可重新分发；
- 保存 supported-allele 原始输出；
- 检查 `coverage_audit.csv` 中全部缺失 allele；
- 不根据 fixed test 或 pooled strict OOF 结果挑选 score transform；
- 在论文中把外部模型称为 frozen general-signal controls；
- 明确外部模型可能与 IEDB/免疫肽组训练数据存在预训练重叠。
