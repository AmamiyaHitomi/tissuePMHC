# humanPMHC premium 单 seed 简单测试

本目录只测试原人类项目的四个阶段性入口：

1. `E0`：五种按 tissue-HLA 独立训练的传统基线；
2. `E2`：共享 peptide MLP 编码器和 tissue-HLA 任务头；
3. `E14a`：global auxiliary MLP 与 HLA plain MLP 的双分支融合；
4. `E29`：最终的 multi-kernel CNN 双分支模型。

所有模型都固定只使用一个 seed：

```text
20260704
```

不运行多 seed 集成、OOF、screen、调参、消融、bootstrap 或复杂的 test 比较。每个脚本只用
`humanPMHC_train.csv.gz` 训练一次，然后在 `humanPMHC_test.csv.gz` 上评估一次。

## 数据与输出隔离

输入数据固定为：

```text
data/humanPMHC_premium/humanPMHC_train.csv.gz
data/humanPMHC_premium/humanPMHC_test.csv.gz
```

所有新输出固定写入：

```text
extra_premium/results/<model_name>/
```

代码不会写入原来的 `data/tissuePMHC*`、`phase*` 或 `results/tissuePMHC*`。

每个入口只产生四个基础文件：

- `test_predictions.csv`：test 每行的预测分数；
- `per_task_metrics.csv`：每个 tissue-HLA task 的基础指标；
- `summary_metrics.csv`：单行宏平均汇总；
- `run_settings.json`：seed、数据路径和实际训练参数。

基础指标包括 AUROC、AUPRC、accuracy、MCC 和成对准确率 PairAcc。

## 运行方式

在项目根目录分别运行，不提供 all-in-one 入口：

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\run_e0_traditional.py
```

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\run_e2_shared_heads.py --device cuda
```

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\run_e14a_auxiliary_dual_branch.py --device cuda
```

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\run_e29_multikernel_cnn.py --device cuda
```

默认使用原项目的 25 epochs。若只想确认代码能够快速跑通，可以显式减少 epoch，例如：

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\run_e2_shared_heads.py --device cuda --epochs 3
```

减少 epoch 的结果只能作为 smoke test，不能直接与原项目的 25-epoch 结果比较。

## 说明

- E0 保留原项目全部五个组成：BLOSUM62 logistic regression、one-hot logistic
  regression、BLOSUM62 random forest、BLOSUM62 Extra Trees 和 BLOSUM62 histogram
  gradient boosting。五个模型共用一个 E0 输出目录，但在 CSV 中按 `model` 列区分。
- E17 和 E26 没有加入：E17 的核心是多 seed 集成，E26 的核心是 OOF 候选选择，均不符合本次单
  seed、短时间、简单 fixed-test 的要求。
- E29 保留原模型结构和 task-rank 双分支融合，但不运行原来的 OOF screen，也不做三 seed
  ensemble。
- test 数据只用于最终预测和指标计算，不参与超参数选择或模型筛选。

## NetMHCpan 与 MHCflurry

这两个模型是冻结的通用 pMHC 预测器，不读取 premium train，也不使用本数据的标签训练。
只对 premium test 去重后的 peptide–HLA 组合预测。

所有查询、原始输出和 score cache 写入：

```text
extra_premium/external/
```

最终基础指标写入：

```text
extra_premium/results/external_predictors/
```

分别运行以下入口，不提供 all-in-one：

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\build_external_queries.py
```

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\run_mhcflurry.py
```

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\run_netmhcpan.py
```

```powershell
& E:\ancd\envs\my_pytorch\python.exe extra_premium\evaluate_external_predictors.py
```

默认沿用本机已经完成原项目 Issue 5 时使用的安装：

- MHCflurry 2.2.1 legacy CLI；
- `models_class1_presentation` bundle，禁用 flanking；
- Ubuntu WSL 中的 `/home/hitomi/netMHCpan-4.1/netMHCpan`；
- NetMHCpan-4.1b，同时输出 EL 和 BA 分数。

若安装位置发生改变，两个 runner 都提供相应的命令行参数。NetMHCpan 已存在的逐 allele
输出默认复用；只有明确需要覆盖时才给 `run_netmhcpan.py` 传 `--force`。

外部分数没有统一概率阈值，因此只报告完整 pair 上的 AUROC、AUPRC 和 PairAcc，不报告
accuracy 或 MCC。主结果为 MHCflurry presentation score、NetMHCpan EL rank 和 BA
rank；其余 affinity/EL score 作为附加敏感性结果。
