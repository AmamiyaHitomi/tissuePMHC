# 2026.7.5 研究日志

## 1. 今日目标

今天的目标是完成 `NEW_RESEARCH_ROADMAP.md` 中 Stage 1 的第一轮深度学习优化实验。

核心问题是：

```text
传统单任务 baseline 之后，多任务神经网络是否能进一步提升 tissuePMHC 的预测性能？
```

本次重点实现和比较以下模型：

1. `neural_single_task`
2. `shared_peptide_encoder_task_heads`
3. `conditioned_tissue_hla`

其中，`shared_peptide_encoder_task_heads` 被确定为当前 E2 主 baseline。

## 2. 数据和任务设置

本次实验沿用已有的 `tissuePMHC` 标准 train/test split。

数据规模：

```text
Number of tasks: 44
Training rows: 96,972
Test rows: 8,800
Test set per task: 100 positive + 100 negative
Peptide length: 9
```

任务定义仍然是：

```text
task = target_tissue + mhc_restriction
```

也就是说，每一个 tissue-HLA 组合是一个二分类任务。

## 3. 代码实现

今天新增并整理了两个 PyTorch 训练脚本。

第一版脚本：

```text
scripts/run_tissuepmhc_neural_baselines.py
```

用途：

```text
单次运行 neural_single_task、shared_peptide_encoder_task_heads、conditioned_tissue_hla。
```

第二版脚本：

```text
scripts/run_tissuepmhc_neural_baselines_v2.py
```

用途：

```text
保留 E2 主 baseline，重复 3 个 seed，并对 conditioned_tissue_hla 做一轮调参。
```

第一版脚本保留作为可复现实验代码，第二版脚本单独输出到：

```text
results/tissuePMHC_neural_baselines_v2/
```

这样不会覆盖第一版结果。

## 4. 模型说明

### 4.1 neural_single_task

`neural_single_task` 表示每一个 tissue-HLA 任务单独训练一个小神经网络。

输入：

```text
peptide sequence
```

输出：

```text
binary label
```

这个模型用于检查简单神经网络是否能超过传统机器学习 baseline。

### 4.2 shared_peptide_encoder_task_heads

`shared_peptide_encoder_task_heads` 是本次最重要的 E2 模型。

结构：

```text
peptide sequence
-> shared peptide encoder
-> task-specific head
-> binary prediction
```

`shared peptide encoder` 指所有任务共享同一个肽段特征提取器。

`task-specific head` 指每个 tissue-HLA 任务有自己的分类层。

这个模型的含义是：

```text
不同任务共享 peptide 表示，但最后的分类边界由每个任务自己学习。
```

### 4.3 conditioned_tissue_hla

`conditioned_tissue_hla` 是条件模型。

结构：

```text
peptide encoder
+ tissue embedding
+ HLA embedding
-> shared classifier
-> binary prediction
```

`embedding` 指可训练向量表示，用来把 tissue 或 HLA 这种类别变量变成神经网络可以使用的数值向量。

该模型的目标是学习：

```text
给定 peptide、tissue 和 HLA，预测该 peptide 是否偏向在该 tissue-HLA 条件下出现。
```

## 5. 评价指标说明

本次记录的核心指标包括：

```text
AUROC
AUPRC
accuracy
balanced accuracy
F1
MCC
worst-10-task mean AUROC
```

`AUROC` 是 Area Under the Receiver Operating Characteristic Curve，表示模型整体区分正负样本的能力。

`AUPRC` 是 Area Under the Precision-Recall Curve，更关注正样本检出能力。

`MCC` 是 Matthews Correlation Coefficient，综合考虑 true positive、true negative、false positive 和 false negative，在二分类任务中通常比 accuracy 更稳。

`worst-10-task mean AUROC` 表示 AUROC 最差的 10 个任务的平均值，用来观察模型是否只提升平均性能，还是也改善困难任务。

## 6. 第一版实验结果

第一版实验输出目录：

```text
results/tissuePMHC_neural_baselines/
```

整体结果如下：

| model | mean AUROC | mean AUPRC | mean accuracy | mean MCC |
|---|---:|---:|---:|---:|
| `shared_peptide_encoder_task_heads` | 0.7944 | 0.7766 | 0.7225 | 0.4498 |
| `conditioned_tissue_hla` | 0.7756 | 0.7620 | 0.7053 | 0.4125 |
| `neural_single_task` | 0.7326 | 0.7223 | 0.6669 | 0.3356 |

原传统 baseline 中最好的平均模型是：

```text
onehot_logistic_regression
mean AUROC = 0.7558
mean AUPRC = 0.7384
mean accuracy = 0.6909
mean MCC = 0.3841
```

因此，第一版实验说明：

```text
shared_peptide_encoder_task_heads 明显超过传统 baseline。
conditioned_tissue_hla 也超过传统 baseline，但弱于 shared_peptide_encoder_task_heads。
neural_single_task 弱于传统 baseline。
```

## 7. 第二版实验设计

第二版实验将 E2 作为主 baseline，并重复 3 个 random seed。

`random seed` 是随机数种子，用来控制模型初始化、batch shuffle 等随机过程。重复多个 seed 可以判断结果是否稳定。

使用的 seed：

```text
20260704
20260705
20260706
```

第二版实验包含 5 组实验：

```text
E2_shared_peptide_encoder_task_heads
E3_conditioned_default
E3_conditioned_wider_condition
E3_conditioned_wider_hidden
E3_conditioned_low_dropout
```

其中 conditioned model 的调参设置为：

| experiment | condition_dim | hidden_dim | dropout | learning_rate |
|---|---:|---:|---:|---:|
| `E3_conditioned_default` | 16 | 128 | 0.2 | 0.001 |
| `E3_conditioned_wider_condition` | 32 | 128 | 0.2 | 0.001 |
| `E3_conditioned_wider_hidden` | 32 | 256 | 0.2 | 0.0005 |
| `E3_conditioned_low_dropout` | 32 | 128 | 0.1 | 0.001 |

`dropout` 是一种正则化方法，训练时随机丢弃一部分神经元输出，用来降低过拟合风险。

## 8. 第二版稳定性结果

第二版实验输出目录：

```text
results/tissuePMHC_neural_baselines_v2/
```

3 个 seed 的稳定性结果如下：

| experiment | mean AUROC | AUROC std | mean AUPRC | mean accuracy | mean MCC | worst-10 mean AUROC |
|---|---:|---:|---:|---:|---:|---:|
| `E2_shared_peptide_encoder_task_heads` | 0.7927 | 0.0010 | 0.7777 | 0.7180 | 0.4404 | 0.7178 |
| `E3_conditioned_wider_condition` | 0.7833 | 0.0029 | 0.7685 | 0.7152 | 0.4350 | 0.6992 |
| `E3_conditioned_wider_hidden` | 0.7825 | 0.0025 | 0.7714 | 0.7102 | 0.4241 | 0.6882 |
| `E3_conditioned_default` | 0.7821 | 0.0020 | 0.7720 | 0.7112 | 0.4255 | 0.6912 |
| `E3_conditioned_low_dropout` | 0.7770 | 0.0039 | 0.7618 | 0.7088 | 0.4216 | 0.6860 |

`AUROC std` 是 3 个 seed 的 AUROC 标准差。标准差越小，说明不同随机种子下结果越稳定。

关键结论：

```text
E2_shared_peptide_encoder_task_heads 的 mean AUROC 最高，而且 AUROC std 只有 0.0010。
```

这说明 E2 的提升不是一次随机运气，而是比较稳定的模型收益。

## 9. E2 与传统 baseline 的比较

传统最强平均 baseline：

```text
onehot_logistic_regression mean AUROC = 0.7558
```

E2 三个 seed 平均：

```text
E2 mean AUROC = 0.7927
```

提升：

```text
0.7927 - 0.7558 = 0.0369
```

这说明共享 peptide encoder 的多任务学习明显优于当前传统单任务 baseline。

E2 相比每个任务最好的传统 baseline：

```text
30 tasks improved
14 tasks degraded
mean AUROC delta = +0.0201
mean AUPRC delta = +0.0199
```

## 10. E2 提升最大的任务

E2 相比每个任务最好的传统 baseline，提升最大的任务如下：

| target_tissue | mhc_restriction | mean AUROC | baseline AUROC | AUROC delta |
|---|---:|---:|---:|---:|
| thymus | HLA-A*02:01 | 0.9017 | 0.7974 | +0.1043 |
| blood | HLA-C*05:01 | 0.8332 | 0.7440 | +0.0892 |
| lung | HLA-A*02:01 | 0.8272 | 0.7455 | +0.0817 |
| small intestine | HLA-B*15:01 | 0.8027 | 0.7218 | +0.0809 |
| umbilical cord blood | HLA-A*02:01 | 0.9511 | 0.8958 | +0.0553 |
| ovary | HLA-A*02:01 | 0.8440 | 0.7922 | +0.0518 |

这些任务说明 shared peptide encoder 确实能从其他任务中学到可迁移信息。

## 11. E2 下降最大的任务

E2 相比每个任务最好的传统 baseline，下降最大的任务如下：

| target_tissue | mhc_restriction | mean AUROC | baseline AUROC | AUROC delta |
|---|---:|---:|---:|---:|
| lymphoid | HLA-C*12:02 | 0.7301 | 0.7798 | -0.0497 |
| brain | HLA-B*40:02 | 0.7035 | 0.7378 | -0.0343 |
| blood | HLA-B*27:05 | 0.7490 | 0.7818 | -0.0328 |
| uterine cervix | HLA-B*07:02 | 0.8003 | 0.8298 | -0.0295 |
| lymph node | HLA-A*02:01 | 0.8219 | 0.8491 | -0.0272 |
| lung | HLA-A*24:02 | 0.8297 | 0.8475 | -0.0178 |

这些下降任务提示存在 `negative transfer`。

`negative transfer` 指多任务共享学习后，某些任务受到其他任务干扰，性能反而下降。

## 12. E2 按 HLA 分组的变化

E2 在不同 HLA 上的平均 AUROC delta：

| HLA | n_tasks | mean AUROC delta |
|---|---:|---:|
| HLA-C*05:01 | 3 | +0.0477 |
| HLA-A*02:01 | 12 | +0.0411 |
| HLA-A*24:02 | 6 | +0.0298 |
| HLA-B*51:01 | 3 | +0.0280 |
| HLA-B*15:01 | 5 | +0.0251 |
| HLA-B*27:05 | 2 | -0.0192 |
| HLA-C*12:02 | 1 | -0.0497 |

初步观察：

```text
HLA-A*02:01 和 HLA-C*05:01 相关任务从共享学习中受益明显。
HLA-B*27:05 和 HLA-C*12:02 相关任务可能更容易受到 negative transfer。
```

## 13. E2 按组织分组的变化

E2 在不同 tissue 上的平均 AUROC delta：

| tissue | n_tasks | mean AUROC delta |
|---|---:|---:|
| small intestine | 1 | +0.0809 |
| thymus | 2 | +0.0763 |
| umbilical cord blood | 1 | +0.0553 |
| ovary | 1 | +0.0518 |
| breast | 1 | +0.0492 |
| lung | 3 | +0.0348 |
| blood | 9 | +0.0130 |
| lymphoid | 12 | +0.0083 |
| lymph node | 4 | -0.0022 |
| spleen | 1 | -0.0096 |

初步观察：

```text
部分小组织任务从共享学习中获益更明显。
lymphoid 和 blood 虽然任务数较多，但平均提升不算最大。
```

## 14. conditioned_tissue_hla 调参结论

本次 conditioned model 最好的配置是：

```text
E3_conditioned_wider_condition
condition_dim = 32
hidden_dim = 128
dropout = 0.2
learning_rate = 0.001
mean AUROC = 0.7833
```

相比默认 conditioned model：

```text
0.7833 - 0.7821 = +0.0012
```

提升很小。

相比 E2：

```text
0.7833 - 0.7927 = -0.0094
```

因此，本轮调参没有让 conditioned model 超过 E2。

结论：

```text
普通 tissue embedding + HLA embedding 的 conditioned model 目前不是最强模型。
```

但是 conditioned model 仍然有价值，因为它是后续 E4 HLA pseudo-sequence conditioning 的基础结构。

## 15. 今日主要结论

今天最重要的结论是：

```text
E2 shared_peptide_encoder_task_heads 是当前最强、最稳定的早期神经网络 baseline。
```

更具体地说：

1. `neural_single_task` 不如传统 baseline，说明单任务小神经网络在当前数据量下不稳定。
2. `conditioned_tissue_hla` 超过传统 baseline，但不如 E2。
3. `shared_peptide_encoder_task_heads` 明显超过传统 baseline，并且 3 个 seed 下非常稳定。
4. E2 有 30 个任务超过每任务最佳传统 baseline，但也有 14 个任务下降。
5. negative transfer 已经出现，后续需要做任务分组和更细的 transfer analysis。
6. 普通 HLA embedding 的 conditioned model 调参收益有限，下一步更应该加入 HLA pseudo-sequence。

## 16. 当前模型排序

根据今天结果，当前模型排序为：

```text
E2 shared peptide encoder + task-specific heads
>
E3 conditioned tissue-HLA model
>
traditional single-task baseline
>
E1 neural single-task baseline
```

其中 E2 是当前主 baseline。

## 17. 下一步计划

下一步建议进入 E4：

```text
conditioned model with tissue embedding + HLA pseudo-sequence
```

具体任务：

1. 获取或构建 HLA pseudo-sequence 表。
2. 实现 HLA pseudo-sequence encoder。
3. 比较：

```text
conditioned model + HLA embedding
vs
conditioned model + HLA pseudo-sequence
```

如果 HLA pseudo-sequence 有提升，说明模型确实受益于更有生物意义的 HLA 表示。

之后再考虑：

```text
E4 + FAMO
task grouping and negative transfer analysis
CAGrad
```

`FAMO` 是 Fast Adaptive Multitask Optimization，用于自动调节不同任务的 loss 权重。

`CAGrad` 是 Conflict-Averse Gradient Descent，用于缓解不同任务梯度方向冲突的问题。

## 18. 重要文件

第一版代码：

```text
scripts/run_tissuepmhc_neural_baselines.py
```

第二版代码：

```text
scripts/run_tissuepmhc_neural_baselines_v2.py
```

第一版结果：

```text
results/tissuePMHC_neural_baselines/per_task_metrics.csv
results/tissuePMHC_neural_baselines/summary_metrics.csv
results/tissuePMHC_neural_baselines/metadata.json
```

第二版结果：

```text
results/tissuePMHC_neural_baselines_v2/per_task_metrics.csv
results/tissuePMHC_neural_baselines_v2/summary_metrics.csv
results/tissuePMHC_neural_baselines_v2/stability_metrics.csv
results/tissuePMHC_neural_baselines_v2/metadata.json
```

## 19. 后续编号校正说明

后续复盘时确认：本日志第 17 节中的 `E4 + FAMO / task grouping / CAGrad` 是 2026-07-05 当天基于当时信息做出的临时计划，不再作为当前正式编号使用。

原因是后续实验显示：

```text
E4 HLA pseudo-sequence 线没有超过 E2。
因此，后续性能主线应继续沿 E2 shared peptide encoder + task-specific heads 展开。
```

当前正式编号应理解为：

```text
E2: shared peptide encoder + task-specific heads
E3: conditioned tissue + HLA ID
E4: conditioned tissue + HLA pseudo-sequence
E4b: HLA ID + HLA pseudo-sequence hybrid
E5: FAMO on E2
E6: HLA/tissue task grouping on E2
E7: selective HLA/global hard selection
E8: global/HLA soft ensemble
E9: E2 + CAGrad
```

也就是说，E4 线现在保留为 HLA biological representation analysis，也就是 HLA 生物表示分析线；E2/E8 线才是当前性能主线。
