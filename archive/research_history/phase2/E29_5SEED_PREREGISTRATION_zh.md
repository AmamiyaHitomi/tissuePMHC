# E29 5-seed 增量扩展预注册

预注册日期：2026-07-11  
状态：已按预注册完成；OOF 门槛通过后仅执行一次固定 5-seed test 评估

## 目标

检验 E29 Multi-kernel CNN E14a 从 3 seeds 扩展到 5 seeds 后，OOF 性能是否继续提高。该实验是 standard split 上最后一次性能扩展；无论结果如何，完成后停止基于该 split 的模型或融合调参。

## 固定成员

已有成员：20260704、20260705、20260706。  
新增成员：20260707、20260708。  
禁止在结果生成后删除单个 seed、改成 4-seed 子集或根据 test 选择成员。

## 固定模型与训练配置

完全复用 E29 3-seed 配置：position-preserving multi-kernel Conv1d peptide encoder；kernel size 2、3、5；每个 kernel 32 channels；embedding dimension 16；hidden dimension 128；dropout 0.2；25 epochs；AdamW；learning rate 0.001；weight decay 0.0001；tissue/HLA auxiliary loss weight 均为 0.1；batch size 512。

只训练新增 seeds 20260707 与 20260708。已有三个 seed 的 OOF 和 test 预测直接复用，不重复训练。

## OOF 决策规则

先对两个新增 seed 运行与 E29 3-seed 完全相同的 3-fold pair-grouped OOF。将五个 seed 的 OOF 预测平均后，与已经固定的 E29 3-seed OOF 比较。

只有以下三项同时满足，才允许读取已有 test 预测并训练两个新增 seed 的 full-train test 模型：

```text
5-seed OOF mean AUROC - 3-seed OOF mean AUROC >= 0.0010
5-seed OOF worst-10 mean AUROC - 3-seed OOF worst-10 mean AUROC >= -0.0010
5-seed OOF mean AUPRC - 3-seed OOF mean AUPRC >= -0.0005
```

任一条件失败时，实验停止于 OOF，E29 3-seed 继续作为正式主结果，不读取或生成新增 seed 的 test 预测。

## Test 政策

OOF 通过后，才训练新增两个 seed 的完整训练集模型。随后将其与已有三个 seed 的固定 test 预测合并，计算一次 E29 5-seed mean。不得根据 test 修改模型、权重、成员、阈值或融合规则。

正式主比较为 E29 5-seed mean 相对 E29 3-seed mean；同时保留相对 E17 5-seed 的固定比较。此次评估完成后，不再继续增加 seed 或在 standard split 上选择模型。

## 执行结果（预注册后记录）

两个新增 seed 已按预注册配置完成训练。5-seed OOF 相对 3-seed 的 mean AUROC 增益为 0.00191，worst-10 mean AUROC 增益为 0.00298，mean AUPRC 增益为 0.00242；三项门槛均通过，因此允许并完成了一次固定的 full-train 5-seed test 评估。

正式 test 中，E29 5-seed mean 达到 mean AUROC 0.8373、mean AUPRC 0.8259、mean accuracy 0.7588、mean MCC 0.5175、worst-10 mean AUROC 0.7670。相对 E29 3-seed 的 AUROC 增益为 0.00316。该结果不触发成员筛选、融合权重调整或进一步 standard split 调参。
