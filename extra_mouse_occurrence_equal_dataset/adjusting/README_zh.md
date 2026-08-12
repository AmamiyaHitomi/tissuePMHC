# 小鼠 occurrence-equal tissuePMHC 调参

> **版本标注：** 本目录产生的结果统一称为
> **tissuePMHC（调参后版本；小鼠训练集 CV 选择）**，英文标签为
> `TissuePMHC (tuned; mouse-training-CV-selected)`。不得与原始 tissuePMHC
> 默认配置结果混写为同一个版本。

本目录只使用 `data/mousePMHC_occurence_equal_dataset`，不会覆盖已有实验结果。

`run_tuning.py` 执行以下流程：

1. 在训练集内按 task 和 pair 分组三折，测试采样策略与模型容量；
2. 围绕第一阶段最佳配置搜索辅助损失、融合权重、学习率和正则化；
3. 将所有训练集内验证结果排序并锁定一个配置；
4. 仅对锁定配置使用 seeds 20260704、20260705、20260706 在完整训练集重训；
5. 在固定测试集上报告独立 seed 均值和逐行预测 ensemble，并与已有 DB-MTL 结果比较。

正式运行：

```powershell
E:\ancd\envs\my_pytorch\python.exe extra_mouse_occurence_equal_dataset\adjusting\run_tuning.py --device cuda
```

冒烟测试：

```powershell
E:\ancd\envs\my_pytorch\python.exe extra_mouse_occurence_equal_dataset\adjusting\run_tuning.py --device cuda --smoke
```

每个 epoch、fold、seed 和总运行时间都会打印到终端，同时写入结果目录的
`timing_results.csv`；完整进度另写入 `run.log`。
