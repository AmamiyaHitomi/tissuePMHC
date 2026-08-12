# Overleaf 第二版修改与复查记录

项目：`tissuePMHC_2`  
日期：2026-07-23

## 已完成

1. 将全文模型名称 `TissuePMHC-Net` 统一为 `TissuePMHC`。
2. 将 RQ1 标题改为问题式标题：`Is Tissue-specific Preference Learnable?`。
3. 将 RQ1 的三行裸表格扩展为正式编号表：
   - 11 个已完成历史 baseline；
   - AUROC、AUPRC、Worst-10、随机种子数；
   - 包含 one-hot、HLA pseudo-sequence、FAMO、pair ranking、CAGrad、DB-MTL、shared heads、MMoE、auxiliary、soft ensemble 和 auxiliary ensemble。
4. RQ2 人类主结果表增加：
   - 正式编号、caption 和 label；
   - Median AUROC；
   - 与早期 44-task baseline 表的任务范围区别。
5. 将原 Figure 1 的简单柱状图改为同时展示 Mean、Median 和 Worst-10 AUROC 的分布摘要图。
6. 解释 auxiliary dual branch 被用作 matched baseline 的原因：它与最终系统共享双分支监督和融合设定，主要差别是编码器。
7. 将 matched OOF 比较改为正式编号表，并增加正文交叉引用。
8. 删除 RQ1/RQ2 对同一 strict 结果的重复数值，将 globally unseen-peptide 结果集中到 RQ4。
9. RQ3 新增正式的 component evidence 表，集中呈现：
   - multi-kernel encoder；
   - branch complementarity；
   - fusion rule；
   - seed averaging。
10. 在 Methods 和 RQ4 中明确解释：
    - OOF = out-of-fold；
    - pair-disjoint 只隔离 pair ID；
    - connected-component peptide-disjoint 隔离全局 peptide identity；
    - strict 评估是 globally unseen peptide / seen task；
    - 它不是仅在 specific task 中 unseen，也不是 unseen-task 或 protein-disjoint。
11. RQ4 标准/严格比较改为正式编号表，并说明差值为 strict minus matched standard。
12. Figure 2 的 `y=x` 参考线改为清晰的灰色粗虚线；图注明确它不是实验数据点。
13. RQ5 小鼠结果重组为：
    - 完整 baseline 表；
    - standard OOF、fixed test、peptide-disjoint OOF 完整协议汇总表；
    - H2 restriction 分层表；
    - tissue 分层的高低端汇总表。
14. 删除原 Figure 3：它只重复相邻表格的三个汇总值，没有提供分布信息。
15. 人类 Task Heterogeneity 新增：
    - tissue-level 高低端汇总表；
    - 正式编号的 HLA locus 表。
16. 附录 paired-statistics 表取消 `resizebox`，按 Human/Mouse 拆成两个面板，消除强制缩小。
17. 重写附录 Figure 4 前导解释、标题、坐标轴和图注：
    - 上图是 strict minus standard AUROC；
    - 下图是 held-out parent protein 与 fitting partition 的重叠比例。
18. 所有新增表格均采用正式 `table`、caption、label 和交叉引用。

## 无法可靠补充

1. **Sensitivity 和 Specificity**
   - 早期 44-task baseline 归档只有排名指标和部分汇总指标；
   - 没有保存所有模型在统一阈值下的混淆矩阵；
   - 不能从 AUROC/AUPRC 反推出 Sensitivity/Specificity；
   - 因此未编造这两个数值，正文中已说明缺失原因。

2. **将 NetMHCpan、MHCflurry、BigMHC 直接加入数值比较表**
   - 这些方法预测一般 binding/presentation；
   - 当前论文预测 matched tissue--MHC preference；
   - 项目中没有它们在同一 matched benchmark、相同 split 和相同 task inventory 上的 held-out predictions；
   - 直接并列表格会构成不公平比较，因此仍在 Related Work 中作任务层面的定性比较。

3. **完整 boxplot/violin plot**
   - 第二版中各方法并非都保存了完全一致的逐任务、逐种子预测；
   - 强行合并会混合 44-task、157-task、fixed-test 和 OOF 样本池；
   - Figure 1 改用 Mean、Median、Worst-tail 三种可验证的任务分布摘要，避免制造不可比的“分布”。

4. **strict protocol 下的完整 component ranking**
   - 第二版明确没有完成所有组件在同一 strict folds 上的隔离消融；
   - 因此 RQ3 只对 standard benchmark 作系统级贡献陈述，没有把 standard ablation 外推为 strict component superiority。

## 最终检查

- Overleaf 编译成功：33 页。
- LaTeX Errors：0。
- LaTeX Warnings：0。
- Overfull boxes：0。
- 剩余 Underfull boxes：12，均为自动换行松散提示，不造成越界或内容丢失。
- `TissuePMHC-Net` 残留：0。
- 重复 label：0。
- 缺失交叉引用：0。
- `table`、`figure`、`tabular`、`tikzpicture`、`axis` 环境均闭合。
