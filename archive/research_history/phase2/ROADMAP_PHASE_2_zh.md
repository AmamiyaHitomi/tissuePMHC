# tissuePMHC Phase 2 实验路线图

基准模型：E14a（auxiliary global branch + plain HLA-specific branch + 固定概率平均）。当前最佳已更新为 E29 Multi-kernel CNN E14a 5-seed ensemble：mean AUROC 0.8373、mean AUPRC 0.8259、worst-10 mean AUROC 0.7670。E15–E27 与 E29 已完成；E28 Negative Correlation Learning 不再默认执行，standard split 性能探索按预注册承诺结束。

## 准备步骤（不计实验编号）

保存每个分支的逐样本预测，并以 `sample_id`、`target_tissue`、`mhc_restriction` 对齐。融合只能在完全对齐的预测上进行。使用 validation/OOF 的实验必须保证 test 只用于最终评估。

## 实验顺序

| 顺序 | 实验编号 | 实验 | 融合/训练算法 | 算法出处 |
|---:|---|---|---|---|
| 1 | E15 | 固定融合规则消融：概率平均、logit 平均、task 内 rank 平均 | score averaging / logit averaging / rank aggregation | [Dietterich, 2000](https://link.springer.com/chapter/10.1007/3-540-45014-9_1)；[Bajor et al., 2019](https://jmlr.csail.mit.edu/papers/v20/18-094.html) |
| 2 | E16 | global 与 HLA 分支分别进行 5、10、20 次 MC Dropout 预测后平均，再采用 E15 最优规则融合 | Monte Carlo Dropout | [Gal & Ghahramani, 2016](https://proceedings.mlr.press/v48/gal16.html) |
| 3 | E17 | 在 global 分支内、HLA 分支内分别平均 3 个独立 seed，随后扩展到 5 个 seed | Deep Ensemble / prediction averaging | [Lakshminarayanan et al., 2017](https://arxiv.org/abs/1612.01474) |
| 4 | E18 | 以 validation/OOF 预测选择一个全局 global 权重；比较固定 0.5 与验证集选择的全局权重 | cross-validated convex weighting | [van der Laan et al., 2007](https://pubmed.ncbi.nlm.nih.gov/17910531/) |
| 5 | E19 | 在每个分支内比较 checkpoint ensemble、snapshot ensemble 与 E14 原训练流程 | Checkpoint Ensemble / Snapshot Ensemble | [Chen et al., 2017](https://arxiv.org/abs/1710.03282)；[Huang et al., 2017](https://arxiv.org/abs/1704.00109) |
| 6 | E20 | 从完全相同的训练前缀和随机数状态出发，比较 SWA 与原 final checkpoint | Stochastic Weight Averaging | [Izmailov et al., 2018](https://arxiv.org/abs/1803.05407) |
| 7 | E21 | 在 E14a global auxiliary branch 中，以主分类 BCE 为目标任务，根据 shared encoder 上的梯度余弦动态门控 tissue/HLA auxiliary loss；只重训 global branch，并复用已保存的 HLA plain 分支预测 | Gradient-Similarity Auxiliary Gating | [Du et al., 2018](https://arxiv.org/abs/1812.02224) |
| 8 | E22 | 只在主分类、tissue auxiliary、HLA auxiliary 三个宏目标间使用 Nash bargaining 权重；每 10–20 个 batch 更新一次权重，其余 batch 复用最近解 | Periodic Nash-MTL | [Navon et al., ICML 2022](https://proceedings.mlr.press/v162/navon22a.html) |
| 9 | E23 | 对 E14a global auxiliary branch 周期性 fork 不同 auxiliary-weight 训练路径，以 pair-grouped validation 主任务误差选择并合并有效更新，再与固定 auxiliary 权重比较 | ForkMerge | [Jiang et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/60f9118a849e8e9a0c67e2a36ad80ebf-Abstract-Conference.html) |
| 10 | E24 | 使用 train 内 pair-grouped validation 主任务作为 meta-loss，动态学习 tissue/HLA auxiliary 权重；主分类权重固定为 1，避免牺牲目标任务换取 auxiliary accuracy | Auto-Lambda | [Liu et al., TMLR 2022](https://openreview.net/forum?id=KKeCMim5VN) |
| 11 | E25 | 构建 HLA-Structured PLE：设置少量 global shared experts 和按 HLA 路由的 lightweight private expert；每个 tissue-HLA task 的 gate 融合共享与对应 HLA 表示；禁止为 44 个 task 各复制完整 expert | Progressive Layered Extraction | [Tang et al., RecSys 2020](https://doi.org/10.1145/3383313.3412236) |
| 12 | E26 | 构建候选模型库并按 validation/OOF mean task AUROC 贪心选择成员；根据 E19/E20 结果默认排除 checkpoint ensemble、snapshot ensemble 与 SWA | Greedy Ensemble Selection | [Caruana et al., 2004](https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf) |
| 13 | E27 | 在 E26 候选库的 task-rank 特征上训练固定强 L2 正则的 Logistic Regression 二层模型；使用 OOF 分数训练，独立 test 仅评估一次 | Super Learner / stacked generalization | [Wolpert, 1992](https://doi.org/10.1016/S0893-6080(05)80023-1)；[van der Laan et al., 2007](https://pubmed.ncbi.nlm.nih.gov/17910531/) |
| 14 | E28 | 联合训练 global 与 HLA 分支，并在主任务损失外加入成员间负相关误差项；与独立训练的 E14a 比较 | Negative Correlation Learning | [Liu & Yao, 1999](https://www.sciencedirect.com/science/article/pii/S0893608099000738) |
| 15 | E29 | 保留 E14a 的 global auxiliary + HLA plain 双分支和 task-rank fusion，仅将 Flatten-MLP peptide encoder 替换为保留位置信息的 multi-kernel CNN；1-/3-seed OOF 与预注册 5-seed OOF 均通过，5-seed test 成为最终 standard-split 最佳 | Multi-kernel CNN representation diversity | [Kim, 2014](https://aclanthology.org/D14-1181/) |

## 执行顺序

```text
E15 → E16 → E17 → E18 → E19 → E20 → E21 → E22 → E23 → E24 → E25 → E26 → E27
                                                                            ↓
                                                     E29 1-seed OOF（通过）→ E29 3-seed OOF/test（通过）→ 预注册 E29 5-seed OOF/test（通过）
                                                                                                                ↓
                                                                                         standard split 性能探索结束；转向泛化验证
```

E15–E27 与 E29 已完成，编号、代码和结果目录保持不变。E28 尚未运行，因 E29 已成功提供表示多样性而不再作为默认后续实验。E29 5-seed 是预注册的最后一次 standard split 性能扩展，现已完成。

## E21–E25：已完成的辅助权重与结构路线

E21–E24 只修改并重训 E14a global auxiliary branch，固定复用相同 seed 的既有 HLA plain 分支预测。这样既能隔离 global 分支算法改动，也能避免重复训练 HLA 分支。它们使用的三个宏目标定义为：

```text
主任务：44 个 tissue-HLA heads 的整体 BCE
辅助任务 1：tissue classification
辅助任务 2：HLA classification
```

E22 Nash-MTL 只处理三个宏目标，禁止扩展为 44-task 逐任务 Nash bargaining。权重每 10–20 个 batch 更新一次，其余 batch 复用最近解；若 1-seed 估时表明正式实验超过预算，则增加更新间隔。

E23 ForkMerge 和 E24 Auto-Lambda 均必须使用 train 内 pair-grouped validation，test 不参与 fork 合并、meta-weight 更新或超参数选择。Auto-Lambda 的 meta-objective 只使用主分类 loss，主任务权重固定为 1。

E25 是结构型实验，训练一个 HLA-Structured PLE 模型，不复用 E14a global branch。为控制成本，默认采用 2 个窄 global experts 和每个样本只激活的 1 个 HLA-private expert；不得实现为 44 套完整 private experts。E25 先作为单模型与 E14a 分支级基线比较，只有单模型表现有竞争力时才考虑与现有 HLA/plain 或 E17 成员融合。

| 优先级 | 实验 | 预计正式 3-seed 用时 | 进入下一阶段的最低条件 |
|---:|---|---:|---|
| 1 | E21 gradient-similarity auxiliary gating | 约 1.3–1.8 小时 | 相对同 seed E14a/E15 mean AUROC 不下降；gating 不能长期全开或全关 |
| 2 | E22 periodic Nash-MTL | 约 1.7–2.3 小时 | bargaining 权重有限且稳定；三个宏目标均未被长期压制；相对固定权重基线无明显退化 |
| 3 | E23 ForkMerge | 约 2.5–4 小时 | validation 合并稳定过滤部分 auxiliary 更新；相对固定权重 E14a 有正向配对增益 |
| 4 | E24 Auto-Lambda | 约 2.5–4.5 小时 | meta-weight 有限且随训练产生可解释变化；主任务 validation/test 不低于固定权重基线 |
| 5 | E25 HLA-Structured PLE | 约 3–5 小时 | 单模型 mean AUROC 超过 E10 MMoE，并接近或超过对应 3-seed E14a anchor；gate 未塌缩到单一 expert |

时间为根据本机 E14–E20 运行记录做出的工程估计，不是论文给出的保证。每个实验采用两阶段停止规则：

```text
1. 先运行 1 seed smoke/screen，并记录算法实际额外耗时和诊断量。
2. 只有相对同 seed 基线无明显退化，才运行完整 3 seeds。
3. 先与 E14a/E15 的配对 3-seed 结果比较。
4. 只有获得稳定增益后，才扩展至 5 seeds，并挑战 E17 的 0.8263 AUROC。
```

## E26–E27：已完成的严格 OOF 集成

E26 已生成 3-fold pair-grouped OOF 预测和独立 full-train test 预测，候选池包含 E14 final 的三个单 seed、E16 MC-20 的三个单 seed，以及两类 3-seed mean。OOF greedy selection 只选择 `e14_final_3seed_mean`。其 test mean AUROC 为 0.8246，略高于 E17 3-seed 的 0.8243，但低于 E17 5-seed 的 0.8263。

E27 复用同一 OOF/test 候选库，以候选 task-rank 为输入训练固定 `C=0.1` 的 L2 Logistic Regression。其 test mean AUROC 为 0.8243，未超过 E26 或 E17。E14 final 3-seed mean 与 MC-20 3-seed mean 的 OOF task-rank correlation 为 0.9987，说明失败原因主要是候选同质性，而不是缺少更复杂的融合器。

## E29：下一优先路线——Multi-kernel CNN E14a

E14a 当前 peptide encoder 为 amino-acid embedding、Flatten 和两层 MLP。E29 保留已验证有效的 global auxiliary branch、plain HLA-specific branch 与 task-rank fusion，只替换 peptide encoder，以降低对 E14 成员的表示相关性，同时控制结构变化范围。

固定首轮配置：embedding dimension 16；kernel size 为 2、3、5；每个卷积分支 32 channels；卷积输出保留位置后拼接并投影至 hidden dimension 128；dropout 0.2；25 epochs；AdamW；learning rate 0.001；weight decay 0.0001。禁止只用 global max pooling 丢弃 9-mer 的锚定位置信息。

E29 的 seed 20260704、3-fold pair-grouped OOF screen 已完成且未读取 test。CNN 单模型 OOF mean AUROC 为 0.8007，高于匹配 E14 seed 的 0.7915；与匹配 E14 的 task-rank correlation 为 0.8727。与 E14 3-seed mean 等权 rank 融合后，OOF mean AUROC 从 0.8042 提高到 0.8097，worst-10 mean AUROC 从 0.7257 提高到 0.7314，四项条件全部通过。

预先固定的通过条件为：

```text
1. 相对匹配 seed 的 E14 OOF mean AUROC 下降不超过 0.005；
2. 与 E14 的 OOF task-rank correlation 低于 0.97；
3. E14/E29 等权 task-rank 融合相对 E14 OOF mean AUROC 至少提升 0.001；
4. worst-10 mean OOF AUROC 下降不超过 0.001。
```

E29 3-seed 阶段已完成。3-seed CNN OOF mean AUROC 为 0.8138，高于 E14 3-seed 的 0.8042；相关性为 0.9428；等权 rank 融合相对 E14 OOF 提升 0.0090 AUROC，四项条件再次全部通过。随后固定模型进行一次 test 评估，E29 3-seed mean 达到 mean AUROC 0.8341、mean AUPRC 0.8228、worst-10 mean AUROC 0.7634，超过 E17 5-seed 的 0.8263/0.8139/0.7573。

在新增 seed 训练前，项目预注册了最后一次 5-seed 增量扩展：只训练 20260707、20260708，复用前三个 seed；5-seed OOF 必须相对 3-seed 满足 AUROC 增益至少 0.0010、worst-10 AUROC 增益不低于 −0.0010、AUPRC 增益不低于 −0.0005。结果为 OOF AUROC +0.00191、worst-10 +0.00298、AUPRC +0.00242，三项均通过。一次固定的 5-seed test 评估达到 mean AUROC 0.8373、mean AUPRC 0.8259、worst-10 mean AUROC 0.7670，相对 E29 3-seed 分别增加 0.00316、0.00315、0.00359。E29 输出沿用 E26 长表格式：`split,candidate,seed,sample_id,target_tissue,mhc_restriction,label,score`。

**决策：** E29 5-seed mean 成为最终 standard split 主结果。E14/E29 等权融合在 3-seed test 上 AUROC 为 0.8340，略低于 E29 单独结果，因此不取代 E29；不得根据已观察 test 调融合权重或选择成员。

作为 standard split 的最后一次确认性扩展，E29 5-seed 增量实验已按预注册完成。只训练新增 seeds 20260707/20260708，并复用现有 20260704–20260706 的 OOF/test 预测；三项 OOF gate 均通过后才生成新增 test 预测。完整规则与执行结果见 `E29_5SEED_PREREGISTRATION_zh.md`，一键脚本为 `scripts/run_tissuepmhc_e29_incremental_5seed.py`。至此停止 standard split 性能调参。

## E28：后备路线

E28 不再作为 E27 后的默认下一步。Negative Correlation Learning 不引入新的输入信息，可能以牺牲成员自身排序为代价制造分歧，而且相关系数权重需要额外选择。原先仅在 E29 失败时考虑 E28；现在 E29 已在 OOF 和 test 上成功，因此 E28 停止默认执行，仅在论文需要完整方法覆盖时保留为固定单个相关性权重的 1-seed OOF 补充实验。

## 参考方法核对

以下条目给出路线图中 E21–E29 的方法来源。Uncertainty Weighting 与 Aligned-MTL 已从当前正式路线移除。

1. **E21 — Gradient-Similarity Auxiliary Gating**  
   Du, Y., Czarnecki, W. M., Jayakumar, S. M., Farajtabar, M., Pascanu, R., & Lakshminarayanan, B. *Adapting Auxiliary Losses Using Gradient Similarity*. arXiv:1812.02224, 2018.  
   https://arxiv.org/abs/1812.02224

2. **E22 — Nash-MTL**  
   Navon, A., Shamsian, A., Achituve, I., Maron, H., Kawaguchi, K., Chechik, G., & Fetaya, E. *Multi-Task Learning as a Bargaining Game*. Proceedings of the 39th International Conference on Machine Learning, PMLR 162, pp. 16428–16446, 2022.  
   https://proceedings.mlr.press/v162/navon22a.html

3. **E23 — ForkMerge**  
   Jiang, J., Chen, B., Pan, J., Wang, X., Liu, D., Jiang, J., & Long, M. *ForkMerge: Mitigating Negative Transfer in Auxiliary-Task Learning*. Advances in Neural Information Processing Systems 36, 2023.  
   https://proceedings.neurips.cc/paper_files/paper/2023/hash/60f9118a849e8e9a0c67e2a36ad80ebf-Abstract-Conference.html

4. **E24 — Auto-Lambda**  
   Liu, S., James, S., Davison, A. J., & Johns, E. *Auto-Lambda: Disentangling Dynamic Task Relationships*. Transactions on Machine Learning Research, 2022.  
   https://openreview.net/forum?id=KKeCMim5VN

5. **E25 — Progressive Layered Extraction (PLE)**  
   Tang, H., Liu, J., Zhao, M., & Gong, X. *Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations*. Proceedings of the 14th ACM Conference on Recommender Systems, pp. 269–278, 2020. DOI: 10.1145/3383313.3412236.  
   https://doi.org/10.1145/3383313.3412236

6. **E26 — Greedy Ensemble Selection**  
   Caruana, R., Niculescu-Mizil, A., Crew, G., & Ksikes, A. *Ensemble Selection from Libraries of Models*. Proceedings of ICML, 2004.  
   https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf

7. **E27 — Stacked Generalization / Super Learner**  
   Wolpert, D. H. *Stacked Generalization*. Neural Networks, 5(2), 241–259, 1992. DOI: 10.1016/S0893-6080(05)80023-1.  
   van der Laan, M. J., Polley, E. C., & Hubbard, A. E. *Super Learner*. Statistical Applications in Genetics and Molecular Biology, 6(1), 2007.  
   https://doi.org/10.1016/S0893-6080(05)80023-1  
   https://pubmed.ncbi.nlm.nih.gov/17910531/

8. **E28 — Negative Correlation Learning**  
   Liu, Y., & Yao, X. *Ensemble Learning via Negative Correlation*. Neural Networks, 12(10), 1399–1404, 1999.  
   https://www.sciencedirect.com/science/article/pii/S0893608099000738

9. **E29 — Multi-kernel CNN Peptide Encoder**  
   Kim, Y. *Convolutional Neural Networks for Sentence Classification*. Proceedings of EMNLP, pp. 1746–1751, 2014. E29 借用多尺度一维卷积提取局部序列模式的基本思想，但针对固定 9-mer 保留卷积后位置信息。  
   https://aclanthology.org/D14-1181/
