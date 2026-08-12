# HumanPMHC Premium B–E 三 seed 结果分析

分析日期：2026-07-31

统一协议：premium train-only、固定 3-fold pair-grouped OOF、25 epochs、
seeds `20260704/20260705/20260706`。所有模型分数均有限且位于 `[0,1]`。
运行耗时只输出到终端日志，不保存在实验结果文件中。

## B：辅助任务诊断

### B1 正负例差异

三 seed、三 fold 汇总后：

| label | tissue accuracy | query tissue probability | tissue NLL | HLA accuracy |
|---:|---:|---:|---:|---:|
| 0 | 0.370313 | 0.270777 | 1.649659 | 0.600910 |
| 1 | 0.414692 | 0.292351 | 1.496266 | 0.774884 |

负例的 tissue/HLA auxiliary 明显更难，支持“query tissue 对负例缺少 peptide
支持”的诊断。随着 `other_tissue_count` 增加，tissue accuracy 进一步下降。

### B2 梯度冲突

| label | main–tissue weighted cosine | tissue conflict fraction | main–HLA weighted cosine | HLA conflict fraction |
|---:|---:|---:|---:|---:|
| 0 | -0.048064 | 0.675214 | -0.147277 | 0.927350 |
| 1 | +0.047443 | 0.329060 | +0.148079 | 0.072650 |

负例上 main 与 auxiliary 的梯度方向系统性更冲突，且 HLA 冲突比 tissue
冲突更强。这是机制诊断，不等同于删除 auxiliary 会改善最终性能。

### B3 tissue-label shuffle

相对正确 tissue auxiliary，fold 内打乱 tissue 标签后的平均变化：

| 指标 | shuffled − current |
|---|---:|
| mean task AUROC | -0.000491 |
| mean task AUPRC | -0.000105 |
| mean task PairAcc | +0.001032 |
| mean task MCC | -0.000764 |

变化接近零，任务胜负也接近 50/50。因此 tissue auxiliary 的精确标签语义
不是当前性能的主要来源；它更像一般正则化信号。B2 的梯度冲突真实存在，
但没有转化成“打乱 tissue 标签后性能明显改变”。

## C：条件结构

### 三 seed 均值

| 候选 | AUROC | AUPRC | PairAcc | MCC | worst-10 AUROC |
|---|---:|---:|---:|---:|---:|
| C0 | 0.676489 | 0.658684 | 0.678689 | 0.261562 | 0.520727 |
| C1 | 0.676574 | 0.659261 | 0.681228 | 0.261408 | 0.534830 |
| C2 | 0.681516 | 0.661815 | 0.686922 | 0.270728 | **0.538574** |
| C3 | 0.681446 | 0.662249 | 0.685361 | 0.270936 | 0.529718 |
| C4 | **0.685377** | **0.666096** | **0.690374** | **0.275577** | 0.536561 |

C2、C3 均满足相对 C0 的晋级条件；C4 是总体最优结构。C4 相对 C0：

- AUROC `+0.008888`，225 个 seed-task 比较中 66.7% 为正；
- AUPRC `+0.007413`，60.0% 为正；
- PairAcc `+0.011686`，60.4% 为正；
- MCC `+0.014015`，61.8% 为正；
- 上述 AUROC、AUPRC、PairAcc 在三个 seed 中均为正。

C4 在完全 unseen peptide 层仍有 AUROC `+0.001360`、AUPRC `+0.002513`、
PairAcc `+0.004005`，没有系统性退化；主要收益仍集中在 fitting 中出现两次
的 peptide 子集。按 tissue 宏平均，C4 的 AUROC 和 PairAcc 都在 14 个组织中
有 10 个提升；kidney 是主要退化组织，AUROC 下降 `0.038347`。

C3 的 task residual 参数为 9,675 个；移除后成为 C4，性能反而更好。因此
task residual 在当前数据规模下更像过拟合来源，推荐 C4 而不是 C3。

## D：模型是否依赖条件输入

### D2 validation tissue shuffle

三 seed 平均，相对各自 baseline：

| 候选 | ΔAUROC | ΔAUPRC | ΔPairAcc | ΔMCC |
|---|---:|---:|---:|---:|
| C0 | 0 | 0 | 0 | 0 |
| C2 | -0.014002 | -0.013572 | -0.012401 | -0.021920 |
| C3 | -0.004841 | -0.005319 | -0.004200 | -0.004472 |

C0 不读取 tissue ID，结果严格不变。C2/C3 在三个 seed 中均下降，说明二者
确实依赖 tissue 输入；C2 的依赖明显强于 C3。

### D3 组件关闭

三 seed 平均 AUROC 变化：

| 候选 | tissue off | HLA off | task residual off | auxiliary off/retrain |
|---|---:|---:|---:|---:|
| C0 | 不适用 | 不适用 | 不适用 | -0.010679 |
| C2 | -0.012316 | -0.025296 | 不适用 | -0.012360 |
| C3 | -0.003393 | -0.012624 | -0.003775 | -0.009673 |

HLA 条件贡献大于 tissue 条件。`auxiliary off` 同时关闭 tissue 和 HLA
auxiliary，因此不能把下降单独归因于 tissue auxiliary；结合 B3，更可能是
HLA auxiliary 或总体正则化贡献。C3 训练后直接关闭 task residual 会下降，
但重新训练且从结构上删除 residual 的 C4 更好，说明这是共同适应效应，
不能据此否定 C4。

### D1 tissue swap

与真实 observed-tissue 方向一致率：

| 候选 | 三 seed 平均方向一致率 |
|---|---:|
| C0 | 0.734264 |
| C2 | **0.763018** |
| C3 | 0.729060 |

C0 因切换 task head 本身就会改变分数，所以其高一致率不是组织机制证据。
C2 比 C0 高约 2.88 个百分点，并同时通过 D2 shuffle 和 D3 tissue-off，
可解释为真正使用了 tissue 条件。C3 的 swap 一致率未超过 C0，但 D2/D3
仍证明它存在较弱的 tissue 条件依赖。

D 只能证明条件依赖，不能单独证明组织特异性生物学机制。

## E：processing-first新版代码完成，未强行正式运行

检查发现premium每个正负pair的target tissue、HLA和parent UniProt完全相同，
所以parent expression单独无法直接区分pair内正负。旧expression-first代码
已删除并替换为E0–E4 processing-first设计：先用parent sequence提取真实N/C
flanks和processing score，再测试tissue processing machinery与flank表示的
交互；expression-only保留为预期较弱的负对照。

新版代码已完成全流程合成输入smoke test，合成结果随后删除。正式运行仍需
真实parent FASTA、UniProt-to-Ensembl映射、HPA表达矩阵及人工审核的组织映射。
未经审核，不把bone、brain、lymphoid或umbilical cord blood静默映射为近似组织。

## 最终判断

1. 负例 auxiliary 梯度冲突明确存在，但 tissue 标签精确语义对最终性能影响很小。
2. 让 tissue/HLA 进入主表示确实有效，最佳结构是无 task residual 的 C4。
3. C2 对 tissue 条件的真实依赖证据最强；C3 依赖较弱。
4. 当前证据支持“统计条件化有效”，尚不支持“组织加工/表达机制有效”。
5. 下一步应先完成parent peptide定位和人工组织映射，再单seed运行新版E；
   不使用未经审核的近似表达特征。
