# Mouse ← Human 方法迁移报告

更新时间：2026-07-28  
状态：迁移代码已完成；正式训练未执行。

补充说明：Human/Mouse strict identical-fold 表另缺少 three matched controls。核查确认
不是已有结果漏写，而是 one-hot logistic regression、无辅助 dual branch、带辅助 dual
branch 尚未在冻结 mouse E33 peptide-disjoint folds 上运行。现已新增
`run_mouse_strict_missing_methods.py` 和三方法总入口
`run_strict_three_all_in_one.py`；该补充实验不读取 fixed test。

## 1. 方法数量

人类 157-task Table 5 的方法池包含 28 种去重方法。论文另外记录了 capacity-matched
MHC-only、MHCflurry presentation 和 NetMHCpan EL 三种对照，因此整篇人类论文目前
共有 **31 种独立方法**。

参照 mouse Phase 3–6、mouse peptide-disjoint architecture experiments 和 issue5
general-pMHC controls，31 种方法中：

- 15 种已在小鼠项目测试；
- 16 种尚未测试；
- 16 种缺口由本目录的 12 个独立入口覆盖。

完整逐方法审计见 `METHOD_INVENTORY_zh.md`。

## 2. 不重复迁移的原则

已有小鼠实验按方法机制映射，而不是按人类实验编号映射。例如：

- human MHC-grouped 对应 mouse H2-grouped E2；
- human MMoE 对应 mouse E3/E3b；
- human FAMO 对应 mouse E5；
- human fixed global/MHC fusion 对应 mouse E11；
- human pair ranking 对应 mouse E17；
- human auxiliary supervision 对应 mouse E12；
- traditional baselines 对应 mouse E0；
- MHC-only/MHCflurry/NetMHCpan 已由 issue5 完成。

这些方法不在新目录重复训练。映射只说明方法机制已测试，不把不同 backbone 或协议
解释为完全 matched comparison。

## 3. 新迁移内容

| 入口 | 新增人类方法数 | 说明 |
|---|---:|---|
| `run_neural_single_conditioned.py` | 2 | neural single-task、conditioned tissue/H2 |
| `run_h2_pseudoseq.py` | 1 | H2 pseudo-sequence；等待真实序列文件 |
| `run_h2_hybrid.py` | 1 | H2 ID + pseudo-sequence；等待真实序列文件 |
| `run_tissue_grouping.py` | 1 | tissue-grouped hard sharing |
| `run_selective_grouping.py` | 1 | validation-selected global/H2 |
| `run_adaptive_soft_ensemble.py` | 2 | validation-delta、validation-softmax |
| `run_cagrad.py` | 1 | CAGrad |
| `run_mmoe_tuning.py` | 2 | 4×256 与 6×128 MMoE |
| `run_dbmtl.py` | 1 | DB-MTL |
| `run_auxiliary_soft.py` | 2 | 两种 auxiliary dual-branch 配置 |
| `run_mlp_dual_seed_ensemble.py` | 1 | row-level seed ensemble |
| `run_tissuepmhc_full.py` | 1 | full multi-kernel dual-branch TissuePMHC |
| **合计** | **16** | 12 个入口 |

## 4. Full TissuePMHC 的判定

mouse Phase 4 E9 已测试 multi-kernel CNN shared heads，但没有测试人类 TissuePMHC 的
完整组合：multi-kernel global auxiliary branch、MHC-specific plain branch、task-wise
rank fusion 和 multi-seed aggregation。因此完整 TissuePMHC 仍计为未完成方法。

`run_tissuepmhc_full.py` 复用冻结的 mouse E15 OOF predictions 作为已有 mouse baseline，
并将对应 candidate 名写入 transfer contract。它仍执行 OOF screen；若 gate 不通过，
原 runner 会拒绝进入 fixed test，避免绕过既有防泄漏规则。

## 5. H2 pseudo-sequence 边界

仓库当前没有经过来源核验的 H2-Db/H2-Kb/H2-Kd/H2-Kk pseudo-sequence 文件。因此
两个相关入口已迁移代码和参数，但正式运行前必须提供
`data/processed/h2_pseudo_sequences.csv`。适配器会在文件缺失时停止，不能用 HLA
序列、空序列或人工虚构序列替代。

## 6. 路径与可复现性

所有新实验读取：

```text
data/mousePMHC/mousePMHC_train.csv.gz
data/mousePMHC/mousePMHC_test.csv.gz
```

所有新结果写入：

```text
results/mousePMHC_human_method_transfer/<experiment>/
```

每个正式运行写入 `transfer_contract.json`，记录人类方法名、source runner、mouse
train/test、seed、超参数和输出位置。路径审计拒绝把输出写入历史 mouse 或 human
结果目录。

## 7. All-in-one 入口

新增 `run_all_in_one.py`，按依赖顺序串行调用全部 12 个独立入口；各实验仍写入原有
独立子目录，不合并结果。入口默认支持断点续跑，并把逐项开始时间、结束时间、返回码
和状态写入 `all_in_one_status.json`。

H2 pseudo-sequence 文件缺失时，默认仅将两个相关实验记为
`skipped_missing_h2_pseudo`，不阻塞其余十个入口。`--require-h2-pseudo` 可改为启动前
强制检查；`--continue-on-error` 可在某个实验失败后继续后续独立实验。正式实验仍未
由本次代码迁移执行。
