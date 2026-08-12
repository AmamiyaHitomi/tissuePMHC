# TissuePMHC v8 构建说明

## 本次更新

- 用 Human/Mouse、3 seeds 的正式同条件组件消融替换旧的描述性组件比较。
- 加入可训练 MHC-only CNN、逐任务配对统计和输入不变性审计。
- 加入正式三 seed 单残基 ISM 扰动验证、SHAP--ISM 一致性分析、锚定位点检验和 Figure 10。
- 同步更新实验设置、摘要、讨论、局限、结论及中文译读稿。
- 保留 v7 原稿；v8 中已删除从 v7 复制而来的旧 PDF、辅助文件和旧渲染图，避免误认为是 v8 编译结果。

## 编译

在 Overleaf 中将 `main.tex` 设为主文件，或在安装了完整 TeX Live/MiKTeX 的环境中运行：

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

若不使用 `latexmk`，至少连续运行两次 `pdflatex main.tex`，以解析表格和页码交叉引用。

## 已完成的静态审计

- 17 个 TeX 文件的 `begin/end` 环境均闭合。
- 无重复 `label`。
- 除 `lastpage` 宏包在编译时生成的 `LastPage` 外，无缺失交叉引用目标。
- 25 个参考文献条目覆盖全部 23 个正文引用键；8 个主文图文件全部存在。
- 新增 ISM 图已独立渲染并目视检查；TeX 源文件无 Unicode 替换字符或异常连续问号。
- 已清除“正式消融尚未完成”和“trainable tissue-blind control 未重跑”等过时表述。
- 新增结果与 `results/occurrence_equal_ablation_mhc_only` 中的正式汇总及配对统计逐项核对。
- ISM 正文数值与 `extra_occurrence_equal_dataset/results/e29_tuned_ism` 的正式结果逐项核对。

本机 Windows 和 WSL 环境均没有可用的 LaTeX 编译器，因此未在本机生成或目视核验 v8 PDF。
