# 主文与补充材料的交叉引用编译

`main.tex` 与 `supplementary_main.tex` 使用 `xr-hyper` 相互引用。不能把两份文档彼此独立地各编译一次；否则先编译的文档读不到另一份文档的 `.aux` 文件，会把有效引用排成 `??`。

在安装了完整 MiKTeX 或 TeX Live 的 Windows 环境中运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\build_papers.ps1
```

脚本按“补充材料 -> 主文”的顺序交替运行 XeLaTeX，直到两份 `.aux` 文件稳定，并在结束前检查未解析引用、未解析文献引用、未导入外部标签和重复标签。

若使用 Overleaf，应先编译 `supplementary_main.tex`，再编译 `main.tex`，随后交替重编译两者，直至日志不再提示需要重编译且正文中没有 `??`。
