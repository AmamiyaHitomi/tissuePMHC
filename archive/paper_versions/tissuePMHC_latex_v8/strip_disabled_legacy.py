"""Remove disabled legacy-result blocks from the v7 LaTeX sources."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent


def strip_iffalse(text: str) -> str:
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    depth = 0
    for line in lines:
        token = line.strip()
        if token == r"\iffalse":
            depth += 1
            continue
        if token == r"\fi" and depth:
            depth -= 1
            continue
        if depth == 0:
            output.append(line)
    if depth:
        raise ValueError("Unclosed \\iffalse block")
    return "".join(output)


for relative in (
    "sections/03_problem_formulation_and_benchmark.tex",
    "sections/06_results.tex",
):
    path = ROOT / relative
    path.write_text(strip_iffalse(path.read_text(encoding="utf-8")), encoding="utf-8")

