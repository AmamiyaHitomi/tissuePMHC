"""End-to-end external-evaluation smoke test using deterministic synthetic scores.

Synthetic scores are strictly for code validation and are written only to a
temporary directory. They must never be used as scientific results.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

try:
    from .common import build_query_frame
    from .evaluate_external import evaluate
    from .stack_increment import stack
except ImportError:
    from common import build_query_frame
    from evaluate_external import evaluate
    from stack_increment import stack


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rows = []
        for species in ("human", "mouse"):
            queries = build_query_frame(species)
            cache = queries[
                ["query_id", "species", "peptide_sequence", "mhc_restriction"]
            ].copy()
            cache["predictor"] = "synthetic_smoke_only"
            cache["scoring_mode"] = "deterministic_hash"
            cache["score"] = cache["query_id"].map(
                lambda value: int(str(value)[:12], 16) / float(16**12 - 1)
            )
            cache["is_supported"] = True
            cache["missing_reason"] = ""
            rows.append(cache)
        cache_path = root / "synthetic_cache.csv.gz"
        pd.concat(rows, ignore_index=True).to_csv(cache_path, index=False)
        output = root / "evaluation"
        evaluate([cache_path], output, bootstrap_iterations=20)
        summary = pd.read_csv(output / "summary_metrics.csv")
        external = summary[
            summary["evaluated_model"].eq(
                "synthetic_smoke_only::deterministic_hash"
            )
        ]
        assert len(external) == 6
        assert set(external["species"]) == {"human", "mouse"}
        assert set(external["protocol"]) == {
            "standard_fixed_test",
            "matched_standard_oof",
            "peptide_disjoint_oof",
        }
        stack_output = root / "stack"
        stack(
            output / "row_predictions.csv.gz",
            stack_output,
            bootstrap_iterations=20,
        )
        stack_summary = pd.read_csv(stack_output / "summary_metrics.csv")
        assert len(stack_summary) == 12
        assert set(stack_summary["stacker"]) == {
            "external_only_stacker",
            "external_plus_tissue_stacker",
        }
        print("Issue 5 external-evaluation smoke test passed.")


if __name__ == "__main__":
    main()
