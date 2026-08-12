from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .common import (
        SPECS,
        bh_adjust,
        build_query_frame,
        load_strict_folds,
        make_standard_folds,
        paired_comparison,
        per_task_metrics,
        query_id,
        read_benchmark,
        tool_allele,
    )
    from .import_scores import import_mhcflurry, import_netmhcpan
except ImportError:
    from common import (
        SPECS,
        bh_adjust,
        build_query_frame,
        load_strict_folds,
        make_standard_folds,
        paired_comparison,
        per_task_metrics,
        query_id,
        read_benchmark,
        tool_allele,
    )
    from import_scores import import_mhcflurry, import_netmhcpan


def tiny_pairs() -> pd.DataFrame:
    rows = []
    for task, tissue, mhc in (
        ("T1||HLA-A*02:01", "T1", "HLA-A*02:01"),
        ("T2||HLA-A*03:01", "T2", "HLA-A*03:01"),
    ):
        for index in range(3):
            pair = f"{task}-p{index}"
            rows.extend(
                [
                    {
                        "sample_id": pair + "+",
                        "pair_id": pair,
                        "target_tissue": tissue,
                        "mhc_restriction": mhc,
                        "peptide_sequence": "AAAAAAAAA",
                        "label": 1,
                        "task_name": task,
                        "score": 0.9 - index * 0.1,
                    },
                    {
                        "sample_id": pair + "-",
                        "pair_id": pair,
                        "target_tissue": tissue,
                        "mhc_restriction": mhc,
                        "peptide_sequence": "CCCCCCCCC",
                        "label": 0,
                        "task_name": task,
                        "score": 0.1 + index * 0.1,
                    },
                ]
            )
    return pd.DataFrame(rows)


def test_metrics() -> None:
    frame = tiny_pairs()
    metrics, coverage = per_task_metrics(frame, "score", "test")
    assert len(metrics) == 2
    assert np.allclose(metrics["auroc"], 1.0)
    assert np.allclose(metrics["pair_acc"], 1.0)
    assert coverage["full_coverage"].all()
    frame.loc[frame["pair_id"].eq(frame["pair_id"].iloc[0]), "score"] = np.nan
    metrics, coverage = per_task_metrics(frame, "score", "test")
    assert coverage["n_complete_pairs"].sum() == 5


def test_statistics() -> None:
    external = pd.DataFrame({"task_name": ["a", "b", "c"], "auroc": [0.6, 0.7, 0.8]})
    main = pd.DataFrame({"task_name": ["a", "b", "c"], "auroc": [0.7, 0.8, 0.9]})
    summary, deltas = paired_comparison(
        external, main, "auroc", bootstrap_iterations=100
    )
    assert np.isclose(summary["mean_difference"], 0.1)
    assert summary["wins"] == 3
    assert len(deltas) == 3
    adjusted = bh_adjust(pd.Series([0.01, 0.04, 0.03]))
    assert (adjusted >= pd.Series([0.01, 0.04, 0.03])).all()


def test_names() -> None:
    assert tool_allele("H2-Kb") == "H-2-Kb"
    assert tool_allele("HLA-A*02:01") == "HLA-A*02:01"
    assert query_id("human", "AAAAAAAAA", "HLA-A*02:01") == query_id(
        "human", "AAAAAAAAA", "HLA-A*02:01"
    )


def test_importers() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        queries = pd.DataFrame(
            [
                {
                    "query_id": "q1",
                    "species": "human",
                    "peptide_sequence": "AAAAAAAAA",
                    "mhc_restriction": "HLA-A*02:01",
                    "tool_allele": "HLA-A*02:01",
                },
                {
                    "query_id": "q2",
                    "species": "human",
                    "peptide_sequence": "CCCCCCCCC",
                    "mhc_restriction": "HLA-A*02:01",
                    "tool_allele": "HLA-A*02:01",
                },
            ]
        )
        query_path = root / "queries.csv"
        queries.to_csv(query_path, index=False)
        flurry = pd.DataFrame(
            {
                "query_id": ["q1", "q2"],
                "mhcflurry_affinity": [10.0, np.nan],
                "mhcflurry_affinity_percentile": [0.1, np.nan],
                "mhcflurry_presentation_score": [0.9, np.nan],
            }
        )
        flurry_path = root / "flurry.csv"
        flurry.to_csv(flurry_path, index=False)
        flurry_cache = root / "flurry_cache.csv"
        import_mhcflurry(query_path, flurry_path, flurry_cache, "test")
        imported = pd.read_csv(flurry_cache)
        assert set(imported["scoring_mode"]) == {
            "affinity_nm",
            "affinity_percentile",
            "presentation_score",
        }
        assert imported.groupby("scoring_mode")["is_supported"].sum().eq(1).all()

        table = pd.DataFrame(
            {
                "Peptide": ["AAAAAAAAA", "CCCCCCCCC"],
                "EL-score": [0.9, 0.1],
                "%Rank_EL": [0.1, 5.0],
                "%Rank_BA": [0.2, 6.0],
                "Aff(nM)": [20.0, 2000.0],
            }
        )
        output = root / "HLA_A_02_01.xls"
        output.write_text("\t\t\tHLA-A02:01\n", encoding="utf-8")
        table.to_csv(output, sep="\t", index=False, mode="a")
        manifest = pd.DataFrame(
            [
                {
                    "mhc_restriction": "HLA-A*02:01",
                    "expected_output": str(output),
                }
            ]
        )
        manifest_path = root / "manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        net_cache = root / "net_cache.csv"
        import_netmhcpan(query_path, manifest_path, net_cache, "4.1-test")
        imported = pd.read_csv(net_cache)
        assert {"ba_rank", "el_rank", "affinity_nm", "el_score"} == set(
            imported["scoring_mode"]
        )
        assert imported["is_supported"].all()



def test_real_inputs() -> None:
    expected = {"human": (79759, 35), "mouse": (6663, 4)}
    for species, (n_queries, n_alleles) in expected.items():
        spec = SPECS[species]
        train = read_benchmark(spec.train, species, "train")
        queries = build_query_frame(species)
        assert len(queries) == n_queries
        assert queries["mhc_restriction"].nunique() == n_alleles
        standard = make_standard_folds(train)
        assert set(standard) == {0, 1, 2}
        strict = load_strict_folds(train, spec.strict_manifest)
        assert set(strict) == {0, 1, 2}


def main() -> None:
    test_metrics()
    test_statistics()
    test_names()
    test_importers()
    test_real_inputs()
    print("Issue 5 tests passed.")


if __name__ == "__main__":
    main()
