from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TASK_COLUMNS = ["target_tissue", "mhc_restriction"]
TARGET_TEST_PAIRS_PER_TASK = 50


def read_partition(path: Path) -> pd.DataFrame:
    # "NA" is a real tissue label in this benchmark, not a missing value.
    return pd.read_csv(path, keep_default_na=False)


class DSU:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def pair_table(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby("pair_id", sort=False)
    pair_sizes = grouped.size()
    if not pair_sizes.eq(2).all():
        raise ValueError("Every pair_id must contain exactly two rows.")
    label_sets = grouped["label"].agg(lambda values: tuple(sorted(values.astype(int))))
    if not label_sets.map(lambda labels: labels == (0, 1)).all():
        raise ValueError("Every pair_id must contain one positive and one negative row.")
    task_counts = grouped[TASK_COLUMNS].nunique(dropna=False)
    if not task_counts.eq(1).to_numpy().all():
        raise ValueError("Both rows of every pair must have the same tissue-HLA task.")

    pairs = grouped.agg(
        target_tissue=("target_tissue", "first"),
        mhc_restriction=("mhc_restriction", "first"),
    ).reset_index()
    pairs["pair_id"] = pairs["pair_id"].astype(str)
    pairs["task"] = pairs["target_tissue"].astype(str) + "||" + pairs["mhc_restriction"].astype(str)

    # A pair is atomic. Connect repeated peptide occurrences only when their
    # labels agree. Opposite-label reuse across different tasks remains legal.
    pair_peptides = rows[["pair_id", "peptide_sequence", "label"]].drop_duplicates().copy()
    pair_peptides["pair_id"] = pair_peptides["pair_id"].astype(str)
    dsu = DSU(pairs["pair_id"].tolist())
    for _, peptide_group in pair_peptides.groupby(
        ["peptide_sequence", "label"], sort=False
    ):
        pair_ids = peptide_group["pair_id"].tolist()
        for pair_id in pair_ids[1:]:
            dsu.union(pair_ids[0], pair_id)

    # Exact model inputs may not cross the split even if the source data ever
    # assigns conflicting labels to the same tissue-HLA-peptide query.
    exact_inputs = rows[
        ["pair_id", "target_tissue", "mhc_restriction", "peptide_sequence"]
    ].drop_duplicates().copy()
    exact_inputs["pair_id"] = exact_inputs["pair_id"].astype(str)
    for _, input_group in exact_inputs.groupby(
        ["target_tissue", "mhc_restriction", "peptide_sequence"], sort=False
    ):
        pair_ids = input_group["pair_id"].tolist()
        for pair_id in pair_ids[1:]:
            dsu.union(pair_ids[0], pair_id)
    pairs["component_id"] = pairs["pair_id"].map(dsu.find)
    return pairs


def choose_test_components(pairs: pd.DataFrame, seed: int) -> tuple[set[str], pd.DataFrame]:
    components: list[dict[str, object]] = []
    for component_id, group in pairs.groupby("component_id", sort=False):
        counts = group["task"].value_counts().to_dict()
        components.append(
            {
                "component_id": str(component_id),
                "pairs": int(len(group)),
                "task_counts": counts,
                "private_singleton": len(group) == 1,
            }
        )

    tasks = sorted(pairs["task"].unique())
    target = {task: TARGET_TEST_PAIRS_PER_TASK for task in tasks}
    test_fraction = (TARGET_TEST_PAIRS_PER_TASK * len(tasks)) / len(pairs)

    singleton_by_task: dict[str, list[str]] = {task: [] for task in tasks}
    for component in components:
        if component["private_singleton"]:
            only_task = next(iter(component["task_counts"]))
            singleton_by_task[only_task].append(str(component["component_id"]))

    # Sample non-singleton peptide components at the overall target fraction.
    # Whole components are selected, so a peptide can never cross the split.
    non_singletons = [c for c in components if not c["private_singleton"]]
    selected: set[str] | None = None
    selected_counts: dict[str, int] | None = None
    rng: np.random.Generator | None = None
    for attempt in range(100):
        attempt_rng = np.random.default_rng(seed + attempt)
        attempt_selected: set[str] = set()
        attempt_counts = {task: 0 for task in tasks}
        for index in attempt_rng.permutation(len(non_singletons)):
            component = non_singletons[int(index)]
            if attempt_rng.random() >= test_fraction:
                continue
            counts = component["task_counts"]
            if any(attempt_counts[task] + count > target[task] for task, count in counts.items()):
                continue
            attempt_selected.add(str(component["component_id"]))
            for task, count in counts.items():
                attempt_counts[task] += int(count)
        if all(
            target[task] - attempt_counts[task] <= len(singleton_by_task[task])
            for task in tasks
        ):
            selected = attempt_selected
            selected_counts = attempt_counts
            rng = attempt_rng
            break
    if selected is None or selected_counts is None or rng is None:
        shortages = {
            task: len(singleton_by_task[task])
            for task in tasks
            if len(singleton_by_task[task]) < TARGET_TEST_PAIRS_PER_TASK
        }
        raise RuntimeError(
            "Could not construct an exactly balanced peptide-disjoint split after "
            f"100 deterministic attempts. Singleton-component shortages: {shortages}"
        )

    # Private singleton components provide exact per-task balancing without
    # changing any previously selected peptide component.
    for task in tasks:
        need = target[task] - selected_counts[task]
        candidates = singleton_by_task[task]
        if len(candidates) < need:
            raise RuntimeError(f"Task {task} needs {need} singleton peptides but only {len(candidates)} are available.")
        chosen = rng.choice(np.asarray(candidates, dtype=object), size=need, replace=False).tolist()
        selected.update(chosen)
        selected_counts[task] += need

    if selected_counts != target:
        raise AssertionError("Failed to reach the exact 50-pair target for every task.")

    component_rows = []
    for component in components:
        component_rows.append(
            {
                "component_id": component["component_id"],
                "pairs": component["pairs"],
                "tasks": len(component["task_counts"]),
                "private_singleton": component["private_singleton"],
                "assigned_split": "test" if component["component_id"] in selected else "train",
            }
        )
    return selected, pd.DataFrame(component_rows)


def reset_partition_metadata(rows: pd.DataFrame, split: str) -> pd.DataFrame:
    result = rows.copy().reset_index(drop=True)
    result["split"] = split
    result["sample_id"] = [f"humanPMHC_{split}_{i:012d}" for i in range(1, len(result) + 1)]
    return result


def validate_and_summarize(
    train: pd.DataFrame, test: pd.DataFrame, seed: int
) -> tuple[dict[str, object], pd.DataFrame]:
    train_pairs = pair_table(train)
    test_pairs = pair_table(test)
    train_peptides = set(train["peptide_sequence"].astype(str))
    test_peptides = set(test["peptide_sequence"].astype(str))
    train_labeled_peptides = set(
        map(tuple, train[["peptide_sequence", "label"]].drop_duplicates().to_numpy())
    )
    test_labeled_peptides = set(
        map(tuple, test[["peptide_sequence", "label"]].drop_duplicates().to_numpy())
    )
    exact_columns = ["target_tissue", "mhc_restriction", "peptide_sequence"]
    train_exact_inputs = set(map(tuple, train[exact_columns].drop_duplicates().to_numpy()))
    test_exact_inputs = set(map(tuple, test[exact_columns].drop_duplicates().to_numpy()))
    pair_overlap = set(train_pairs["pair_id"]) & set(test_pairs["pair_id"])
    peptide_overlap = train_peptides & test_peptides
    labeled_peptide_overlap = train_labeled_peptides & test_labeled_peptides
    exact_input_overlap = train_exact_inputs & test_exact_inputs

    train_labels_by_peptide = train.groupby("peptide_sequence")["label"].agg(
        lambda values: frozenset(map(int, values))
    )
    test_labels_by_peptide = test.groupby("peptide_sequence")["label"].agg(
        lambda values: frozenset(map(int, values))
    )
    opposite_only_peptides = {
        peptide
        for peptide in peptide_overlap
        if train_labels_by_peptide[peptide].isdisjoint(test_labels_by_peptide[peptide])
    }

    train_labeled_queries = set(
        map(
            tuple,
            train[["peptide_sequence", "mhc_restriction", "label"]]
            .drop_duplicates()
            .to_numpy(),
        )
    )
    train_query_labels = train.groupby(
        ["peptide_sequence", "mhc_restriction"]
    )["label"].agg(lambda values: frozenset(map(int, values)))
    test_categories: list[str] = []
    for row in test.itertuples(index=False):
        labels = train_query_labels.get(
            (row.peptide_sequence, row.mhc_restriction), frozenset()
        )
        if not labels:
            category = "query_absent"
        elif labels == frozenset([int(row.label)]):
            category = "same_label_only"
        elif labels == frozenset([1 - int(row.label)]):
            category = "opposite_label_only"
        else:
            category = "both_labels"
        test_categories.append(category)
    category_counts = pd.Series(test_categories).value_counts().to_dict()

    inventory = (
        train_pairs.groupby(TASK_COLUMNS, dropna=False).size().rename("train_pairs").reset_index()
        .merge(
            test_pairs.groupby(TASK_COLUMNS, dropna=False).size().rename("test_pairs").reset_index(),
            on=TASK_COLUMNS,
            how="outer",
            validate="one_to_one",
        )
        .fillna(0)
    )
    inventory[["train_pairs", "test_pairs"]] = inventory[["train_pairs", "test_pairs"]].astype(int)

    errors = {
        "pair_overlap": len(pair_overlap),
        "same_label_peptide_overlap": len(labeled_peptide_overlap),
        "exact_tissue_hla_peptide_overlap": len(exact_input_overlap),
        "tasks_without_50_test_pairs": int((inventory["test_pairs"] != TARGET_TEST_PAIRS_PER_TASK).sum()),
        "tasks_missing_from_train": int((inventory["train_pairs"] == 0).sum()),
        "duplicate_sample_ids": int(pd.concat([train, test])["sample_id"].duplicated().sum()),
    }
    if any(errors.values()):
        raise AssertionError(f"Split validation failed: {errors}")

    summary = {
        "split_type": "same-label peptide-disjoint",
        "protocol": (
            "pair-atomic; identical peptide+label cannot cross splits; identical "
            "tissue+HLA+peptide cannot cross splits; opposite-label peptide reuse "
            "across different tasks is allowed"
        ),
        "split_seed": int(seed),
        "n_tasks": int(len(inventory)),
        "train_rows": int(len(train)),
        "train_pairs": int(len(train_pairs)),
        "test_rows": int(len(test)),
        "test_pairs": int(len(test_pairs)),
        "test_pairs_per_task": TARGET_TEST_PAIRS_PER_TASK,
        "train_unique_peptides": int(len(train_peptides)),
        "test_unique_peptides": int(len(test_peptides)),
        "cross_split_peptide_overlap_allowed": int(len(peptide_overlap)),
        "cross_split_opposite_label_only_peptides": int(len(opposite_only_peptides)),
        "test_row_query_exposure": {
            key: int(category_counts.get(key, 0))
            for key in (
                "query_absent",
                "same_label_only",
                "opposite_label_only",
                "both_labels",
            )
        },
        "largest_full_data_component_pairs": int(
            pair_table(pd.concat([train, test], ignore_index=True))
            .groupby("component_id")
            .size()
            .max()
        ),
        "validation_errors": errors,
    }
    return summary, inventory


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=root / "data" / "humanPMHC_occurence_equal_dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "humanPMHC_occurence_equal_same_label_peptide_disjoint",
    )
    parser.add_argument("--seed", type=int, default=20260704)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    source_train = read_partition(args.input_dir / "humanPMHC_train.csv.gz")
    source_test = read_partition(args.input_dir / "humanPMHC_test.csv.gz")
    all_rows = pd.concat([source_train, source_test], ignore_index=True)
    pairs = pair_table(all_rows)
    test_components, components = choose_test_components(pairs, args.seed)
    test_pair_ids = set(pairs.loc[pairs["component_id"].isin(test_components), "pair_id"])
    pair_assignments = pairs.copy()
    pair_assignments["assigned_split"] = np.where(
        pair_assignments["component_id"].isin(test_components), "test", "train"
    )

    new_test = reset_partition_metadata(all_rows.loc[all_rows["pair_id"].isin(test_pair_ids)], "test")
    new_train = reset_partition_metadata(all_rows.loc[~all_rows["pair_id"].isin(test_pair_ids)], "train")
    summary, inventory = validate_and_summarize(new_train, new_test, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    new_train.to_csv(args.output_dir / "humanPMHC_train.csv.gz", index=False, compression="gzip")
    new_test.to_csv(args.output_dir / "humanPMHC_test.csv.gz", index=False, compression="gzip")
    inventory.to_csv(args.output_dir / "task_inventory.csv", index=False)
    components.to_csv(args.output_dir / "peptide_component_assignments.csv.gz", index=False, compression="gzip")
    pair_assignments.to_csv(
        args.output_dir / "pair_component_assignments.csv.gz",
        index=False,
        compression="gzip",
    )
    (args.output_dir / "split_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
