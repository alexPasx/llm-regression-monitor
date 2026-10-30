
from src.evaluator import calculate_metrics, LABELS


def check_regression(
    baseline_df,
    candidate_df,
    max_accuracy_drop=0.01,
    max_f1_drop=0.01,
    max_recall_drop=0.05
):
    comparison = baseline_df[
        ["id", "expected", "predicted"]
    ].merge(
        candidate_df[["id", "expected", "predicted"]],
        on="id",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one"
    )

    if len(comparison) != len(baseline_df) or len(comparison) != len(candidate_df):
        raise ValueError("Datasets contain different IDs.")

    if not (
        comparison["expected_baseline"]
        == comparison["expected_candidate"]
    ).all():
        raise ValueError("Expected labels do not match.")

    baseline_metrics = calculate_metrics(baseline_df)
    candidate_metrics = calculate_metrics(candidate_df)

    accuracy_drop = (
        baseline_metrics["accuracy"]
        - candidate_metrics["accuracy"]
    )

    f1_drop = (
        baseline_metrics["macro_f1"]
        - candidate_metrics["macro_f1"]
    )

    recall_drops = {
        label: (
            baseline_metrics["recall_per_category"][label]
            - candidate_metrics["recall_per_category"][label]
        )
        for label in LABELS
    }

    new_errors = comparison[
        (comparison["predicted_baseline"]
         == comparison["expected_baseline"])
        &
        (comparison["predicted_candidate"]
         != comparison["expected_candidate"])
    ]

    checks = {
        "accuracy": accuracy_drop <= max_accuracy_drop,
        "macro_f1": f1_drop <= max_f1_drop,
        "recall": all(
            drop <= max_recall_drop
            for drop in recall_drops.values()
        ),
        "new_errors": len(new_errors) == 0
    }

    return {
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "accuracy_drop": float(accuracy_drop),
        "macro_f1_drop": float(f1_drop),
        "recall_drops": recall_drops,
        "new_error_ids": new_errors["id"].tolist(),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL"
    }
