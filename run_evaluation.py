
import json
from pathlib import Path

import pandas as pd

from src.evaluator import calculate_metrics
from src.regression import check_regression


def main():
    results_dir = Path("results")

    baseline_df = pd.read_csv(results_dir / "v1_predictions.csv")
    candidate_df = pd.read_csv(results_dir / "v2_predictions.csv")

    report = check_regression(baseline_df, candidate_df)

    with open(results_dir / "regression_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("Baseline metrics:", calculate_metrics(baseline_df))
    print("Candidate metrics:", calculate_metrics(candidate_df))
    print("Regression status:", report["status"])
    print("New error IDs:", report["new_error_ids"])


if __name__ == "__main__":
    main()
