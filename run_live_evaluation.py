
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from groq import Groq

from src.classifier import PromptConfig
from src.evaluator import evaluate_dataset
from src.regression import check_regression


def main():
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing.")

    client = Groq(api_key=api_key)

    with open("prompts/v2.yaml", encoding="utf-8") as f:
        config = PromptConfig(**yaml.safe_load(f))

    dataset = pd.read_csv("data/golden_dataset.csv")
    baseline = pd.read_csv("results/v1_predictions.csv")

    predictions = evaluate_dataset(dataset, config, client)

    Path("results").mkdir(exist_ok=True)
    predictions.to_csv("results/live_predictions.csv", index=False)

    report = check_regression(baseline, predictions)

    with open("results/live_regression_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("Prompt version:", config.version)
    print("Regression status:", report["status"])
    print("New errors:", report["new_error_ids"])

    if report["status"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
