# LLM Regression Monitor

An automated evaluation and regression detection system for Large Language Model (LLM) applications, built with Python, Groq, Pydantic, and GitHub Actions.

## Project Overview

This project monitors how changes to LLM prompts affect classification performance.

The application classifies customer support requests into three categories:

- **Billing:** Payments, refunds, invoices, and charges.
- **Account:** Registration, password recovery, and account management.
- **General:** Contact requests, delivery questions, and other support inquiries.

Each prediction includes a category and a one-sentence summary, validated using Pydantic.

## Key Features

- Version-controlled prompts using YAML.
- Golden dataset containing 90 manually verified classification labels.
- Automated LLM evaluation through the Groq API.
- Accuracy, Macro F1, and per-category Recall metrics.
- Comparison of candidate predictions against a baseline.
- Detection of newly introduced classification errors.
- Automated PASS/FAIL regression checks.
- GitHub Actions integration for Pull Requests.
- Protected main branch requiring successful checks.

## Evaluation Results

| Metric | Prompt v1 | Prompt v2 |
|---|---|---|
| Accuracy | 97.78% | 97.78% |
| Macro F1 | 0.9776 | 0.9778 |
| Billing Recall | 100% | 93.33% |
| Account Recall | 93.33% | 100% |
| General Recall | 100% | 100% |

Although v2 maintained the same overall accuracy, it introduced two new classification errors (IDs 3 and 26).

**Regression status: FAIL**

This demonstrates why overall accuracy alone is insufficient for detecting regressions.

## Technology Stack

Python · Groq API · Pydantic · scikit-learn · pandas · PyYAML · GitHub Actions

## CI Workflow

1. A developer updates the prompt or evaluation code in the `dev` branch.
2. A Pull Request is opened against `main`.
3. GitHub Actions executes the evaluation against the golden dataset.
4. The system compares candidate results with the baseline.
5. The regression gate returns PASS or FAIL.
6. Evaluation reports are uploaded as GitHub Actions artifacts.

A failed required check prevents merging into the protected `main` branch.

## Repository Structure

```text
llm-regression-monitor/
├── .github/workflows/regression.yml
├── data/golden_dataset.csv
├── prompts/
│   ├── v1.yaml
│   └── v2.yaml
├── results/
│   ├── v1_predictions.csv
│   ├── v2_predictions.csv
│   └── regression_report.json
├── src/
│   ├── classifier.py
│   ├── evaluator.py
│   └── regression.py
├── run_evaluation.py
├── run_live_evaluation.py
├── requirements.txt
└── README.md
```

## Security

API credentials are managed through GitHub Actions Secrets and environment variables. Secrets are not committed to the repository.

## Limitations

The golden dataset consists of 90 short, synthetic customer support requests. Results should not be interpreted as evidence of production-level reliability.

The current evaluation focuses on classification performance. Summary quality is not yet evaluated against manually verified reference summaries.

## Future Improvements

- Expand the evaluation dataset.
- Add independent holdout testing.
- Introduce summary-quality evaluation.
- Add historical performance tracking and drift monitoring.
- Package the application using Docker.
