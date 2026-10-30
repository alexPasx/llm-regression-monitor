
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tqdm.auto import tqdm

from src.classifier import classify_email


LABELS = ["billing", "account", "general"]


def evaluate_dataset(dataset, config, client):

    results = []

    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):

        prediction = classify_email(
            email_text=row["email"],
            config=config,
            client=client
        )

        results.append({
            "id": row["id"],
            "expected": row["expected_category"],
            "predicted": prediction.category,
            "correct": prediction.category == row["expected_category"],
            "summary": prediction.summary
        })

    return pd.DataFrame(results)


def calculate_metrics(predictions_df):

    expected = predictions_df["expected"]
    predicted = predictions_df["predicted"]

    recall_values = recall_score(
        expected,
        predicted,
        labels=LABELS,
        average=None
    )

    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(
            f1_score(expected, predicted, labels=LABELS, average="macro")
        ),
        "recall_per_category": {
            label: float(value)
            for label, value in zip(LABELS, recall_values)
        }
    }
