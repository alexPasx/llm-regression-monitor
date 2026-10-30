
import json
import os
from dataclasses import dataclass
from typing import Any, Literal

from groq import Groq
from pydantic import BaseModel, Field, ConfigDict


class EmailClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Literal["billing", "account", "general"]
    summary: str = Field(min_length=1)


@dataclass
class PromptConfig:
    version: str
    created_at: str
    system_prompt: str
    few_shot_examples: list[dict[str, Any]]


def classify_email(
    email_text: str,
    config: PromptConfig,
    client: Groq,
) -> EmailClassification:

    if not email_text.strip():
        raise ValueError("Email text cannot be empty.")

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    config.system_prompt
                    + "\nReturn only a JSON object with category and summary."
                ),
            },
            {
                "role": "user",
                "content": email_text,
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    raw_output = response.choices[0].message.content
    parsed_output = json.loads(raw_output)

    return EmailClassification.model_validate(parsed_output)
