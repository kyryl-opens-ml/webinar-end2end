from typing import Tuple

import argilla as rg
import pandas as pd
import typer
from datasets import load_dataset
from pydantic_settings import BaseSettings, SettingsConfigDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    ARGILLA_URI: str
    ARGILLA_KEY: str
    ARGILLA_NAMESPACE: str


ARGILLA_URI = Settings().ARGILLA_URI
ARGILLA_KEY = Settings().ARGILLA_KEY
ARGILLA_NAMESPACE = Settings().ARGILLA_NAMESPACE


def load_text_to_sql_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading sql-create-context dataset")
    dataset = load_dataset("b-mc2/sql-create-context")
    print(f"Loaded dataset {dataset}")

    print("Split to train & test")
    df = dataset["train"].to_pandas()
    df = df.rename(columns={"answer": "sql_query", "context": "sql_schema", "question": "user_query"})

    df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Train size {df_train.shape} {df_train.columns}, Validation size {df_val.shape} {df_val.columns}")

    return df_train, df_val


def _upload_from_pandas(df: pd.DataFrame, dataset_name: str) -> str:
    dataset = rg.FeedbackDataset(
        guidelines="Text to SQL.",
        fields=[
            rg.TextField(name="user_query", title="Text from user"),
            rg.TextField(name="sql_query", title="SQL query"),
            rg.TextField(name="sql_schema", title="SQL schema"),
        ],
        questions=[
            rg.TextQuestion(
                name="corrected_sql_query",
                title="Provide a correction to the query.",
                required=False,
                use_markdown=True,
            ),
            rg.TextQuestion(
                name="corrected_sql_schema",
                title="Provide a correction to the table schema.",
                required=False,
                use_markdown=True,
            ),
            rg.LabelQuestion(name="correct", title="Is sample correct", labels=["true", "false"]),
        ],
    )

    records = []
    for idx in tqdm(range(len(df))):
        sample = df.iloc[idx]
        record = rg.FeedbackRecord(fields=sample)
        records.append(record)

    dataset.add_records(records)
    res = dataset.push_to_argilla(name=dataset_name)
    return res.url


def load_data_for_labeling(
    dataset_name: str = "text2sql", sample: bool = False, num_sample: int = 10_000
) -> Tuple[str, str]:
    df_train, df_val = load_text_to_sql_dataset()
    if sample:
        df_train = df_train.sample(n=num_sample)

    rg.init(api_url=ARGILLA_URI, api_key=ARGILLA_KEY, workspace=ARGILLA_NAMESPACE)
    url_train = _upload_from_pandas(df=df_train, dataset_name=f"{dataset_name}-train")
    print(f"url_train = {url_train}")
    url_val = _upload_from_pandas(df=df_val, dataset_name=f"{dataset_name}-val")
    print(f"url_val = {url_val}")
    return url_train, url_val


def cli():
    app = typer.Typer()
    app.command()(load_text_to_sql_dataset)
    app.command()(load_data_for_labeling)
    app()


if __name__ == "__main__":
    cli()
