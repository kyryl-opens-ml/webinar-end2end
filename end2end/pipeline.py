from dagster import AssetExecutionContext, AssetOut, Config, MetadataValue, Output, asset, multi_asset
from datasets import concatenate_datasets

from end2end.data import ARGILLA_KEY, ARGILLA_NAMESPACE, ARGILLA_URI, load_data_for_labeling, load_text_to_sql_dataset
from end2end.experiments import (
    AutoTokenizer,
    ScriptArguments,
    Seq2SeqTrainingArguments,
    eval_model,
    get_argilla_dataset_formatted,
    get_model,
    train_model,
)


@multi_asset(
    outs={
        "df_train": AssetOut(),
        "df_val": AssetOut(),
    },
    group_name="data",
)
def origin_text2sql_dataset(context: AssetExecutionContext):
    df_train, df_val = load_text_to_sql_dataset()

    context.add_output_metadata({"df_train": MetadataValue.md(df_train.head().to_markdown())}, output_name="df_train")
    context.add_output_metadata({"df_val": MetadataValue.md(df_train.head().to_markdown())}, output_name="df_val")

    return df_train, df_val


class LabelingDataConfig(Config):
    dataset_name: str = "pipeline-dataset"
    sample: bool = True
    num_sample: int = 1000


@asset(group_name="data")
def labeling_data(context: AssetExecutionContext, config: LabelingDataConfig, df_train, df_val) -> str:
    dataset_name = f"{config.dataset_name}-{context.run_id}"
    url_train, url_val = load_data_for_labeling(
        dataset_name=dataset_name, sample=config.sample, num_sample=config.num_sample
    )

    context.add_output_metadata(
        {
            "url_train": MetadataValue.url(url_train),
            "url_val": MetadataValue.url(url_val),
        }
    )
    return dataset_name


@asset(group_name="data")
def test_dataset(context: AssetExecutionContext, labeling_data: str):
    dataset_name_full = f"{labeling_data}-val"
    dataset = get_argilla_dataset_formatted(
        dataset_name_full=dataset_name_full, api_url=ARGILLA_URI, api_key=ARGILLA_KEY, workspace=ARGILLA_NAMESPACE
    )
    return dataset


@asset(group_name="data")
def train_dataset(context: AssetExecutionContext, labeling_data: str):
    dataset_name_full = f"{labeling_data}-train"
    dataset = get_argilla_dataset_formatted(
        dataset_name_full=dataset_name_full, api_url=ARGILLA_URI, api_key=ARGILLA_KEY, workspace=ARGILLA_NAMESPACE
    )
    return dataset


@asset(group_name="data")
def feedback_dataset():
    dataset_name_full = "feedback-open-model"
    dataset = get_argilla_dataset_formatted(
        dataset_name_full=dataset_name_full, api_url=ARGILLA_URI, api_key=ARGILLA_KEY, workspace=ARGILLA_NAMESPACE
    )
    return dataset


@asset(group_name="model")
def training_args():
    return Seq2SeqTrainingArguments(
        output_dir="result-flan-t5-small",  # Output directory for model checkpoints
        overwrite_output_dir=True,  # Overwrite the content of the output directory
        do_train=True,  # Run training
        do_eval=True,  # Run evaluation
        evaluation_strategy="steps",  # Evaluation is done (and logged) every logging_steps
        per_device_train_batch_size=16,  # Batch size for training
        per_device_eval_batch_size=16,  # Batch size for evaluation
        learning_rate=1e-3,  # Learning rate
        # num_train_epochs=10.0,                    # Total number of training epochs
        num_train_epochs=0.0001,  # Total number of training epochs
        hub_model_id="kyryl-opens-ml/flan-t5-small-sql",
        hub_token="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",
    )


@asset(group_name="model")
def script_args():
    return ScriptArguments()


@asset(group_name="model")
def pre_trained_llm(script_args):
    return get_model(model_name=script_args.model_name)


@asset(group_name="model")
def tokenizer(script_args):
    return AutoTokenizer.from_pretrained(script_args.model_name)


@asset(group_name="model")
def trained_model(
    context: AssetExecutionContext,
    train_dataset,
    test_dataset,
    tokenizer,
    pre_trained_llm,
    training_args: Seq2SeqTrainingArguments,
    script_args,
):
    train_model(
        dataset={"train": train_dataset, "test": test_dataset},
        tokenizer=tokenizer,
        model=pre_trained_llm,
        training_args=training_args,
        script_args=script_args,
    )

    metrics = eval_model(test_data=test_dataset.select(list(range(100))), model_name=training_args.output_dir)
    metadata = {name: MetadataValue.float(float(value)) for name, value in metrics.items()}
    context.add_output_metadata(metadata)


@asset(group_name="model")
def trained_model_with_feedback(
    context: AssetExecutionContext,
    train_dataset,
    feedback_dataset,
    test_dataset,
    tokenizer,
    pre_trained_llm,
    training_args,
    script_args,
):
    autgmented_dataset = concatenate_datasets([train_dataset, feedback_dataset])
    train_model(
        dataset={"train": autgmented_dataset, "test": test_dataset},
        tokenizer=tokenizer,
        model=pre_trained_llm,
        training_args=training_args,
        script_args=script_args,
    )
    metrics = eval_model(test_data=test_dataset.select(list(range(100))), model_name=training_args.output_dir)
    metadata = {name: MetadataValue.float(float(value)) for name, value in metrics.items()}
    context.add_output_metadata(metadata)
