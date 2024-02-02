from dataclasses import dataclass, field
from typing import Dict, Tuple

import argilla as rg
import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def get_argilla_dataset_formatted(dataset_name_full: str, api_url: str, api_key: str, workspace: str) -> Dataset:
    rg.init(api_url=api_url, api_key=api_key, workspace=workspace)

    remote_dataset = rg.FeedbackDataset.from_argilla(dataset_name_full)
    local_dataset = remote_dataset.pull()
    df = local_dataset.format_as("datasets").to_pandas()
    prompt = "### SQL schema: {sql_schema} ### User query: {user_query} ### SQL query:"

    format_data = []
    for idx in range(df.shape[0]):
        sample = df.iloc[idx]
        format_data.append(
            {
                "input_text": prompt.format(sql_schema=sample["sql_schema"], user_query=sample["user_query"]),
                "output_text": sample["sql_query"],
            }
        )
    dataset = Dataset.from_pandas(pd.DataFrame(format_data))
    return dataset


def get_dataset_argilla(dataset_name: str, api_url: str, api_key: str, workspace: str) -> Dict[str, Dataset]:

    dataset_train = get_argilla_dataset_formatted(
        dataset_name_full=f"{dataset_name}-train", api_url=api_url, api_key=api_key, workspace=workspace
    )
    dataset_val = get_argilla_dataset_formatted(
        dataset_name_full=f"{dataset_name}-val", api_url=api_url, api_key=api_key, workspace=workspace
    )

    return {"train": dataset_train, "test": dataset_val}


def get_max_len(dataset, tokenizer) -> Tuple[int, int]:
    tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(
        lambda x: tokenizer(x["input_text"], truncation=True),
        batched=True,
        remove_columns=["input_text", "output_text"],
    )
    input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
    max_source_lengths = int(np.percentile(input_lenghts, 95))

    tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(
        lambda x: tokenizer(x["output_text"], truncation=True),
        batched=True,
        remove_columns=["input_text", "output_text"],
    )
    target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
    max_target_lengths = int(np.percentile(target_lenghts, 95))

    return max_source_lengths, max_target_lengths


@dataclass
class ScriptArguments:
    model_name: str = field(default="google/flan-t5-small", metadata={"help": "the model name"})

    dataset_name: str = field(default="text2sql-small", metadata={"help": "the dataset name"})
    api_url: str = field(default="http://3.80.58.157:6900", metadata={"help": "argilla url"})
    api_key: str = field(default="adminadmin", metadata={"help": "argilla key"})
    workspace: str = field(default="admin", metadata={"help": "argilla workspace"})


def get_model(model_name: str) -> AutoModelForSeq2SeqLM:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")

    # Define LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.2,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


@torch.inference_mode()
def eval_model(test_data: Dataset, model_name: str):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="cuda:0")

    predictions = []

    for idx in tqdm(range(len(test_data))):

        input_text = "Generate SQL query based on next information: " + test_data[idx]["input_text"]
        input_ids = tokenizer(input_text, return_tensors="pt").to("cuda:0")
        outputs = model.generate(**input_ids, max_length=100)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(prediction)

    metric = evaluate.load("rouge")
    references = test_data["output_text"]
    rogue = metric.compute(predictions=predictions, references=references, use_stemmer=True)
    print(f"rogue = {rogue}")
    return rogue


def train_model(dataset, tokenizer, model, training_args, script_args):
    max_source_lengths, max_target_lengths = get_max_len(dataset=dataset, tokenizer=tokenizer)
    print(f"Max source lengths: {max_source_lengths}")
    print(f"Max target lengths: {max_target_lengths}")

    def preprocess_function(sample, padding="max_length"):
        # add prefix to the input for t5
        inputs = [item for item in sample["input_text"]]
        # tokenize inputs
        model_inputs = tokenizer(inputs, max_length=max_source_lengths, padding=padding, truncation=True)
        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=sample["output_text"],
            max_length=max_target_lengths,
            padding=padding,
            truncation=True,
        )
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = {}
    tokenized_dataset["train"] = dataset["train"].map(
        preprocess_function, batched=True, remove_columns=["input_text", "output_text"]
    )
    tokenized_dataset["test"] = dataset["test"].map(
        preprocess_function, batched=True, remove_columns=["input_text", "output_text"]
    )
    print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )
    # silence the warnings. Please re-enable for inference!
    model.config.use_cache = False

    train_result = trainer.train()
    trainer.save_model()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    metrics = trainer.evaluate(metric_key_prefix="eval")
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": script_args.model_name, "tasks": "text2sql"}
    trainer.push_to_hub(**kwargs)


def main():
    parser = HfArgumentParser((ScriptArguments, Seq2SeqTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    dataset = get_dataset_argilla(
        dataset_name=script_args.dataset_name,
        api_url=script_args.api_url,
        api_key=script_args.api_key,
        workspace=script_args.workspace,
    )
    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Test dataset size: {len(dataset['test'])}")
    print(f"training_args = {training_args}")

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    model = get_model(model_name=script_args.model_name)
    train_model(dataset=dataset, tokenizer=tokenizer, model=model, training_args=training_args, script_args=script_args)
    eval_model(test_data=dataset["test"].select(list(range(100))), model_name=training_args.output_dir)


if __name__ == "__main__":
    main()
