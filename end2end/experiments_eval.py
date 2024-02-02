import json
from dataclasses import dataclass, field
from typing import Dict, Tuple

import evaluate
import numpy as np
import pandas as pd
import torch
import transformers
import typer
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_int8_training
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainingArguments,
)
from trl import ModelConfig, SFTTrainer, get_kbit_device_map, get_peft_config, get_quantization_config


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


def get_dataset() -> Dict[str, Dataset]:
    print("Loading sql-create-context dataset")
    dataset = load_dataset("b-mc2/sql-create-context")
    print(f"Loaded dataset {dataset}")

    print("Split to train & test")
    df = dataset["train"].to_pandas()
    df = df.rename(columns={"answer": "sql_query", "context": "sql_schema", "question": "user_query"})

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

    format_data = pd.DataFrame(format_data)
    train, test = train_test_split(format_data, test_size=0.1, random_state=42)
    dataset_train = Dataset.from_pandas(train)
    dataset_test = Dataset.from_pandas(test)
    return {"train": dataset_train, "test": dataset_test}


@dataclass
class ScriptArguments:
    # dataset_name: str = field(default="b-mc2/sql-create-context", metadata={"help": "the dataset name"})
    # dataset_text_field: str = field(default="text", metadata={"help": "the text field of the dataset"})
    # max_seq_length: int = field(default=512, metadata={"help": "The maximum sequence length for SFT Trainer"})
    model_name: str = field(default="google/flan-t5-small", metadata={"help": "the model name"})


def evaluate_peft_model(sample, model, tokenizer, max_target_length=512):
    # generate summary
    outputs = model.generate(
        input_ids=sample["input_ids"].unsqueeze(0).cuda(), do_sample=True, top_p=0.9, max_new_tokens=max_target_length
    )
    prediction = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
    # decode eval sample
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(sample["labels"] != -100, sample["labels"], tokenizer.pad_token_id)
    labels = tokenizer.decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    return prediction, labels


def codellama_eval(test_data: Dataset):
    model = "codellama/CodeLlama-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(model)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    predictions = []

    for idx in tqdm(range(len(test_data))):

        prompt = "Generate SQL query based on next information: " + test_data[idx]["input_text"]
        sequences = pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            temperature=0.1,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=150,
        )
        prediction = sequences[0]["generated_text"][len(prompt) :].split("\n\n")[0].strip()
        predictions.append(prediction)

    metric = evaluate.load("rouge")
    references = test_data["output_text"]
    rogue = metric.compute(predictions=predictions, references=references, use_stemmer=True)
    print(f"rogue = {rogue}")


def gemma_eval(test_data: Dataset):

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", device_map="auto")

    idx = 32
    input_text = "Generate SQL query based on next information: " + test_data[idx]["input_text"]
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = model.generate(**input_ids, max_length=100)
    print(tokenizer.decode(outputs[0]))

    predictions = []

    for idx in tqdm(range(len(test_data))):

        prompt = "Generate SQL query based on next information: " + test_data[idx]["input_text"]
        sequences = pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            temperature=0.1,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=150,
        )
        prediction = sequences[0]["generated_text"][len(prompt) :].split("\n\n")[0].strip()
        predictions.append(prediction)

    metric = evaluate.load("rouge")
    references = test_data["output_text"]
    rogue = metric.compute(predictions=predictions, references=references, use_stemmer=True)
    print(f"rogue = {rogue}")


def flan_eval(test_data: Dataset, tokenizer_name: str, model_name: str):
    print(f"Running tokenizer_name {tokenizer_name} and model_name = {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")

    predictions = []

    for idx in tqdm(range(len(test_data))):

        input_text = "Generate SQL query based on next information: " + test_data[idx]["input_text"]
        input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
        outputs = model.generate(**input_ids, max_length=100)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(prediction)

    metric = evaluate.load("rouge")
    references = test_data["output_text"]
    rogue = metric.compute(predictions=predictions, references=references, use_stemmer=True)
    print(f"rogue = {rogue}")


def self_eval(test_data: Dataset):
    predictions = test_data["output_text"]
    references = test_data["output_text"]

    metric = evaluate.load("rouge")
    rogue = metric.compute(predictions=predictions, references=references, use_stemmer=True)
    print(f"rogue = {rogue}")


def main():

    dataset = get_dataset()
    test_data = dataset["test"]
    test_data = test_data.select(list(range(100)))

    print("self eval")
    self_eval(test_data=test_data)

    # print("codellama eval")
    # codellama_eval(test_data=test_data)

    print("flan eval")

    # flan_eval(test_data=test_data, tokenizer_name='google/flan-t5-small', model_name='google/flan-t5-small')
    # flan_eval(test_data=test_data, tokenizer_name='google/flan-t5-small', model_name='result-flan-t5-small/')

    # flan_eval(test_data=test_data, tokenizer_name='google/flan-t5-base', model_name='google/flan-t5-base')
    # flan_eval(test_data=test_data, tokenizer_name='google/flan-t5-base', model_name='result-flan-t5-base')

    flan_eval(test_data=test_data, tokenizer_name="google/flan-t5-large", model_name="google/flan-t5-large")
    flan_eval(test_data=test_data, tokenizer_name="google/flan-t5-large", model_name="result-flan-t5-large/")

    # flan_eval(test_data=test_data, tokenizer_name='google/flan-t5-small', model_name='google/flan-t5-small')


if __name__ == "__main__":
    main()
