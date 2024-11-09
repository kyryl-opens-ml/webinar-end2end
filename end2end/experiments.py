import json
import logging
from dataclasses import dataclass
from pathlib import Path

import evaluate
import torch
from datasets import Dataset
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
from random import randrange

from datasets import DatasetDict, load_dataset

import logging
from functools import partial
from pathlib import Path

import torch
from datasets import DatasetDict
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer


import logging
import sys
from pathlib import Path

import datasets
import transformers
import wandb

import typer



logger = logging.getLogger(__name__)



@dataclass
class DataTrainingArguments:
    train_file: str
    test_file: str


@dataclass
class ModelArguments:
    model_id: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float




def _get_sql_data(random_state: int = 42, subsample: float = None) -> DatasetDict:
    dataset_name = "b-mc2/sql-create-context"
    dataset = load_dataset(dataset_name, split="train")
    print(f"dataset size: {len(dataset)}")
    print(dataset[randrange(len(dataset))])

    if subsample is not None:
        dataset = dataset.shuffle(seed=random_state).select(
            range(int(len(dataset) * subsample))
        )
        print(f"dataset new size: {len(dataset)}")

    datasets = dataset.train_test_split(test_size=0.05, seed=random_state)
    return datasets


def load_sql_data(path_to_save: Path | str, subsample: float = None):
    path_to_save = Path(path_to_save)
    path_to_save.mkdir(parents=True, exist_ok=True)

    datasets = _get_sql_data(subsample=subsample)

    datasets["train"].to_json(path_to_save / "train.json")
    datasets["test"].to_json(path_to_save / "test.json")



class Predictor:
    def __init__(self, model_load_path: str):
        device_map = {"": 0}
        new_model = AutoPeftModelForCausalLM.from_pretrained(
            model_load_path,
            # low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device_map,
        )
        merged_model = new_model.merge_and_unload()

        tokenizer = AutoTokenizer.from_pretrained(
            model_load_path, trust_remote_code=True
        )
        pipe = pipeline("text-generation", model=merged_model, tokenizer=tokenizer)
        self.pipe = pipe

    @torch.no_grad()
    def predict(self, question: str, context: str) -> str:
        pipe = self.pipe

        messages = [{"content": f"{context}\n Input: {question}", "role": "user"}]

        prompt = pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = pipe(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            num_beams=1,
            temperature=0.3,
            top_k=50,
            top_p=0.95,
            max_time=180,
        )
        sql = outputs[0]["generated_text"][len(prompt) :].strip()
        return sql


def run_evaluate_on_json(json_path: Path, model_load_path: Path, result_path: Path):
    df = Dataset.from_json(str(json_path)).to_pandas()
    model = Predictor(model_load_path=model_load_path)

    generated_sql = []
    for idx in tqdm(range(len(df))):
        context = df.iloc[idx]["context"]
        question = df.iloc[idx]["question"]

        sql = model.predict(question=question, context=context)
        generated_sql.append(sql)

    gt_sql = df["answer"].values
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=generated_sql, references=gt_sql)
    print(f"Metrics {results}")
    with open(result_path, "w") as f:
        json.dump(results, f)


def create_message_column(row):
    messages = []
    user = {"content": f"{row['context']}\n Input: {row['question']}", "role": "user"}
    messages.append(user)
    assistant = {"content": f"{row['answer']}", "role": "assistant"}
    messages.append(assistant)
    return {"messages": messages}


def format_dataset_chatml(row, tokenizer):
    return {
        "text": tokenizer.apply_chat_template(
            row["messages"], add_generation_prompt=False, tokenize=False
        )
    }


def process_dataset(model_id: str, train_file: str, test_file: str) -> DatasetDict:
    dataset = DatasetDict(
        {
            "train": Dataset.from_json(train_file),
            "test": Dataset.from_json(test_file),
        }
    )

    tokenizer_id = model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.padding_side = "right"

    dataset_chatml = dataset.map(create_message_column)
    dataset_chatml = dataset_chatml.map(
        partial(format_dataset_chatml, tokenizer=tokenizer)
    )
    return dataset_chatml


def get_model(model_id: str, device_map):
    # if torch.cuda.is_bf16_supported():
    #     compute_dtype = torch.bfloat16
    #     attn_implementation = "flash_attention_2"
    #     # If bfloat16 is not supported, 'compute_dtype' is set to 'torch.float16' and 'attn_implementation' is set to 'sdpa'.
    # else:
    #     compute_dtype = torch.float16
    #     attn_implementation = "sdpa"

    #     # This line of code is used to print the value of 'attn_implementation', which indicates the chosen attention implementation.
    #     print(attn_implementation)

    compute_dtype = torch.bfloat16
    attn_implementation = "flash_attention_2"
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, add_eos_token=True, use_fast=True
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    return tokenizer, model


def get_config(config_dict: dict):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_dict(config_dict)
    return model_args, data_args, training_args


def train(config_dict: Path):
    setup_logger(logger)

    model_args, data_args, training_args = get_config(config_dict=config_dict)

    logger.info(f"model_args = {model_args}")
    logger.info(f"data_args = {data_args}")
    logger.info(f"training_args = {training_args}")

    device_map = {"": 0}
    target_modules = [
        "k_proj",
        "q_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj",
    ]

    set_seed(training_args.seed)

    dataset_chatml = process_dataset(
        model_id=model_args.model_id,
        train_file=data_args.train_file,
        test_file=data_args.test_file,
    )
    logger.info(dataset_chatml["train"][0])

    tokenizer, model = get_model(model_id=model_args.model_id, device_map=device_map)
    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_chatml["train"],
        eval_dataset=dataset_chatml["test"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()
    trainer.save_model()
    trainer.create_model_card()




def setup_logger(logger):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = "INFO"
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def upload_to_registry(model_name: str, model_path: Path):
    with wandb.init() as _:
        art = wandb.Artifact(model_name, type="model")
        art.add_file(model_path / "README.md")
        art.add_file(model_path / "adapter_config.json")
        art.add_file(model_path / "adapter_model.safetensors")
        art.add_file(model_path / "special_tokens_map.json")
        art.add_file(model_path / "tokenizer.json")
        art.add_file(model_path / "tokenizer_config.json")
        art.add_file(model_path / "training_args.bin")
        wandb.log_artifact(art)


def load_from_registry(model_name: str, model_path: Path):
    with wandb.init() as run:
        artifact = run.use_artifact(model_name, type="model")
        artifact_dir = artifact.download(root=model_path)
        print(f"{artifact_dir}")


def end2end_training_eval():
    path_to_save = "./data/"
    load_sql_data(path_to_save=path_to_save)


    config = json.loads("""
            {
            "train_file": "./data/train.json",
            "test_file": "./data/test.json",

            "model_id": "microsoft/Phi-3-mini-4k-instruct",
            "lora_r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.05,


            "output_dir": "./phi-3-mini-lora-text2sql",
            "eval_strategy": "steps",
            "do_eval": true,
            "optim": "adamw_torch",
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "learning_rate": 0.0001,
            "num_train_epochs": 1,
            "warmup_ratio": 0.1,
            "logging_first_step": true,
            "logging_steps": 500,
            "save_steps": 500,
            "seed": 42,
            "bf16": true,
            "fp16": false,
            "eval_steps": 500,
            "report_to": ["none"],
            "lr_scheduler_type": "linear",
            "log_level" : "debug",
            "evaluation_strategy": "steps",
            "eval_on_start": true
            }
    """)
    train(config_dict=config)

    # run_evaluate_on_json(json_path=config['test_file'], 
    #                      model_load_path=config['output_dir'], 
    #                      result_path='metrics.json')

# app = typer.Typer()
# app.command()(load_sql_data)
# app.command()(train)
# app.command()(upload_to_registry)
# app.command()(load_from_registry)
# app.command()(run_evaluate_on_json)


if __name__ == "__main__":
    end2end_training_eval()
