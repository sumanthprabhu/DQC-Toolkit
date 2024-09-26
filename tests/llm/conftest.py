import gc
from typing import Tuple, Union

import pandas as pd
import pytest
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


@pytest.fixture(scope="session")
def data():
    dataset = "ChilleD/SVAMP"
    dset = load_dataset(dataset, trust_remote_code=True)
    data = pd.DataFrame(dset["test"])[["question_concat", "Equation"]]
    return data.loc[:10]


@pytest.fixture(scope="session")
def ds_column_mapping():
    return {
        "valid": {"QUESTION": "question_concat", "EQUATION": "Equation"},
        "invalid_entity": {"": "question_concat", None: "Equation"},
        "invalid_column": {"QUESTION": None, "EQUATION": "nonexistent"},
    }


@pytest.fixture(scope="session")
def prompt_variants():
    return {
        "valid": [
            "You are a helpful assistant. ",
            "You are an honest assistant",
            "Please respond with the correct answer to the following question",
        ],
        "invalid_count": ["", ""],
        "invalid_prompt": ["", None, ""],
    }


@pytest.fixture(scope="session")
def confidence_dataset_row():
    return {
        "target_text": "sample sentence.",
        "reference_1": "sample sentence.",
        "reference_2": "SAMPLE sentence.",
        "reference_3": "different sentence.",
    }


@pytest.fixture(scope="session")
def model_and_tokenizer():
    return initialize_model_and_tokenizer("distilbert/distilgpt2")


def initialize_model_and_tokenizer(
    model_name: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Util function to construct the model and tokenizer objects"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return model, tokenizer


@pytest.fixture(scope="session")
def ref_col_set():
    return set(
        ["reference_prediction1", "reference_prediction2", "reference_prediction3"]
    )
