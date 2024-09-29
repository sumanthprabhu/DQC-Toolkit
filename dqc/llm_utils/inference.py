import gc
import random
from typing import Union

import datasets
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)


def _set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _generate_predictions(
    example: datasets.formatting.formatting.LazyBatch,
    generator: pipeline,
    llm_prompt_col_name: str,
    llm_response_raw_col_name: str = "llm_response",
    **options,
) -> dict:
    """
    Generates predictions using the text generation model for a given example.

    Args:
        example (datasets.formatting.formatting.LazyBatch): Batch of samples from a dataset.
        generator (pipeline): Huggingface pipeline for text generation.
        llm_prompt_col_name (str): Prompt for the text generation model.
        llm_response_raw_col_name (str, optional): Name of the column containing prediction. Defaults to 'llm_response'.

    Returns:
        dict: A dictionary containing the generated predictions.
    """
    predictions = []
    batch_results = generator(
        example[llm_prompt_col_name], early_stopping=True, **options
    )
    res_dict = {}

    predictions = [result[0]["generated_text"] for result in batch_results]
    res_dict[llm_response_raw_col_name] = predictions

    return res_dict


def build_LLM_prompt(
    input_ds: Dataset,
    ds_column_mapping: dict,
    prompt_template_prefix: str = "",
    answer_start_token: str = "",
    llm_prompt_col_name: str = "llm_prompt",
) -> Dataset:
    """Util function to build the LLM prompt from input text data

    Args:
        input_ds (Dataset): Input dataset containing text
        ds_column_mapping (dict): Dictionary mapping prompt entities to dataset column names.
        prompt_template_prefix (Union[str, None], optional): Text instruction to prepend to each transformed input text sample. Defaults to "".
        answer_start_token (str, optional): Token to append to the prompt to indicate start of the answer. Defaults to ""
        llm_prompt_col_name (str, optional): Name of the column for the built LLM prompts. Defaults to 'llm_prompt'
    Returns:
        Dataset: Dataset with generated predictions.
    """
    if type(input_ds) == pd.DataFrame:
        input_ds = Dataset.from_pandas(input_ds)

    def _helper(
        example: datasets.formatting.formatting.LazyBatch,
        prompt_template_prefix: str,
        ds_column_mapping: dict,
        llm_prompt_col_name: str,
    ) -> dict:
        llm_prompt = prompt_template_prefix
        for entity_name, col_name in ds_column_mapping.items():
            if col_name:
                entity_value = example[col_name]
                if type(entity_value) == list:
                    entity_value = "|| ".join(map(str, entity_value))
                else:
                    entity_value = str(entity_value)
                llm_prompt += f"[{entity_name}]{entity_value}[/{entity_name}]"

        if answer_start_token:
            llm_prompt += answer_start_token

        return {llm_prompt_col_name: llm_prompt}

    input_ds = input_ds.map(
        _helper,
        fn_kwargs={
            "prompt_template_prefix": prompt_template_prefix,
            "ds_column_mapping": ds_column_mapping,
            "llm_prompt_col_name": llm_prompt_col_name,
        },
    )
    return input_ds


def infer_LLM(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ds: Dataset,
    llm_prompt_col_name: str = "llm_prompt",
    llm_response_raw_col_name: str = "llm_response",
    **options,
) -> Dataset:
    """
    Util function to run LLM inference

    Args:
        model (AutoModelForCausalLM): LLM artifact.
        tokenizer (Autotokenizer) : LLM tokenizer object
        input_ds (Dataset): Input dataset containing text prompts.
        llm_prompt_col_name (str, optional): Name of the column containing text prompts. Defaults to 'llm_prompt'.
        llm_response_raw_col_name (str, optional): Name of the column containing prediction. Defaults to 'llm_response'.

    Returns:
        dataset: Dataset with generated predictions.
    """
    if options["random_state"]:
        _set_seed(options["random_state"])
        del options["random_state"]

    text_generator = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, truncation=False, **options
    )
    text_generator.tokenizer.pad_token_id = model.config.eos_token_id

    batch_size = options["batch_size"] if "batch_size" in options else 8

    input_ds = input_ds.map(
        _generate_predictions,
        fn_kwargs={
            "generator": text_generator,
            "llm_prompt_col_name": llm_prompt_col_name,
            "llm_response_raw_col_name": llm_response_raw_col_name,
            **options,
        },
        batched=True,
        batch_size=batch_size,
    )

    return input_ds


def _postprocess(
    sample: datasets.formatting.formatting.LazyRow,
    llm_prompt_col_name: str = "llm_prompt",
    llm_response_raw_col_name: str = "llm_response",
    llm_response_cleaned_col_name: str = "llm_response_cleaned",
    answer_end_token: str = "",
) -> dict:
    """Util function to extract the generated answer from the generated LLM prediction

    Args:
        sample (datasets.formatting.formatting.LazyRow): Batch of samples from a dataset
        llm_prompt_col_name (str, optional): Name of the column containing the LLM prompts. Defaults to 'llm_prompt'
        llm_response_raw_col_name (str, optional): Name of the column containing prediction. Defaults to 'llm_response'.
        llm_response_cleaned_col_name (str, optional): Name of the column for the final processed result. Defaults to 'llm_response_cleaned'
        answer_end_token (str, optional): Token to use to separate noise from expected output

    Returns:
        dict: Dictionary of extracted answer sequences
    """
    prompt_length = len(sample[llm_prompt_col_name])

    extracted_answer = sample[llm_response_raw_col_name][prompt_length:].strip("\n")

    if answer_end_token:
        extracted_answer = extracted_answer.split(answer_end_token)[0]

    return {llm_response_cleaned_col_name: extracted_answer}


def run_LLM(
    val_data: Union[pd.DataFrame, Dataset],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    ds_column_mapping: dict,
    prompt_template_prefix: Union[str, None] = "",
    llm_prompt_col_name: str = "llm_prompt",
    llm_response_raw_col_name: str = "llm_response",
    llm_response_cleaned_col_name: str = "llm_response_cleaned",
    answer_start_token: str = "",
    answer_end_token: str = "",
    **options,
) -> dict:
    """Run end-to-end LLM inference (from pre-processing input data to post-processing the predictions) and return the computed performance metrics on input validation data

    Args:
        val_data (Union[pd.DataFrame, Dataset]): Validation data with labels
        model (AutoModelForCausalLM): LLM artifact.
        tokenizer (Autotokenizer) : LLM tokenizer object
        ds_column_mapping (dict): Dictionary mapping prompt entities to dataset column names.
        prompt_template_prefix (Union[str, None], optional): Text instruction to prepend to each transformed input text sample. Defaults to "".
        llm_prompt_col_name (str, optional): Name of the column with the built LLM prompts. Defaults to 'llm_prompt'
        llm_response_raw_col_name (str, optional): Name of the column containing prediction. Defaults to 'llm_response'.
        llm_response_cleaned_col_name (str, optional): Name of the column containing the final post processed result. Defaults to 'llm_response_cleaned'
        answer_start_token (str, optional): Token that indicates the start of answer generation. Defaults to ''
        answer_end_token (str, optional): Token that indicates the end of answer generation. Defaults to ''

    Returns:
        dict: A dictionary containing F1 score.
    """
    predicted_label_list = []

    val_ds = build_LLM_prompt(
        val_data,
        ds_column_mapping=ds_column_mapping,
        prompt_template_prefix=prompt_template_prefix,
        answer_start_token=answer_start_token,
        llm_prompt_col_name=llm_prompt_col_name,
    )

    val_ds_with_pred = infer_LLM(
        model,
        tokenizer,
        val_ds,
        llm_prompt_col_name=llm_prompt_col_name,
        llm_response_raw_col_name=llm_response_raw_col_name,
        **options,
    )

    val_ds_with_pred = val_ds_with_pred.map(
        _postprocess,
        fn_kwargs={
            "llm_prompt_col_name": llm_prompt_col_name,
            "llm_response_raw_col_name": llm_response_raw_col_name,
            "llm_response_cleaned_col_name": llm_response_cleaned_col_name,
            "answer_end_token": answer_end_token,
        },
    )

    return val_ds_with_pred
