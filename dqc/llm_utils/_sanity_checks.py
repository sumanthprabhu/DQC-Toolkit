from typing import List, Tuple, Union

import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def _validate_ds_column_mapping(data: pd.DataFrame, ds_column_mapping: dict):
    """Sanity checks for `ds_column_mapping`

    Args:
        data (Union[pd.DataFrame, Dataset]): Input data for LLM based curation
        ds_column_mapping (dict): Mapping of entities to be used in the LLM prompt and the corresponding columns in the input data.

    Raises:
        ValueError: If any of the entities or column names in `ds_column_mapping` are invalid.
    """
    valid_column_names = data.columns

    if not ds_column_mapping:
        raise ValueError(
            f"`ds_column_mapping` cannot be empty when `skip_llm_inference=False`. Please pass a valid non empty dictionary."
        )

    for entity, col_name in ds_column_mapping.items():
        if not entity or entity == "":
            raise ValueError(
                f"Entity `{entity}` for column name {col_name} in `ds_column_mapping` is invalid. Please make sure to pass valid non empty texts only."
            )

        if not col_name or col_name == "" or col_name not in valid_column_names:
            raise ValueError(
                f"Column name `{col_name}` for entity `{entity}` in `ds_column_mapping` is invalid. Please make sure to pass valid column names from the input data."
            )

    return


def _check_null_values(
    data: pd.DataFrame, column_to_curate: str, ds_column_mapping: dict
) -> List[str]:
    """Sanity checks to detect null entries in the data

    Args:
        data (pd.DataFrame): Input data for LLM based curation
        column_to_curate (str): Column name in `data` with the text that needs to be curated
        ds_column_mapping (dict): Mapping of entities to be used in the LLM prompt and the corresponding columns in the input data

    Raises:
        ValueError: If null values are found in the data

    Returns:
        List[str]: List of column names containing blank values
    """
    if (
        column_to_curate is None
        or column_to_curate == ""
        or column_to_curate not in data.columns
    ):
        raise ValueError(
            f"`column_to_curate` should be a valid column name from the input data columns"
        )

    empty_string_col_list = []
    for _, column_name in ds_column_mapping.items():
        if len(data.loc[data[column_name].isnull(), column_name]) > 0:
            raise ValueError(
                f"Found `None` entries under column {column_name} in the input data."
            )

        count = (data[column_name] == "").sum()
        if count > 0:
            empty_string_col_list.append(column_name)

    return empty_string_col_list


def _check_num_rows(data: pd.DataFrame):
    """Check if the input data contains atleast one row

    Args:
        data (pd.DataFrame): Input data for LLM based curation

    Raises:
        ValueError: If data contains zero rows
    """
    if len(data) == 0:
        raise ValueError(
            "Input data must be non null when `skip_llm_inference` is set to `False`."
        )

    return


def _validate_prompts_and_response_col_names(
    prompt_variants: List[str], llm_response_cleaned_column_list: List[str]
):
    """Sanity checks to detect invalid entries in `prompt_variants` and `llm_response_cleaned_column_list`

    Args:
        prompt_variants (List[str]): List of different LLM prompts to be used to curate the labels under `column_to_curate`.
        llm_response_cleaned_column_list (list): Names of the columns that will contain LLM predictions for each input prompt in `prompt_variants`.

    Raises:
        ValueError: If either `prompt_variants` or `llm_response_cleaned_column_list` contain invalid entries or number of entries in both lists do not match
    """
    if None in prompt_variants:
        raise ValueError(
            f"Found {None} in `prompt_variants`. Please make sure to pass only non null prompts in `prompt_variants`"
        )

    if None in llm_response_cleaned_column_list:
        raise ValueError(
            f"Found {None} in `llm_response_cleaned_col_name_list`. Please make sure to pass only non empty strings in `llm_response_cleaned_col_name_list` "
        )

    if "" in llm_response_cleaned_column_list:
        raise ValueError(
            f"Found blank string in `llm_response_cleaned_col_name_list`. Please make sure to pass only non empty strings in `llm_response_cleaned_col_name_list`"
        )

    if len(prompt_variants) != len(llm_response_cleaned_column_list):
        raise ValueError(
            "Number of prompts in `prompt_list` should match number of response column names in `llm_response_cleaned_col_name_list`"
        )


def _validate_run_params(
    data: pd.DataFrame,
    column_to_curate: str,
    ds_column_mapping: dict,
    prompt_variants: List[str],
    llm_response_cleaned_column_list: List[str],
) -> List[str]:
    """Run collection of sanity checks on parameters passed to `LLMCurate.run()`

    Args:
        data (pd.DataFrame): Input data for LLM based curation
        column_to_curate (str): Column name in `data` with the text that needs to be curated
        prompt_variants (List[str]): List of different LLM prompts to be used to curate the labels under `column_to_curate`..
        ds_column_mapping (dict): Mapping of entities to be used in the LLM prompt and the corresponding columns in the input data
        llm_response_cleaned_column_list (list): Names of the columns that will contain LLM predictions for each input prompt in `prompt_variants`

    Returns:
        List[str]: List of column names containing blank values
    """
    if isinstance(data, Dataset):
        data = data.to_pandas()

    _check_num_rows(data)
    _validate_ds_column_mapping(data, ds_column_mapping)
    _validate_prompts_and_response_col_names(
        prompt_variants, llm_response_cleaned_column_list
    )

    return _check_null_values(data, column_to_curate, ds_column_mapping)


def is_valid_text_generation_pipeline(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer
):
    """Check if the given model and tokenizer are compatible with text generation.

    Args:
        model (AutoModelForCausalLM): Instantiated LLM
        tokenizer (AutoTokenizer): Instantiated tokenizer corresponding to the `model`

    Raises:
        ValueError: If `model` or `tokenizer` or both are invalid artifacts

    """
    if not hasattr(model, "generate"):
        raise ValueError(
            "Found invalid model: Missing 'generate' method for text generation."
        )

    if not all(hasattr(tokenizer, attr) for attr in ["encode", "decode"]):
        raise ValueError(
            "Found invalid tokenizer: Missing 'encode' or 'decode' methods."
        )

    return


def _validate_init_params(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    verbose: bool,
):
    """Sanity checks to verify the validity of params passed to initialize an instance of LLMCurate

    Args:
        model (AutoModelForCausalLM): Instantiated LLM
        tokenizer (AutoTokenizer): Instantiated tokenizer corresponding to the `model`
        verbose (bool, optional): Sets the verbosity level during execution. `True` indicates logging level INFO and `False` indicates logging level 'WARNING'.
    """

    expected_types = [
        (verbose, "verbose", bool),
    ]

    for obj, var_name, expected_type in expected_types:
        if not isinstance(obj, expected_type):
            raise ValueError(
                f"Expected `{var_name}` to be an instance of `{expected_type}`, but got `{type(obj).__name__}`"
            )

    is_valid_text_generation_pipeline(model, tokenizer)

    return


def _empty_ds_ensemble_handler(empty_ds_ensemble: bool, skip_llm_inference: bool):
    """Exception handling if `ds_ensemble` is `None` when reliability scores need to be computed

    Args:
        empty_ds_ensemble (bool): Indicator variable set to `True` if `ds_ensemble` is empty. Else `False`
        skip_llm_inference (bool): Indicator variable to prevent re-running LLM inference. Set to `True` if artifacts from the previous run of LLMCurate needs to be reused. Else `False`.

    Raises:
        ValueError: If `ds_ensemble` is None
    """
    if empty_ds_ensemble:
        error_message = "Found `self.ds_ensemble` to be `None`."

        if skip_llm_inference:
            error_message += "Try running with parameter `skip_llm_inference=False` to first generate reference responses using the input dataset."

        raise ValueError(error_message)

    return
