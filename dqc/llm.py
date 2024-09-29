from typing import Callable, List, Tuple, Union

import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from dqc.base import BaseCurate
from dqc.llm_utils import (
    _empty_ds_ensemble_handler,
    _validate_init_params,
    _validate_run_params,
    compute_selfensembling_confidence_score,
    run_LLM,
)
from dqc.utils import Logger

logger = Logger("DQC-Toolkit")


class LLMCurate(BaseCurate):
    """
    Args:
        model (AutoModelForCausalLM): Instantiated LLM
        tokenizer (AutoTokenizer): Instantiated tokenizer corresponding to the `model`
        verbose (bool, optional): Sets the verbosity level during execution. `True` indicates logging level INFO and `False` indicates logging level 'WARNING'. Defaults to False.

    Examples:
     ```python

    llmc = LLMCurate(model, tokenizer)
    ds = llmc.run(
            data,
            column_to_curate,
            ds_column_mapping,
            prompt_variants,
            llm_response_cleaned_column_list,
            answer_start_token,
            answer_end_token,
            batch_size,
            max_new_tokens
            )
    ```
    where
    * `model` and `tokenizer` are the instantiated LLM model and tokenizer objects respectively
    * `data` is a pandas dataframe containing samples with our target text for curation under column `column_to_curate`
    * `ds_column_mapping` is the dictionary mapping of entities used in the LLM prompt and the corresponding columns in `data`. For example, `ds_column_mapping={'INPUT' : 'input_column'}` would imply that text under `input_column` in `data` would be passed to the LLM in the format `"[INPUT]row['input_column'][/INPUT]"` for each `row` in `data`
    * `prompt_variants` is the list of LLM prompts to be used to curate `column_to_curate` and `llm_response_cleaned_column_list` is the corresponding list of column names to store the reference responses generated using each prompt
    * `answer_start_token` and `answer_end_token` are optional  text phrases representing the start and end of the answer respectively.

    `ds` is a dataset object with the following additional features -
    1. Feature for each column name in `llm_response_cleaned_column_list`
    2. LLM Confidence score for each text in `column_to_curate`

    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        verbose: bool = False,
        **options,
    ):
        super().__init__(**options)

        _validate_init_params(model, tokenizer, verbose)
        self.model = model
        self.tokenizer = tokenizer
        self.verbose = verbose
        self._set_verbosity(verbose)

        self.ds_ensemble = None

    def __str__(self):
        display_dict = self.__dict__.copy()

        for key in list(display_dict.keys()):
            if key in ["ds_ensemble"]:
                ## Don't need to display these attributes
                del display_dict[key]

        return str(display_dict)

    __repr__ = __str__

    def _set_verbosity(self, verbose: bool):
        """Set logger level based on user input for parameter `verbose`

        Args:
            verbose (bool): Indicator for verbosity
        """
        if verbose:
            logger.set_level("INFO")
        else:
            logger.set_level("WARNING")

    def fit_transform(self):
        pass

    def run(
        self,
        column_to_curate: str,
        data: Union[pd.DataFrame, Dataset] = None,
        ds_column_mapping: dict = {},
        prompt_variants: List[str] = [""],
        skip_llm_inference: bool = False,
        llm_response_cleaned_column_list: List[str] = ["reference_prediction"],
        return_scores: bool = True,
        answer_start_token: str = "",
        answer_end_token: str = "",
        scoring_params: dict = {
            "scoring_method": "exact_match",
            "case_sensitive": False,
        },
        **options,
    ) -> Dataset:
        """Run LLMCurate on the input data

        Args:
            column_to_curate (str): Column name in `data` with the text that needs to be curated
            data (Union[pd.DataFrame, Dataset]): Input data for LLM based curation
            ds_column_mapping (dict, optional): Mapping of entities to be used in the LLM prompt and the corresponding columns in the input data. Defaults to {}.
            prompt_variants (List[str], optional): List of different LLM prompts to be used to curate the labels under `column_to_curate`. Defaults to [''].
            skip_llm_inference (bool, optional): Indicator variable to prevent re-running LLM inference. Set to `True` if artifacts from the previous run of LLMCurate needs to be reused. Else `False`. Defaults to False.
            llm_response_cleaned_column_list (list, optional): Names of the columns that will contain LLM predictions for each input prompt in `prompt_variants`. Defaults to ['reference_prediction'].
            return_scores (bool, optional): Indicator variable set to `True` if label confidence scores are to be computed for each label under `column_to_curate`. Defaults to True.
            answer_start_token (str, optional): Token that indicates the start of answer generation. Defaults to ''
            answer_end_token (str, optional): Token that indicates the end of answer generation. Defaults to ''
            scoring_params (dict, optional): Parameters related to util function `compute_selfensembling_confidence_score` to compute confidence scores of `column_to_curate`

        Returns:
            Dataset: Input dataset with reference responses. If `return_scores=True`, then input dataset with reference responses and confidence scores.
        """
        if not skip_llm_inference:
            empty_string_col_list = _validate_run_params(
                data,
                column_to_curate,
                ds_column_mapping,
                prompt_variants,
                llm_response_cleaned_column_list,
            )

            if len(empty_string_col_list) > 0:
                logger.warning(
                    "Found empty string(s) in the input data under column(s) {empty_string_col_list}"
                )

            logger.info(
                f"Running the LLM to generate the {len(prompt_variants)} reference responses using `prompt_variants`.."
            )
            ds_ensemble = None

            model = self.model
            tokenizer = self.tokenizer

            for index, prompt_template_prefix in enumerate(prompt_variants):
                proposed_answer_col_name = llm_response_cleaned_column_list[index]
                ds = run_LLM(
                    data,
                    model,
                    tokenizer,
                    ds_column_mapping=ds_column_mapping,
                    prompt_template_prefix=prompt_template_prefix,
                    answer_start_token=answer_start_token,
                    answer_end_token=answer_end_token,
                    llm_response_cleaned_col_name=proposed_answer_col_name,
                    random_state=self.random_state,
                    **options,
                )

                if not ds_ensemble:
                    ds_ensemble = ds
                else:
                    ds_ensemble = ds_ensemble.add_column(
                        proposed_answer_col_name, ds[proposed_answer_col_name]
                    )
            self.ds_ensemble = ds_ensemble

        if return_scores:
            if skip_llm_inference:
                if (
                    isinstance(data, pd.DataFrame)
                    or ds_column_mapping
                    or prompt_variants
                ):
                    logger.warning(
                        "Ignoring params `data`, `ds_column_mapping` and `prompt_variants` since `skip_llm_inference` is set to `True`"
                    )

            _empty_ds_ensemble_handler(len(self.ds_ensemble) == 0, skip_llm_inference)

            logger.info(
                "Computing confidence scores using the LLM reference responses.."
            )
            self.ds_ensemble = self.ds_ensemble.map(
                compute_selfensembling_confidence_score,
                fn_kwargs={
                    "target_column": column_to_curate,
                    "reference_column_list": llm_response_cleaned_column_list,
                    **scoring_params,
                },
            )

        return self.ds_ensemble
