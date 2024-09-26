from itertools import combinations
from typing import Callable, List, Union

import datasets
import numpy as np
from transformers import AutoModel, AutoTokenizer


def _compute_exact_match_score(
    target_text: str, reference_text_list: List[str]
) -> float:
    """Util function to quantify the number of exact matches between a target text and a list of text strings.

    Args:
        target_text (str): The text string that will be compared to each text in `reference_text_list`.
        reference_text_list (List[str]): List of text strings that need to be individually matched against `target_text`.

    Returns:
        float: Score between 0 and 1 indicating the percentage of texts in `reference_text_list` that exactly matched `target_text`
    """
    matches = [1 if text == target_text else 0 for text in reference_text_list]
    return sum(matches) / len(matches)


def _compute_custom_match_score(
    target_text: str,
    reference_text_list: List[str],
    scoring_method: Union[Callable, str],
) -> float:
    """Util function to compute the average similarity score between a target text and a list of text string based on a specified scoring function.

    Args:
        target_text (str): The text string that will be compared to each text in `reference_text_list`.
        reference_text_list (List[str]): List of text strings that need to be individually matched against `target_text`
        scoring_method (Union[Callable, str]): A function or the string 'exact_match' to compute the confidence score. Defaults to 'exact_match'.

    Returns:
        float: Score between 0 and 1 indicating how closely the texts in `reference_text_list` match the `target_text`. A score of 1 means perfect matches for all entries,
               while a score of 0 indicates no similarity.
    """
    matches = np.array(
        [scoring_method(text, target_text) for text in reference_text_list]
    )
    return matches.mean()


def compute_selfensembling_confidence_score(
    example: datasets.formatting.formatting.LazyRow,
    target_column: str,
    reference_column_list: List[str],
    scoring_method: Union[Callable[[str, str], float], str] = "exact_match",
    case_sensitive: bool = False,
    **options,
) -> float:
    """Util function to compute confidence score of a given target text using LLM generated reference texts.

    Args:
        example (datasets.formatting.formatting.LazyRow): A row of data from a dataset containing the target and reference texts.
        target_column (str): Name of the column containing the target text for estimation of confidence score.
        reference_column_list (List[str]): Names of the columns containing the reference texts to be compared with the target text.
        scoring_method (Union[Callable[[str, str], float], str], optional): A function or the string 'exact_match' to compute the confidence score. Defaults to 'exact_match'.
        case_sensitive (bool, optional): `True` if string comparisons need to be case aware. Else `False`. Defaults to `False`
    Raises:
        ValueError: If `scoring_method` is neither 'exact_match' nor a valid callable function

    Returns:
        float: Score between 0 and 1 quantifying the confidence score for the target text
    """
    if not callable(scoring_method) and scoring_method != "exact_match":
        raise ValueError(
            "Parameter `scoring_method` must be 'exact_match' or a valid callable that measures string similarity"
        )

    reference_text_list = []
    result_dict = {}
    score = 0

    for col in reference_column_list:
        reference_text_list.append(example[col])

    target_text = example[target_column]

    if not case_sensitive:
        target_text = target_text.lower()
        reference_text_list = [text.lower() for text in reference_text_list]

    if scoring_method == "exact_match":
        score = _compute_exact_match_score(target_text, reference_text_list)
    else:
        score = _compute_custom_match_score(
            target_text, reference_text_list, scoring_method
        )

    return {"confidence_score": score}
