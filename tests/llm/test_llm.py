from typing import Callable

import numpy as np
import pandas as pd
import pytest

from dqc import LLMCurate
from dqc.llm_utils import compute_selfensembling_confidence_score


@pytest.mark.parametrize("model", [None, "random_str", 1])
@pytest.mark.parametrize("tokenizer", [None, "random_str", 1])
def test_init_failure(
    data,
    ds_column_mapping,
    prompt_variants,
    model,
    tokenizer,
    ref_col_set,
    random_state=43,
    batch_size=1,
    max_new_tokens=1,
    verbose=False,
):
    with pytest.raises(ValueError):
        llmc = LLMCurate(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            verbose=verbose,
            random_state=random_state,
        )
        ds = llmc.run(
            data=data,
            column_to_curate="Equation",
            ds_column_mapping=ds_column_mapping["valid"],
            prompt_variants=prompt_variants["valid"],
            llm_response_cleaned_column_list=list(ref_col_set),
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )


def test_init_success(
    data,
    ds_column_mapping,
    prompt_variants,
    model_and_tokenizer,
    ref_col_set,
    random_state=43,
    batch_size=1,
    max_new_tokens=1,
    verbose=False,
):
    model, tokenizer = model_and_tokenizer
    llmc = LLMCurate(
        model=model, tokenizer=tokenizer, random_state=random_state, verbose=verbose
    )
    ds = llmc.run(
        data=data,
        column_to_curate="Equation",
        ds_column_mapping=ds_column_mapping["valid"],
        prompt_variants=prompt_variants["valid"],
        llm_response_cleaned_column_list=list(ref_col_set),
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    ds_col_list = ds.column_names
    assert len(set(ds_col_list).intersection(ref_col_set)) == 3
    assert "confidence_score" in ds_col_list


@pytest.mark.parametrize("random_state", [None, "random_str"])
def test_random_state_failure(
    model_and_tokenizer,
    random_state,
    batch_size=1,
    max_new_tokens=1,
    verbose=False,
):
    model, tokenizer = model_and_tokenizer
    with pytest.raises((TypeError, ValueError)):
        llmc = LLMCurate(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            verbose=verbose,
            random_state=random_state,
        )


@pytest.mark.parametrize("column_to_curate", [None, 1])
def test_column_to_curate_failure(
    data,
    column_to_curate,
    ds_column_mapping,
    prompt_variants,
    model_and_tokenizer,
    ref_col_set,
    batch_size=1,
    max_new_tokens=1,
    verbose=False,
):
    model, tokenizer = model_and_tokenizer
    with pytest.raises(ValueError):
        llmc = LLMCurate(model=model, tokenizer=tokenizer, verbose=verbose)
        ds = llmc.run(
            data=data,
            column_to_curate=column_to_curate,
            ds_column_mapping=ds_column_mapping["valid"],
            prompt_variants=prompt_variants["valid"],
            llm_response_cleaned_column_list=list(ref_col_set),
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )


@pytest.mark.parametrize("colmap_variant", ["invalid_entity", "invalid_column"])
def test_dscolmap_failure(
    data,
    colmap_variant,
    ds_column_mapping,
    prompt_variants,
    model_and_tokenizer,
    ref_col_set,
    batch_size=1,
    max_new_tokens=1,
    verbose=False,
):
    model, tokenizer = model_and_tokenizer
    llmc = LLMCurate(model=model, tokenizer=tokenizer, verbose=verbose)
    with pytest.raises(ValueError):
        ds = llmc.run(
            data=data,
            column_to_curate="Equation",
            ds_column_mapping=ds_column_mapping[colmap_variant],
            prompt_variants=prompt_variants["valid"],
            llm_response_cleaned_column_list=list(ref_col_set),
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )

        ds_col_list = ds.column_names
        assert len(set(ds_col_list).intersection(ref_col_set)) == 3
        assert "confidence_score" in ds_col_list


@pytest.mark.parametrize("p_variant", ["invalid_count", "invalid_prompt"])
def test_prompt_failure(
    data,
    p_variant,
    ds_column_mapping,
    prompt_variants,
    model_and_tokenizer,
    ref_col_set,
    batch_size=1,
    max_new_tokens=1,
    verbose=False,
):
    model, tokenizer = model_and_tokenizer
    llmc = LLMCurate(model=model, tokenizer=tokenizer, verbose=verbose)
    with pytest.raises(ValueError):
        ds = llmc.run(
            data=data,
            column_to_curate="Equation",
            ds_column_mapping=ds_column_mapping["valid"],
            prompt_variants=prompt_variants[p_variant],
            llm_response_cleaned_column_list=list(ref_col_set),
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )

        ds_col_list = ds.column_names
        assert len(set(ds_col_list).intersection(ref_col_set)) == 3
        assert "confidence_score" in ds_col_list


@pytest.mark.parametrize("case_sensitive", [True, False])
def test_skip_llm_inference_success(
    data,
    ds_column_mapping,
    prompt_variants,
    model_and_tokenizer,
    ref_col_set,
    case_sensitive,
    batch_size=1,
    max_new_tokens=1,
    verbose=False,
):
    model, tokenizer = model_and_tokenizer
    llmc = LLMCurate(model=model, tokenizer=tokenizer, verbose=verbose)
    llmc.run(
        data=data,
        column_to_curate="Equation",
        ds_column_mapping=ds_column_mapping["valid"],
        prompt_variants=prompt_variants["valid"],
        llm_response_cleaned_column_list=list(ref_col_set),
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        skip_llm_inference=False,
    )
    ds = llmc.run(
        column_to_curate="Equation",
        llm_response_cleaned_column_list=list(ref_col_set),
        skip_llm_inference=True,
        scoring_params={"case_sensitive": case_sensitive},
    )

    ds_col_list = ds.column_names
    assert len(set(ds_col_list).intersection(ref_col_set)) == 3
    assert "confidence_score" in ds_col_list


def test_noscores_success(
    data,
    ds_column_mapping,
    prompt_variants,
    model_and_tokenizer,
    ref_col_set,
    batch_size=1,
    max_new_tokens=1,
    verbose=False,
):
    model, tokenizer = model_and_tokenizer
    llmc = LLMCurate(model=model, tokenizer=tokenizer, verbose=verbose)
    ds = llmc.run(
        data=data,
        column_to_curate="Equation",
        ds_column_mapping=ds_column_mapping["valid"],
        prompt_variants=prompt_variants["valid"],
        llm_response_cleaned_column_list=list(ref_col_set),
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        return_scores=False,
    )

    ds_col_list = ds.column_names
    assert len(set(ds_col_list).intersection(ref_col_set)) == 3
    assert "confidence_score" not in ds_col_list


@pytest.mark.parametrize("answer_start_token", [None, "[EQUATION]"])
@pytest.mark.parametrize("answer_end_token", [None, "[/PROPOSED_ANSWER]"])
def test_answertoken_success(
    data,
    ds_column_mapping,
    prompt_variants,
    answer_start_token,
    answer_end_token,
    model_and_tokenizer,
    ref_col_set,
    batch_size=1,
    max_new_tokens=1,
    verbose=False,
):
    model, tokenizer = model_and_tokenizer
    llmc = LLMCurate(model=model, tokenizer=tokenizer, verbose=verbose)
    ds = llmc.run(
        data=data,
        column_to_curate="Equation",
        ds_column_mapping=ds_column_mapping["valid"],
        prompt_variants=prompt_variants["valid"],
        llm_response_cleaned_column_list=list(ref_col_set),
        answer_start_token=answer_start_token,
        answer_end_token=answer_end_token,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        return_scores=False,
    )

    ds_col_list = ds.column_names
    assert len(set(ds_col_list).intersection(ref_col_set)) == 3
    assert "confidence_score" not in ds_col_list


def test_llmc_run_success(
    data,
    ds_column_mapping,
    prompt_variants,
    model_and_tokenizer,
    ref_col_set,
    batch_size=1,
    max_new_tokens=1,
    verbose=False,
):
    model, tokenizer = model_and_tokenizer
    llmc = LLMCurate(model=model, tokenizer=tokenizer, verbose=verbose)
    ds = llmc.run(
        data=data,
        column_to_curate="Equation",
        ds_column_mapping=ds_column_mapping["valid"],
        prompt_variants=prompt_variants["valid"],
        llm_response_cleaned_column_list=list(ref_col_set),
        answer_start_token="[EQUATION]",
        answer_end_token="[/PROPOSED_ANSWER]",
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    ds_col_list = ds.column_names
    assert len(set(ds_col_list).intersection(ref_col_set)) == 3
    assert "confidence_score" in ds_col_list


@pytest.mark.parametrize("verbose", [True, False])
def test_verbosity(
    data,
    ds_column_mapping,
    prompt_variants,
    model_and_tokenizer,
    ref_col_set,
    verbose,
    batch_size=1,
    max_new_tokens=1,
):
    model, tokenizer = model_and_tokenizer
    llmc = LLMCurate(model=model, tokenizer=tokenizer)
    ds = llmc.run(
        data=data,
        column_to_curate="Equation",
        ds_column_mapping=ds_column_mapping["valid"],
        prompt_variants=prompt_variants["valid"],
        llm_response_cleaned_column_list=list(ref_col_set),
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    ds_col_list = ds.column_names
    assert len(set(ds_col_list).intersection(ref_col_set)) == 3
    assert "confidence_score" in ds_col_list


def test_import_star():
    try:
        exec("from dqc import *")
        exec("from dqc.llm_utils import *")
    except Exception as e:
        pytest.fail(f"Importing with * raised an error: {e}")


def exact_match(text1: str, text2: str) -> float:
    return 1.0 if text1 == text2 else 0.0


@pytest.fixture
def scoring_method() -> Callable[[str, str], float]:
    return exact_match


def test_exact_match_case_insensitive(confidence_dataset_row, scoring_method):
    res = compute_selfensembling_confidence_score(
        example=confidence_dataset_row,
        target_column="target_text",
        reference_column_list=["reference_1", "reference_2", "reference_3"],
        scoring_method=scoring_method,
    )

    assert res["confidence_score"] == pytest.approx(2 / 3)


def test_exact_match_case_sensitive(confidence_dataset_row, scoring_method):
    res = compute_selfensembling_confidence_score(
        example=confidence_dataset_row,
        target_column="target_text",
        reference_column_list=["reference_1", "reference_2", "reference_3"],
        scoring_method=scoring_method,
        case_sensitive=True,
    )

    assert res["confidence_score"] == pytest.approx(1 / 3)


def test_invalid_scoring_method(confidence_dataset_row):
    with pytest.raises(ValueError):
        compute_selfensembling_confidence_score(
            example=confidence_dataset_row,
            target_column="target_text",
            reference_column_list=["reference_1", "reference_2", "reference_3"],
            scoring_method="invalid_method",
        )
