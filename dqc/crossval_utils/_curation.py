from typing import Generator, Tuple, Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


class SentenceTransformerVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self, model: Union[str, SentenceTransformer] = "BAAI/bge-small-en-v1.5"
    ):
        if isinstance(model, str):
            self.model = SentenceTransformer(model)
        else:
            self.model = model

    def fit(self, X, y=None):
        # SentenceTransformer does not require fitting, so we do nothing here
        return self

    def transform(self, X):
        # X is expected to be a list or array-like of strings (sentences)
        embeddings = self.model.encode(
            X, show_progress_bar=False, normalize_embeddings=True
        )
        # We return the embeddings as a dense numpy array
        return np.array(embeddings)


def _fetch_supported_implementations():
    """Returns the list of valid options that can be passed as values
    for parameters in BaseCurate

    Returns:
        dict: Mapping of parameters and their corresponding list of options
    """
    return {
        "curate_feature_extractor": ["TfidfVectorizer", "CountVectorizer"],
        "calibration_method": [None, "calibrate_using_baseline"],
    }


def _fetch_curation_artifacts(
    curate_feature_extractor: Union[
        str, TfidfVectorizer, CountVectorizer, SentenceTransformerVectorizer
    ],
    curate_model: Union[str, ClassifierMixin],
    calibration_method: Union[str, None],
    **options,
) -> Tuple[
    Union[TfidfVectorizer, CountVectorizer, SentenceTransformerVectorizer],
    ClassifierMixin,
    Union[str, None],
]:
    """Checks if any of the input parameter values are invalid

    Args:
        curate_feature_extractor (Union[str, TfidfVectorizer, CountVectorizer, SentenceTransformerVectorizer]): Feature extraction method
                                    to be used during curation.
        curate_model (Union[str, ClassifierMixin]): Machine learning model to be used during curation.
        calibration_method (Union[str, None]): Approach to be used for calibration
                                     of `curate_model` predictions

    Returns:
        Tuple[Union[TfidfVectorizer, CountVectorizer, SentenceTransformerVectorizer], ClassifierMixin, Union[str, None]] - the curation artifacts

    Raises:
        ValueError: If any of the input parameter values are invalid
    """
    supported = _fetch_supported_implementations()
    msg = ""
    feat_extractor = model = None
    is_fe_valid = False
    is_model_valid = False

    if isinstance(curate_feature_extractor, str):
        if curate_feature_extractor not in supported["curate_feature_extractor"]:
            try:
                SentenceTransformer(curate_feature_extractor)
                is_fe_valid = True
                feat_extractor = SentenceTransformerVectorizer(curate_feature_extractor)
            except OSError as ose:
                pass
        else:
            is_fe_valid = True
            params = {
                "analyzer": options.get("analyzer", "word"),
                "ngram_range": options.get("ngram_range", (1, 1)),
            }

            if curate_feature_extractor == "tfidf":
                feat_extractor = TfidfVectorizer(**params)
            else:
                feat_extractor = CountVectorizer(**params)
    elif isinstance(curate_feature_extractor, TfidfVectorizer) or isinstance(
        curate_feature_extractor, CountVectorizer
    ):
        is_fe_valid = True
        feat_extractor = curate_feature_extractor

    elif (
        isinstance(curate_feature_extractor, SentenceTransformer)
        and curate_feature_extractor[0] is not None
    ):
        is_fe_valid = True
        feat_extractor = SentenceTransformerVectorizer(curate_feature_extractor)

    if not is_fe_valid:
        msg = (
            f"curate_feature_extractor '{curate_feature_extractor}' is not supported. "
        )
        msg += "Currently, we support the following feature extraction methods - \n* "
        msg += (
            f"{', '.join(supported['curate_feature_extractor'])}"
            + " which map to corresponding optimized sklearn implementations"
        )
        msg += "* Sklearn TfidfVectorizer / Sklearn CountVectorizer object\n"
        msg += "* SentenceTransformer object or a string matching the name of a SentenceTransformer model hosted on HuggingFace Model Hub. \n"

    if isinstance(curate_model, ClassifierMixin):
        is_model_valid = True
        model = curate_model

    if not is_model_valid:
        msg += f"curate_model '{curate_model}' is not supported. "
        msg += f"Currently, any sklearn classifier object that  are supported\n"

    if calibration_method not in supported["calibration_method"]:
        msg += f"calibration_method '{calibration_method}' is not supported. "
        msg += f"Currently, {', '.join(map(str, supported['calibration_method']))} are supported\n"

    if len(msg) > 0:
        raise ValueError(msg)

    return feat_extractor, model, calibration_method


def _get_pipeline(
    curate_feature_extractor: Union[
        TfidfVectorizer, CountVectorizer, SentenceTransformerVectorizer
    ],
    curate_model: ClassifierMixin,
    **options,
) -> Pipeline:
    """Returns the pipeline for a given curation feature extractor and model choice

    Args:
        curate_feature_extractor (Union[TfidfVectorizer, CountVectorizer, SentenceTransformerVectorizer]): Feature extraction method to be used during curation.
        curate_model (ClassifierMixin): Machine learning model that is trained with
                     `curate_feature_extractor` features during curation.

    Returns:
        Pipeline: the constructed sklearn pipeline
    """
    return Pipeline(
        [
            ("curate_feature_extractor", curate_feature_extractor),
            ("curate_model", curate_model),
        ]
    )


def _data_splitter(
    df: pd.DataFrame,
    X_col_name: str,
    y_col_name_int: str,
    strategy: str = "stratified_kfold",
    n_splits: int = 5,
) -> Generator:
    """Splits the data into train and validation to be consumed in CrossValCurate

    Args:
        df (pd.DataFrame): Input data
        X_col_name (str): Column to be used to extract input features
        y_col_name_int (str): Label column in data
        strategy (str, optional): Strategy to use while splitting the data. Defaults to 'stratified_kfold'.
        n_splits (int, optional): The number of splits. Defaults to 5.

    Returns:
        StratifiedKFold: Currently, a StratifiedKFold object.
    """

    if strategy == "stratified_kfold":
        cv = StratifiedKFold(n_splits=n_splits)
        return cv.split(df[X_col_name], df[y_col_name_int])

    raise ValueError(
        f"Data splitting strategy {strategy} not supported. Please use `strategy='stratified_kfold'`"
    )
