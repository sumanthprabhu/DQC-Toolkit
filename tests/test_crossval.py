from collections import Counter
from io import StringIO

import pandas as pd
import pytest
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import dqc
from dqc import CrossValCurate
from dqc.utils import add_asymmetric_noise

sio = StringIO()


def sample_dataframe_for_stratified_split(
    df: pd.DataFrame, n_splits: int
) -> pd.DataFrame:
    """Optimally sample the input dataframe to reduce the size without affecting the tests

    Args:
        df (pd.DataFrame): Input data
        n_splits (int): `n_splits` set for CrossValCurate

    Returns:
        pd.DataFrame: Sampled df
    """

    return (
        df.groupby("label").sample(n=n_splits, random_state=42).reset_index(drop=True)
    )


def run_cvc(
    data,
    curate_feature_extractor,
    curate_model,
    n_splits,
    calibration_method,
    random_state,
):
    sampled_data = sample_dataframe_for_stratified_split(data, n_splits)
    cvc = CrossValCurate(
        curate_feature_extractor=curate_feature_extractor,
        curate_model=curate_model,
        n_splits=n_splits,
        calibration_method=calibration_method,
        random_state=random_state,
    )
    data_curated = cvc.fit_transform(sampled_data)

    # Check if dataframe is return
    assert type(data_curated) == pd.DataFrame

    # Check column presence
    assert all(
        col in data_curated.columns
        for col in [
            "prediction_probability",
            "predicted_label",
            "label_correctness_score",
            "is_label_correct",
        ]
    )

    # Check data types
    assert all(
        data_curated[col].dtype == dtype
        for col, dtype in [
            ("prediction_probability", float),
            ("label_correctness_score", float),
            ("is_label_correct", bool),
        ]
    )

    # Check non-null values
    assert all(
        data_curated[col].notnull().all()
        for col in [
            "prediction_probability",
            "predicted_label",
            "label_correctness_score",
            "is_label_correct",
        ]
    )

    print(cvc, file=sio)
    result = sio.getvalue().strip()
    assert result.startswith("{") and result.endswith("}")


@pytest.mark.parametrize(
    "curate_feature_extractor",
    ["TfidfVectorizer", "CountVectorizer", TfidfVectorizer(), CountVectorizer()],
)
@pytest.mark.parametrize(
    "curate_model",
    [
        LogisticRegression(),
        RandomForestClassifier(),
        KNeighborsClassifier(),
    ],
)
@pytest.mark.parametrize(
    "n_splits",
    [3, 5],
)
def test_crossvalcurate_success_feature_and_model(
    data,
    curate_feature_extractor,
    curate_model,
    n_splits,
    calibration_method=None,
    random_state=None,
):
    run_cvc(
        data,
        curate_feature_extractor,
        curate_model,
        n_splits,
        calibration_method,
        random_state,
    )


@pytest.mark.parametrize("calibration_method", [None, "calibrate_using_baseline"])
@pytest.mark.parametrize("random_state", [None, 1])
def test_crossvalcurate_success_calibration_and_randomstate(
    data,
    calibration_method,
    random_state,
    curate_feature_extractor="TfidfVectorizer",
    curate_model=LogisticRegression(),
    n_splits=5,
):
    run_cvc(
        data,
        curate_feature_extractor,
        curate_model,
        n_splits,
        calibration_method,
        random_state,
    )


@pytest.mark.parametrize(
    "curate_feature_extractor",
    [
        SentenceTransformer("BAAI/bge-small-en-v1.5"),
        "paraphrase-MiniLM-L6-v2",
    ],
)
@pytest.mark.parametrize("calibration_method", ["calibrate_using_baseline"])
def test_crossvalcurate_success_sentence_transformers(
    data,
    calibration_method,
    curate_feature_extractor,
    curate_model=LogisticRegression(),
    random_state=None,
    n_splits=5,
):
    run_cvc(
        data,
        curate_feature_extractor,
        curate_model,
        n_splits,
        calibration_method,
        random_state,
    )


@pytest.mark.parametrize(
    "curate_feature_extractor", ["tf_idf", 1, "fasfa", "", SentenceTransformer(), None]
)
@pytest.mark.parametrize(
    "curate_model",
    ["logistic regression", 2, "", None],
)
@pytest.mark.parametrize("calibration_method", ["random", 3, ""])
@pytest.mark.parametrize("random_state", ["randomstr", "", -1])
@pytest.mark.parametrize("n_splits", ["randomstr", -1])
def test_crossvalcurate_failure(
    data,
    curate_feature_extractor,
    curate_model,
    n_splits,
    calibration_method,
    random_state,
):
    with pytest.raises(ValueError):
        cvc = CrossValCurate(
            curate_feature_extractor=curate_feature_extractor,
            curate_model=curate_model,
            n_splits=n_splits,
            calibration_method=calibration_method,
            random_state=random_state,
        )
        data = cvc.fit_transform(data)


def test_crossvalcurate_datafailure(data):
    with pytest.raises(ValueError):
        cvc = CrossValCurate()
        data = data.groupby("label", group_keys=False).sample(
            frac=0.005, random_state=43
        )
        data = cvc.fit_transform(data)


@pytest.mark.parametrize("calibration_method", [None, "calibrate_using_baseline"])
@pytest.mark.parametrize("verbose", [True, False])
def test_verbosity(data, calibration_method, verbose):
    cvc = CrossValCurate(calibration_method=calibration_method, verbose=verbose)
    data = cvc.fit_transform(data)


@pytest.mark.parametrize("noise_prob", [0.1])
@pytest.mark.parametrize("random_state", [None, 1])
def test_add_asymmetric_noise(data, noise_prob, random_state):
    noisy_labels, observed_noise_ratio = add_asymmetric_noise(
        data["label"], noise_prob=noise_prob, random_state=random_state
    )

    assert isinstance(noisy_labels, pd.Series) and isinstance(
        observed_noise_ratio, float
    )
    assert len(noisy_labels) == len(data["label"])


def test_show_versions():
    dqc_info = dqc.show_versions()
    assert type(dqc_info) == dict and len(dqc_info.keys()) == 7
