import pandas as pd
import pytest
from datasets import load_dataset


@pytest.fixture(scope="session")
def data():
    dataset = "mteb/mtop_domain"
    dset = load_dataset(dataset)
    return pd.DataFrame(dset["test"])[["text", "label", "label_text"]]
