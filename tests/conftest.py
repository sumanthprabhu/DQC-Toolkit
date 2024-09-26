import pandas as pd
import pytest
from datasets import load_dataset


@pytest.fixture(scope="session")
def data():
    dataset = "mteb/mtop_domain"
    dset = load_dataset(dataset, trust_remote_code=True)
    return pd.DataFrame(dset["test"])[["text", "label", "label_text"]]
