from abc import ABC, abstractmethod
from typing import Union

from pandas._typing import RandomState


class BaseCurate(ABC):
    """Base class for data curation to compute label correctness scores
      and identify reliable labelled samples

    Args:
        random_state (RandomState, optional): Random seed for
                                reproducibility. Defaults to None.
    """

    def __init__(
        self,
        random_state: RandomState = 42,
        **options,
    ):
        self.random_state = random_state

    @abstractmethod
    def fit_transform(self): ...
