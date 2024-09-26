import random
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch
from pandas._typing import RandomState


class BaseCurate(ABC):
    """Base class for data curation to compute label correctness scores
      and identify reliable labelled samples

    Args:
        random_state (RandomState, optional): Random seed for
                                reproducibility. Defaults to 42.
    """

    def __init__(
        self,
        random_state: Union[int, RandomState] = 42,
        **options,
    ):
        self.random_state = random_state
        self._set_seed(random_state)

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    @abstractmethod
    def fit_transform(self): ...
