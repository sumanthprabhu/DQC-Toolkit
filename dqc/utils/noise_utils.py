from typing import Tuple, Union

import numpy as np
import pandas as pd
from pandas._typing import RandomState


def add_asymmetric_noise(
    labels: pd.Series,
    noise_prob: float,
    random_state: Union[RandomState, None] = 42,
) -> Tuple[pd.Series, float]:
    """
    Util function to add asymmetric noise to labels
    for simulation of noisy label scenarios.

    Args:
        labels (pd.Series): Input pandas series with integer values
                        ranging from 0 to n - 1.
        noise_prob (float): Probability of adding noise to each value.
        random_state (Union[RandomState, None]): Random seed for reproducibility
    Returns:
        pd.Series: Series with asymmetric noise added to it.
        float: Normalized quantification of pairwise disagreement between `labels` and `noisy_labels` for parity check
    """
    # Set seed
    np.random.seed(random_state)

    # Avoid modifying the original data
    noisy_labels = labels.copy()

    # Build a replacement dictionary
    unique_labels = list(set(noisy_labels))
    replacement_dict = {
        label: [candidate for candidate in unique_labels if candidate != label]
        for label in unique_labels
    }

    # Determine the number of samples to modify based on the noise probability
    num_samples = min(len(noisy_labels), int(len(noisy_labels) * noise_prob + 1))

    # Sample random indices from the labels to introduce noise
    target_indices = np.random.choice(len(noisy_labels), num_samples, replace=False)

    for idx in target_indices:
        # Introduce noise
        noisy_labels[idx] = np.random.choice(replacement_dict[noisy_labels[idx]])

    # Parity check
    num_mismatches = sum(
        [
            label != noisy_label
            for label, noisy_label in zip(labels.values, noisy_labels.values)
        ]
    )
    observed_noise_ratio = num_mismatches / len(noisy_labels)

    return noisy_labels, observed_noise_ratio
