import random
import string
from typing import List, Tuple, Union

import pandas as pd
from pandas._typing import RandomState


class _DataProcessor:
    """Encapsulates data preparation steps pre-curation
    and processing results post-curation."""

    def __init__(self, random_state: RandomState):
        self.random_state = random_state

        self.row_id_col = None

    def _generate_random_suffix(self, length: int = 3) -> str:
        """
        Generates and returns a unique string with letters and/or digits.

        Args:
            length (int): Length of the string (default is 3)

        Returns:
            str: Unique short string
        """
        # Define the characters to use in the string
        characters = string.ascii_letters + string.digits

        # Generate a random string of the specified length
        random_string = "".join(random.choices(characters, k=length))

        return random_string

    def _create_col_name(
        self, df_columns: List[Union[str, int]], base_col_name: str
    ) -> str:
        """Returns a valid non conflicting column name based on
        the input `base_col_name`

        Args:
            df_columns (List[Union[str, int]]): List of column names in the dataframe object
            base_col_name (str): base string to which suffix will be added
        Returns:
            str: new column name
        """
        new_col_name = base_col_name
        while new_col_name in df_columns:
            new_col_name += f"_{self._generate_random_suffix()}"

        return new_col_name

    def _add_row_id(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Adds a new column representing row numbers in input data

        Args:
            pd.DataFrame: Input data
        Returns:
            pd.DataFrame: Data with integer labels
            str: Name of the column added
        """
        # Generate a unique column name
        row_id_col = self._create_col_name(df.columns, base_col_name="row_id")

        # Add row numbers as a new column
        df[row_id_col] = range(len(df))

        # Save name of the new column for downstream usage
        self.row_id_col = row_id_col

        return df, row_id_col

    def _shuffle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shuffles the data using pandas sampling

        Args:
            pd.DataFrame: Input data

            Returns:
                pd.DataFrame: Shuffled data
        """
        return df.sample(frac=1.0, random_state=self.random_state).reset_index(
            drop=True
        )

    def _convert_labels_to_int(
        self, df: pd.DataFrame, y_col_name: str
    ) -> Tuple[pd.DataFrame, str]:
        """Map string labels to integers and add them as a new column in the data
        for downstream processing

        Args:
            df (pd.DataFrame): Data with labels
        Returns:
            pd.DataFrame: Data with integer labels
            str: newly added column name for integer labels
        """

        if not df[y_col_name].dtype == "object":
            return df, y_col_name

        # Generate a unique column name
        y_col_name_int = self._create_col_name(df.columns, base_col_name="label_int")

        # Create a mapping dictionary from string values to integers
        unique_labels = df[y_col_name].unique()
        label_mapping = {text: i for i, text in enumerate(unique_labels)}

        # Map string values to integers and create a new column
        df[y_col_name_int] = df[y_col_name].map(label_mapping)

        return df, y_col_name_int

    def _preprocess(
        self, df: pd.DataFrame, y_col_name: str
    ) -> Tuple[pd.DataFrame, str, str]:
        """
        Adds the following columns to the DataFrame -
        1) 'label_int' with integer values for all labels
        2) row numbers if random_state is not None

        Args:
            pd.DataFrame: Data to be processed
            y_col_name: label column in the data
        Returns:
            pd.DataFrame: Data with added columns
            str: Name of the newly added column
        """

        df, y_col_name_int = self._convert_labels_to_int(df, y_col_name)

        df, row_id_col = self._add_row_id(df)
        df = self._shuffle_data(df)

        return df, row_id_col, y_col_name_int

    def _postprocess(self, df: pd.DataFrame, display_cols: List[str]) -> pd.DataFrame:
        """Removes redundant columns and returns the data
        with selected columns to display

        Args:
            df (pd.DataFrame): Data to be processed
            display_cols (List[str]): Non redundant columns in data

        Returns:
            pd.DataFrame: Data after post-processing data
        """

        row_id_col = self.row_id_col

        if not row_id_col:
            return df[display_cols]

        return df.sort_values(by=[row_id_col]).reset_index(drop=True)[display_cols]
