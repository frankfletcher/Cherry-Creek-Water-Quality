# import cudf.pandas

# cudf.pandas.install()
import pandas as pd

# from cuml.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from config import Config
from IPython.display import Markdown, display

import cudf
import cupy as cp

# import numpy as np
from os.path import splitext

# import matplotlib.pyplot as plt
# import seaborn as sns


class ML:
    """A class for performing machine learning tasks.

    Args:
        config: An optional configuration object. If not provided, a default Config object will be used.

    Attributes:
        config: The configuration object.

    Methods:
        load_data: Loads data from a given path.
        get_summary_stats: Displays summary statistics and information about a DataFrame.

    """

    def __init__(self, config=None) -> None:
        """Initializes the ML class.

        Args:
            config: An optional configuration object. If not provided, a default Config object will be used.

        """
        if config is None:
            config = Config()
        self.config = config

        self.rng = cp.random.default_rng(self.config.random_state)

    def __getitem__(self, key):
        if key in self.get_dataframes(as_names=True):
            return getattr(self, key)
        else:
            raise KeyError(f"{key} is not a valid attribute")

    def __setitem__(self, key, value):
        if key in self.get_dataframes(as_names=True):
            setattr(self, key, value)
        else:
            raise KeyError(f"{key} is not a valid attribute")

    def get_dataframes(self, as_names=False):
        """
        Returns a list of all the dataframes stored in the class.

        Args:
            as_names (bool): Whether to return the names of the dataframes or the dataframes themselves. Defaults to `False`.
        """
        if as_names:
            return [x for x in dir(self) if x.startswith("df_")]

        # else returns objects
        return [getattr(self, x) for x in dir(self) if x.startswith("df_")]

    def get_data_reader(self, fname):
        """
        Factory to return the appropriate pandas reader function based on the file extension of the given filename.

        Args:
            fname (str): The filename with extension.

        Returns:
            function: The pandas reader function for the given file extension.

        """
        if fname.endswith(".csv"):
            return pd.read_csv
        elif fname.endswith(".parquet"):
            return pd.read_parquet

    def load_data(
        self,
        path=None,
        fname=None,
        show_head=False,
        return_df=False,
        storage_df_name="df_train",
    ):
        """The `load_data` function loads data from a specified file path and stores it in an attribute of the `self` object.

        Args:
            path: The path to the data file. If not provided, it defaults to the `data_path` attribute of the `config` object.
            fname: The name of the data file. If not provided, it defaults to the `train_fname` attribute of the `config` object.
            show_head: Whether to display the head of the loaded DataFrame. Defaults to `False`.
            return_df: Whether to return the loaded DataFrame. Defaults to `False`.
            storage_df_name: The name to use for storing the loaded DataFrame. Defaults to "df_train".

        Returns:
            If `return_df` is `True`, the loaded DataFrame is returned.

        Examples:
            # Load data from default path and display head
            ml.load_data(show_head=True)

            # Load data from custom path and return DataFrame
            df = ml.load_data(path="/path/to/data", fname="data.csv", return_df=True)"""

        if path is None:
            path = self.config.data_path
        if fname is None and hasattr(self.config, "train_fname"):
            fname = self.config.train_fname

        stem, _ = splitext(fname)
        if stem != storage_df_name:
            storage_df_name = stem

        if not storage_df_name.startswith("df_"):
            storage_df_name = f"df_{storage_df_name}"

        reader = self.get_data_reader(fname)
        df = reader(path / fname)
        self.config.columns = df.columns
        self.config.cats = df.select_dtypes(include="object").columns
        self.config.nums = df.select_dtypes(include="number").columns

        if show_head:
            display(Markdown(f"# Head of {storage_df_name}"), df.head())

        setattr(self, storage_df_name, df)

        if return_df:
            return df

    def get_summary_stats(self, df_list=None):
        """Displays summary statistics and information about a DataFrame.

        Args:
            df_list: A list of DataFrames to analyze.
                If a single dataframe, it will be transformed into a list.
                If not provided, all dataframes present will be used.
        """
        if df_list is None or df_list == "all":
            df_list = self.get_dataframes(as_names=True)
        elif isinstance(df_list, str):
            df_list = [df_list]
        else:
            raise TypeError("df_list must be a string or list of strings")

        for df_name in df_list:
            df = getattr(self, df_name)
            display(Markdown(f"### {df_name}.describe"), df.describe())

            display(Markdown("<br>"), Markdown(f"### {df_name}.info"))
            display(df.info(verbose=True, memory_usage="deep", show_counts=True))

            display(Markdown("<br>"), Markdown(f"### {df_name} Null Counts"))
            display(df.isna().sum().sort_values(ascending=False))

            display(Markdown("<br>"), Markdown(f"### {df_name} Null Percentages"))
            display(df.isna().mean().sort_values(ascending=False).round(4))

            display(Markdown("<br>"), Markdown(f"### {df_name} Unique Counts"))
            display(df.nunique().sort_values(ascending=False))
            display(Markdown("\n<br/>\n\n<hr/>\n\n"))

    def split_data(
        self, df=None, test_size=0.2, random_state=None, shuffle=True, return_data=False
    ):
        """Splits data into train and validation sets.

        Args:
            df: The DataFrame to split. If not provided, self.train_df will be used.

        """
        if df is None:
            df = self.df_train
        elif isinstance(df, str):
            df = getattr(self, df)

        if random_state is None and hasattr(self.config, "split_random_state"):
            random_state = self.config.split_random_state
        else:
            random_state = 42

        if not isinstance(df, cudf.DataFrame):
            df = cudf.from_pandas(df)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            df.drop("target", axis=1, errors="ignore"),
            df["target"],
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )

        if return_data:
            return self.X_train, self.X_val, self.y_train, self.y_val
