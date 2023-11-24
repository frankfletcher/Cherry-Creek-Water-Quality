# import cudf.pandas
# import cuml

# from joblib import Parallel, delayed
# from IPython.display import display

# # import cupy as cp
# import numpy as np
# from sklearn.neighbors import KDTree  # BallTree

# cuml.set_global_output_type("cudf")
# cudf.pandas.install()
# import pandas as pd
# import json
# from geopy.geocoders import Nominatim
# import os
# from tqdm.auto import tqdm

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


class Preprocessor:
    def __init__(self, config=None) -> None:
        self.config = config
