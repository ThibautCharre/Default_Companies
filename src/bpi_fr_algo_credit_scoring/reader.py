import logging

import arff
import pandas as pd

logger = logging.getLogger(__name__)


def read_data(default_year=1) -> pd.DataFrame:
    """Read all data from local files

    Returns:
        pd.DataFrame: _description_
    """
    data_set = {}
    for year in range(1, 6):
        data_set[year] = read_yearly_data(f"data/{str(year)}year.arff")
        data_set[year]["X_65"] = data_set[year]["X_65"].astype("int")
    return data_set[default_year]


def read_yearly_data(path: str) -> pd.DataFrame:
    """Read all data from local files

    Returns:
        pd.DataFrame: _description_
    """
    logger.info(f"File path is {path}")
    dt = pd.DataFrame(
        data=arff.load(open(file=path,
                            mode="r",
                            encoding="utf-8"))["data"],
        columns=[f"X_{n_col}" for n_col in range(1, 66)],
    )
    dt["X_65"] = dt["X_65"].astype("int")
    return dt
