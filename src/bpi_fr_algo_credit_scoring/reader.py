import logging

import arff
import pandas as pd

logger = logging.getLogger(__name__)


def read_yearly_data(path: str,
                     default_year: int = 1) -> pd.DataFrame:
    """

    Parameters
    ----------
    path: String character indicating directory of data file
    default_year: Integer representing default year of companies (default is 1)

    Returns
    -------
    Dataframe containing data of default and non-default companies

    """
    logger.info(f"importing dataset for {default_year} year default companies")
    dt = pd.DataFrame(
        data=arff.load(open(file=f'{path}/{default_year}year.arff',
                            mode="r",
                            encoding="utf-8"))["data"],
        columns=[f"X_{n_col}" for n_col in range(1, 66)],
    )
    dt["X_65"] = dt["X_65"].astype("int")
    logger.info(f"Dataset correctly imported")
    return dt
