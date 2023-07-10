import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def select_features(default_risk_dataset: pd.DataFrame,
                    to_drop_feat: List[str] = None) -> pd.DataFrame:
    """ Drop a list of defined variables from a cleaned dataset

    Parameters
    ----------
    default_risk_dataset: pd.Dataframe
    to_drop_feat: list

    Returns
    -------
    pd.Dataframe

    """
    logger.info(f"Variables to be dropped: {to_drop_feat}")
    if to_drop_feat:
        default_risk_dataset.drop(
            columns=to_drop_feat,
            inplace=True
        )
    logger.info("Variables dropped: Dataset ready for comparisons of models")
    return default_risk_dataset
