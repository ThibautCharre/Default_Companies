import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def select_features(default_risk_dataset: pd.DataFrame,
                    to_drop_feat: List[str] = None) -> pd.DataFrame:
    """Drop of a list of features

    Returns:
        pd.DataFrame: _description_
    """
    if to_drop_feat:
        default_risk_dataset.drop(
            columns=to_drop_feat,
            inplace=True
        )
    return default_risk_dataset
