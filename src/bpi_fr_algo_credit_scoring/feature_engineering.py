import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def feature_engineering(
    default_risk_dataset: pd.DataFrame,
    to_drop_feat: List[str] = None,
    to_drop_companies: List[int] = None,
) -> pd.DataFrame:
    """Drop a collection of features, companies and handle missing values

    Parameters
    ----------
    default_risk_dataset: pd.DataFrame
    to_drop_feat: List
    to_drop_companies: List

    Returns
    -------
    pd.DataFrame

    """
    drop_features(default_risk_dataset, to_drop_feat)
    drop_companies(default_risk_dataset, to_drop_companies)
    handle_missing_na(default_risk_dataset)
    return default_risk_dataset


def drop_features(
    default_risk_dataset: pd.DataFrame, to_drop_feat: List[str] = None
) -> pd.DataFrame:
    """Drop a list of defined variables from a cleaned dataset

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
        default_risk_dataset.drop(columns=to_drop_feat, inplace=True)
    logger.info("Variables dropped: Dataset ready for comparisons of models")
    return default_risk_dataset


def drop_companies(
    default_risk_dataset: pd.DataFrame, to_drop_companies: List[int] = None
) -> pd.DataFrame:
    """Drop a list of defined variables from a cleaned dataset

    Parameters
    ----------
    default_risk_dataset: pd.Dataframe
    to_drop_companies: list

    Returns
    -------
    pd.Dataframe

    """
    if to_drop_companies:
        logger.info(f"Companies IDs to be dropped: {to_drop_companies}")
        default_risk_dataset.drop(index=to_drop_companies, inplace=True)
        logger.info("Companies dropped")
    return default_risk_dataset


def handle_missing_na(default_risk_dataset: pd.DataFrame) -> pd.DataFrame:
    """Replace NA values by mean values of variables from original dataset

    Parameters
    ----------
    default_risk_dataset: pd.Dataframe

    Returns
    -------
    pd.Dataframe

    """
    logger.info("Replacement of NA values by variables means")
    default_risk_dataset = default_risk_dataset.apply(fill_na_by_mean, axis=1)
    logger.info("Replacement of NA values done!")
    return default_risk_dataset


def fill_na_by_mean(pd_series) -> pd.Series:
    """Enable to replace all NA values of a vector by its mean

    Parameters
    ----------
    pd_series: pd.Series

    Returns
    -------
    pd.Series

    """
    return pd_series.fillna(pd_series.mean())
