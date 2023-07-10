import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_dataset(default_risk_dataset: pd.DataFrame,
                  ratio_na_per_features=0.05,
                  nb_na_sample_threshold=0,
                  ratio_under_oversampled=1,
                  random_state=42) -> pd.DataFrame:
    """Clean rows and columns from NAs values

    Returns:
        pd.DataFrame: _description_
    """
    logger.info("clean data")
    default_risk_dataset = select_non_empty_features(
        default_risk_dataset,
        ratio_na_per_features=ratio_na_per_features
    )
    default_risk_dataset = select_non_empty_companies(
        default_risk_dataset,
        nb_na_sample_threshold=nb_na_sample_threshold
    )
    if ratio_under_oversampled != 1:
        default_risk_dataset = resample_dataset(
            default_risk_dataset,
            ratio_under_oversampled=ratio_under_oversampled,
            random_state=random_state
        )
    return default_risk_dataset


def select_non_empty_features(default_risk_dataset: pd.DataFrame,
                              ratio_na_per_features: float = 0.05) -> pd.DataFrame:
    """Clean columns with number of NAs values superior to a threshold.

    Parameters
    ----------
    default_risk_dataset : pd.DataFrame
    ratio_na_per_features : float

    Returns
    -------
    pd.DataFrame

    """
    data_na_columns = (
        default_risk_dataset.isna()
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "Feature", 0: "nb_na"})
    )
    nb_samples = default_risk_dataset.shape[0]
    err_feat = ratio_na_per_features / 100 * nb_samples
    del_feat = data_na_columns.query(f"nb_na > @{err_feat}")["Feature"]
    default_risk_dataset.drop(columns=del_feat, inplace=True)
    return default_risk_dataset


def select_non_empty_companies(default_risk_dataset,
                               nb_na_sample_threshold=1) -> pd.DataFrame:
    """Clean rows containing more than nb_na_sample_threshold NAs

    Returns:
        pd.DataFrame: _description_
    """
    data_na_rows = default_risk_dataset.apply(
        lambda row: np.sum(row.isna()),
        axis=1
    )
    del_rows = data_na_rows[data_na_rows > nb_na_sample_threshold]
    default_risk_dataset.drop(
        index=del_rows.index,
        inplace=True
    )
    return default_risk_dataset


def resample_dataset(default_risk_dataset: pd.DataFrame,
                     ratio_under_oversampled=0.2,
                     random_state=42) -> pd.DataFrame:
    """Under sampling of imbalanced dataset

    Returns:
        pd.DataFrame: _description_
    """
    nb_default = np.sum(default_risk_dataset.loc[default_risk_dataset.X_65 > 0, "X_65"])
    nb_samples = default_risk_dataset.shape[0]
    np.random.seed(random_state)
    drop_indices = np.random.choice(
        default_risk_dataset[default_risk_dataset.X_65 == 0].index,
        int(nb_samples - round(nb_default / ratio_under_oversampled, 0)),
        replace=False,
    )
    default_risk_dataset.drop(index=drop_indices, inplace=True)
    return default_risk_dataset