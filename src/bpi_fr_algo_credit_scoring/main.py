import logging

from bpi_fr_algo_credit_scoring.feature_engineering import select_features
from bpi_fr_algo_credit_scoring.data_pipeline import clean_dataset
from bpi_fr_algo_credit_scoring.reader import read_yearly_data
from bpi_fr_algo_credit_scoring.model import models_comparison

logger = logging.getLogger(__name__)


def run():
    logger.info("Start to read data from website")
    default_risk_dataset = read_yearly_data(default_year=1)
    logger.info("Start to clean dataset")
    cleaned_default_risk = clean_dataset(
        default_risk_dataset,
        ratio_na_per_features=0.05,
        nb_na_sample_threshold=0,
        ratio_under_oversampled=0.2
    )
    logger.info("Feature selection")
    cleaned_default_risk_2 = select_features(
        cleaned_default_risk,
        to_drop_feat=[
            "X_2",
            "X_13",
            "X_19",
            "X_20",
            "X_23",
            "X_30",
            "X_31",
            "X_38",
            "X_39",
            "X_42",
            "X_43",
            "X_44",
            "X_49",
            "X_56",
        ]
    )
    logger.info("Model comparison")
    algorithms_results = models_comparison(
        cleaned_default_risk_2,
        test_size=0.2,
        random_state=42,
        selected_models=['TREE', 'RF', 'XGB']
    )
    logger.info("Job is done")
    return algorithms_results
