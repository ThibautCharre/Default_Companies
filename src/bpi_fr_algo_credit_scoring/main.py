import logging

from bpi_fr_algo_credit_scoring.feature_engineering import select_features
from bpi_fr_algo_credit_scoring.data_pipeline import clean_dataset
from bpi_fr_algo_credit_scoring.reader import read_yearly_data
from bpi_fr_algo_credit_scoring.model import models_comparison
from bpi_fr_algo_credit_scoring.conf import DATA_ROOT, log_config

logger = logging.getLogger(__name__)


def run():
    logger.info("Start of the RUN task")
    default_risk_dataset = read_yearly_data(path=DATA_ROOT, default_year=1)
    cleaned_default_risk = clean_dataset(
        default_risk_dataset,
        ratio_na_per_features=0.05,
        nb_na_sample_threshold=0,
        ratio_under_oversampled=0.2,
    )
    cleaned_default_risk_2 = select_features(
        cleaned_default_risk,
        to_drop_feat=[
            "total liabilities / total assets",
            "(gross profit + depreciation) / sales",
            "gross profit / sales",
            "(inventory * 365) / sales",
            "net profit / sales",
            "(total liabilities - cash) / sales",
            "(gross profit + interest) / sales",
            "constant capital / total assets",
            "profit on sales / sales",
            "profit on operating activities / sales",
            "rotation receivables + inventory turnover in days",
            "(receivables * 365) / sales",
            "EBITDA(profit on operating activities - depreciation) / sales",
            "(sales - cost of products sold) / sales",
        ],
    )
    algorithms_results = models_comparison(
        cleaned_default_risk_2,
        test_size=0.2,
        random_state=42,
        selected_models=["TREE", "RF", "XGB"],
    )
    logger.info("Calculations are done !")
    return algorithms_results
