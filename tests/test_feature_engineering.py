import logging

from bpi_fr_algo_credit_scoring.reader import read_yearly_data
from bpi_fr_algo_credit_scoring.feature_engineering import feature_engineering
from bpi_fr_algo_credit_scoring.conf import DATA_ROOT

logger = logging.getLogger(__name__)


def test_feature_engineering():
    # Given
    dataset = read_yearly_data(path=DATA_ROOT, default_year=1)
    # When
    dataset = feature_engineering(
        default_risk_dataset=dataset,
        to_drop_feat=["total liabilities / total assets"],
        to_drop_companies=[5033],
    )
    # Then
    assert dataset.shape[0] != 0
