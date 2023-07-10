import logging
from pathlib import Path

from bpi_fr_algo_credit_scoring.reader import read_yearly_data
from bpi_fr_algo_credit_scoring.data_pipeline import clean_dataset
from bpi_fr_algo_credit_scoring.feature_engineering import select_features
from bpi_fr_algo_credit_scoring.model import models_comparison
from tests.conftest import TEST_DATA_ROOT

logger = logging.getLogger(__name__)


def test_model():
    # Given
    test_file_path = Path.joinpath(TEST_DATA_ROOT)
    dataset = read_yearly_data(test_file_path)
    dataset = clean_dataset(dataset)
    dataset = select_features(dataset, ['X_2'])
    # When
    results = models_comparison(dataset)
    # Then
    assert len(results) != 0