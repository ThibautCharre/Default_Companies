import logging
from pathlib import Path

from bpi_fr_algo_credit_scoring.reader import read_yearly_data
from bpi_fr_algo_credit_scoring.data_pipeline import clean_dataset
from tests.conftest import TEST_DATA_ROOT

logger = logging.getLogger(__name__)


def test_data_pipeline():
    # Given
    test_file_path = Path.joinpath(TEST_DATA_ROOT)
    dataset = read_yearly_data(test_file_path)
    # When
    dataset = clean_dataset(dataset)
    # Then
    assert dataset.shape[0] != 0