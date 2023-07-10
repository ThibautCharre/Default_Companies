import logging
from pathlib import Path

from bpi_fr_algo_credit_scoring.reader import read_yearly_data
from conftest import TEST_DATA_ROOT

logger = logging.getLogger(__name__)


def test_read_data():
    # Given
    test_file_path = Path.joinpath(TEST_DATA_ROOT, "1year.arff")
    # When
    dataset = read_yearly_data(test_file_path)
    # Then
    logger.info(test_file_path)
    assert dataset.shape[0] != 0
