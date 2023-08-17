import logging

from bpi_fr_algo_credit_scoring.reader import read_yearly_data
from bpi_fr_algo_credit_scoring.conf import DATA_ROOT

logger = logging.getLogger(__name__)


def test_read_data():
    # Given
    dataset = read_yearly_data(path=DATA_ROOT, default_year=1)
    # Then
    assert dataset.shape[0] != 0
