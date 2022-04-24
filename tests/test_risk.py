from tests.test_exploratory_analysis import mock_application, mock_monthly_outcome
from monzo_decision_scientist.data.risk import indexes_where_is_bad_at


def mock_call():
    mock_application, mock_monthly_outcome


def test_is_bad_at(mock_application, mock_monthly_outcome):
    idx_bad_12m = indexes_where_is_bad_at(mock_monthly_outcome.join(mock_application), months=12, repayment_threshold=3)
    is_bad_12m = mock_application.index.isin(idx_bad_12m).astype(float)
    assert all(is_bad_12m == mock_application.is_bad_12m.fillna(0))
