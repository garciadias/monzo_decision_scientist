from pandas.core.generic import NDFrame

from tests.test_exploratory_analysis import mock_application
from monzo_decision_scientist.metrics.gini_coef import gini_score, compare_gini_from_two_models


def mock_call():
    mock_application


def test_gini_score(mock_application):
    y_true = mock_application.is_bad_12m.fillna(0)
    score_1 = mock_application.model_1
    gini_1 = gini_score(y_true, score_1)
    assert isinstance(gini_1, float)
    assert gini_1 >= 0
    assert gini_1 <= 1


def test_compare_gini_from_two_models(mock_application):
    y_true = mock_application.is_bad_12m.fillna(0)
    score_1 = mock_application.model_1
    score_2 = mock_application.model_2
    GINI_A = compare_gini_from_two_models(y_true, score_1, score_2)
    GINI_B = compare_gini_from_two_models(y_true, [0, 0, 0, 0], [0., 0., 0., 0.9], index_label="Test Gini")
    assert isinstance(GINI_A, NDFrame)
    assert GINI_A.shape == (1, 2)
    assert all(GINI_A.values[0] == [1, 1])
    assert all(GINI_B.values[0] == [0, 1])
    assert GINI_B.index[0] == "Test Gini"
