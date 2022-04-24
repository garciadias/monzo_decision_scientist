from pandas.core.generic import NDFrame
import numpy as np
import pandas as pd
import pytest

from monzo_decision_scientist.data.exploratory_analysis import clean_application, clean_monthly_outcome
from monzo_decision_scientist.definitions import APPLICATION_DTYPES, MONTHLY_OUTCOME_DTYPES


@pytest.fixture()
def mock_application():
    mock_data = {
        'stress_score': {0: np.nan, 2: np.nan, 3: 0.5753530494918887, 7: np.nan},
        'is_bad_12m': {0: 0.0, 2: 0.0, 3: np.nan, 7: 1.0},
        'model_1': {0: 0.0043715178437998,
                    2: 0.0077405689558199,
                    3: 0.0038872478723798,
                    7: 0.0196935248858327},
        'model_2': {0: 0.0232937096246875,
                    2: 0.0048397768814009,
                    3: 0.0210340371827449,
                    7: 0.0413110229984667},
        'origination_date': {0: np.datetime64('2019-05-06'),
                             2: np.datetime64('2019-07-31'),
                             3: np.datetime64('2019-12-17'),
                             7: np.datetime64('2019-08-30')},
        'loan_term': {0: 36, 2: 12, 3: 30, 7: 60},
        'loan_amount': {0: 10707.0, 2: 903.0, 3: 9688.0, 7: 7610.0},
        'age_oldest_account': {0: 125.0, 2: 441.0, 3: 59.0, 7: 97.0},
        'total_value_of_mortgage': {0: np.nan, 2: np.nan, 3: np.nan, 7: np.nan},
        'current_utilisation': {0: 121.0, 2: 8.0, 3: 0.0, 7: 88.0},
        'months_since_2_payments_missed': {0: np.nan, 2: 3.0, 3: np.nan, 7: np.nan},
        'number_of_credit_searches_last_3_months': {0: 0.0, 2: 1.0, 3: 1.0, 7: 0.0},
    }
    application = pd.DataFrame(mock_data).astype(APPLICATION_DTYPES)
    application.index.name = "unique_id"
    return application


@pytest.fixture()
def mock_monthly_outcome():
    mock_data = {
        'date': {0: np.datetime64('2020-05-05'),
                 2: np.datetime64('2020-07-30'),
                 7: np.datetime64('2020-08-29'),
                 18: np.datetime64('2020-07-15'),
                 },
        'status': {0: 0, 2: 0, 7: 4, 18: 0},
        'defaulted': {0: False, 2: False, 7: True, 18: False}
    }
    mock_monthly_outcome = pd.DataFrame(mock_data).astype(MONTHLY_OUTCOME_DTYPES)
    dup = pd.DataFrame([[np.datetime64('2020-09-15'), 2, False]], index=[18], columns=mock_monthly_outcome.columns)
    mock_monthly_outcome = pd.concat([mock_monthly_outcome, dup])
    mock_monthly_outcome.index.name = "unique_id"
    return mock_monthly_outcome


@pytest.fixture()
def mock_dirty_monthly_outcome(mock_monthly_outcome):
    dirty_monthly_outcome = mock_monthly_outcome.copy().drop(columns=["defaulted"])
    dirty_monthly_outcome.loc[7, (["status"])] = "D"
    dirty_monthly_outcome.loc[2, (["status"])] = 0.0
    dirty_monthly_outcome = pd.concat([dirty_monthly_outcome, dirty_monthly_outcome.loc[[18, 18]]])
    return dirty_monthly_outcome


@pytest.fixture()
def mock_dirty_application(mock_application):
    dirty_application = mock_application.copy()
    dirty_application.loc[7, (["stress_score"])] = -999997
    dirty_application.loc[2, (["age_oldest_account"])] = -999999
    dirty_application = pd.concat([dirty_application, dirty_application.loc[[7, 7]]])
    return dirty_application


def test_mockdataframes(mock_application, mock_monthly_outcome):
    assert isinstance(mock_application, NDFrame)
    assert isinstance(mock_monthly_outcome, NDFrame)
    assert mock_monthly_outcome.date.dtype == np.dtype('<M8[ns]')


def test_clean_monthly_outcome(mock_monthly_outcome, mock_dirty_monthly_outcome):
    clean_mock_monthly_outcome = clean_monthly_outcome(mock_dirty_monthly_outcome)
    assert all(clean_mock_monthly_outcome == mock_monthly_outcome)
    assert all(clean_mock_monthly_outcome.dtypes == mock_monthly_outcome.dtypes)


def test_clean_application(mock_application, mock_dirty_application):
    clean_mock_aplpication = clean_application(mock_dirty_application)
    assert all(clean_mock_aplpication == mock_application)
    assert all(clean_mock_aplpication.dtypes == mock_application.dtypes)
