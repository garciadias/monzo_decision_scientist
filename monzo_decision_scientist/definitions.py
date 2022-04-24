"""Definitions of useful variables."""
import numpy as np


MONTHLY_OUTCOME_DTYPES = {
    'date': np.dtype('<M8[ns]'),
    'status': np.dtype('int64'),
    'defaulted': np.dtype('bool')
}

APPLICATION_DTYPES = {
    'stress_score': np.dtype('O'),
    'is_bad_12m': np.dtype('O'),
    'model_1': np.dtype('float64'),
    'model_2': np.dtype('float64'),
    'origination_date': np.dtype('<M8[ns]'),
    'loan_term': np.dtype('int64'),
    'loan_amount': np.dtype('float64'),
    'age_oldest_account': np.dtype('O'),
    'total_value_of_mortgage': np.dtype('O'),
    'current_utilisation': np.dtype('O'),
    'months_since_2_payments_missed': np.dtype('O'),
    'number_of_credit_searches_last_3_months': np.dtype('O'),
}

COLORS = ["#327887", "#8abb9c", "#dfc586", "#e83860"]
