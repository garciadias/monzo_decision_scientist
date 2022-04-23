"""
# Task 1.2: another target
In order to understand the performance of a model before it's too late, we also want to monitor the repayment
behaviours after the first few repayments.

Could you please create another "early-risk" target *is_bad_3m* which represents whether the customers ever had
**2 or more** repayments in arrear at any point of their first three scheduled ones?


To solve this problem I create a function that implements the *is bad* criteriasad for a give interval in months
since the origination date and a desired threshold.
"""
import numpy as np
import pandas as pd


def is_bad_at(df, months=3, repayment_threshold=2):
    """Calculates the is bad* criteria.

    Implements the *is bad* criteria for a give interval in months (*months*)
    since the origination date and a desired threshold (*repayment_threshold*).

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing *status*, *date*, and *origination_date* for each loan.
    months : int, default: 3
        The number of months since the origination data where we want to know if there were any arrears.
    repayment_threshold : int, default: 2
        The number of repayments consider to be *bad*.

    Returns
    -------
    numpy array of booleans
        For each loan id returns True if the nubmer of repayments exceeded *repayment_threshold* and False otherwise.
    """
    exceed_repayments = df[df.status >= repayment_threshold].copy()
    exceed_repayments.loc[:, ("loan_age")] = (exceed_repayments["date"]-exceed_repayments["origination_date"])
    exceed_repayments.loc[:, ("loan_age")] = exceed_repayments["loan_age"] / np.timedelta64(1, 'M')
    on_the_period = exceed_repayments[exceed_repayments["loan_age"].between(0, months)]
    index_bad_on_the_period = on_the_period.index.unique()
    return df.index[df.index.isin(index_bad_on_the_period)].unique()


if __name__ == '__main__':
    MONTHLY_OUTCOME_DTYPES = pd.read_csv("data/clean/monthly_outcome_dtypes.csv", index_col=0)["0"].to_dict()
    monthly_outcome = pd.read_csv("data/clean/monthly_outcome.csv", index_col=0).astype(MONTHLY_OUTCOME_DTYPES)

    APPLICATION_DTYPES = pd.read_csv("data/clean/application_dtypes.csv", index_col=0)["0"].to_dict()
    application = pd.read_csv("data/clean/application.csv", index_col=0).astype(APPLICATION_DTYPES)

    application.loc[:, ("is_bad_3m")] = application.index.isin(is_bad_at(monthly_outcome.join(application),
                                                                         months=3, repayment_threshold=2))
    application.dtypes.to_csv("data/processed/application_dtypes_with_is_bad_3m.csv")
    application.to_csv("data/processed/application_with_is_bad_3m.csv")
