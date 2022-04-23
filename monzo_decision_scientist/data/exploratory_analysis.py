"""
General exploration of the data.
Here I explore the data, replacing the large negative numbers by NaN values.
I provide some visializations of the data to
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sweetviz

COLORS = ["#327887", "#8abb9c", "#dfc586", "#e83860"]

monthly_outcome = pd.read_csv("data/raw/monthly_outcome.csv", index_col=0, na_values=[-999997, -999999.0])
monthly_outcome.loc[:, ("date")] = pd.to_datetime(monthly_outcome.date, format='%Y-%m-%d')
monthly_outcome.loc[:, ("defaulted")] = monthly_outcome.status == "D"
monthly_outcome.loc[:, ("status")] = monthly_outcome.status.replace("D", 4).astype(float).astype(int)
monthly_outcome.dtypes.to_csv("data/clean/monthly_outcome_dtypes.csv")
monthly_outcome.to_csv("data/clean/monthly_outcome.csv")

MONTHLY_OUTCOME_DTYPES = pd.read_csv("data/clean/monthly_outcome_dtypes.csv", index_col=0)["0"].to_dict()
monthly_outcome = pd.read_csv("data/clean/monthly_outcome.csv", index_col=0).astype(MONTHLY_OUTCOME_DTYPES)

monthly_outcome_view = sweetviz.analyze(monthly_outcome)
monthly_outcome_view.show_html(filepath="data/reports/monthly_outcome_view.html")

DTYPES = {
    'is_bad_12m': int,
    'loan_term': int,
    'age_oldest_account': int,
    'total_value_of_mortgage': int,
    'current_utilisation': int,
    'months_since_2_payments_missed': int,
    'number_of_credit_searches_last_3_months': int,
    'origination_date': np.datetime64,
}
application = pd.read_csv("data/raw/application.csv", index_col=0, na_values=[-999997, -999999.0])
application = application.fillna(-999999).astype(DTYPES).replace(-999999, None)
application.dtypes.to_csv("data/clean/application_dtypes.csv")
application.to_csv("data/clean/application.csv")
APPLICATION_DTYPES = pd.read_csv("data/clean/application_dtypes.csv", index_col=0)["0"].to_dict()
application = pd.read_csv("data/clean/application.csv", index_col=0).astype(APPLICATION_DTYPES)
application_view = sweetviz.analyze(application)
application_view.show_html(filepath="data/reports/application_view.html")

fig, ax = plt.subplots(figsize=(16*0.8, 9*0.8))
monthly_outcome.resample(rule='M', on='date')["status"].count().plot.bar(ax=ax, color="#8abb9c")
ax.set_xticklabels([txt.get_text().split(" ")[0] for txt in ax.get_xticklabels()])
plt.title("Monthy number of loans")
plt.savefig("data/reports/monthy_number_of_loans.png")
plt.show()

fig, ax = plt.subplots(figsize=(16*0.8, 9*0.8))
application.resample(rule='M', on='origination_date')["origination_date"].count().plot.bar(ax=ax, color="#8abb9c")
ax.set_xticklabels([txt.get_text().split(" ")[0] for txt in ax.get_xticklabels()])
plt.savefig("data/reports/application_origination_date.png")
plt.title("Application Origination Date")
plt.show()
