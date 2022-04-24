"""
General exploration of the data.
Here I explore the data, replacing the large negative numbers by NaN values.
I provide some visializations of the data to
"""
import pandas as pd
import matplotlib.pyplot as plt
import sweetviz

from monzo_decision_scientist.definitions import APPLICATION_DTYPES, MONTHLY_OUTCOME_DTYPES


def clean_monthly_outcome(monthly_outcome):
    monthly_outcome.loc[:, ("date")] = pd.to_datetime(monthly_outcome.date, format='%Y-%m-%d')
    monthly_outcome.loc[:, ("defaulted")] = monthly_outcome.status == "D"
    monthly_outcome.loc[:, ("status")] = monthly_outcome.status.replace("D", 4).astype(float).astype(int)
    monthly_outcome = monthly_outcome.reset_index().drop_duplicates().set_index("unique_id")
    monthly_outcome = monthly_outcome.astype(MONTHLY_OUTCOME_DTYPES)
    return monthly_outcome


def clean_application(application):
    application = application.fillna(-999999).astype(APPLICATION_DTYPES).replace(-999999, None)
    application = application.reset_index().drop_duplicates().set_index("unique_id")
    return application


if __name__ == '__main__':
    monthly_outcome = pd.read_csv("data/raw/monthly_outcome.csv", index_col=0, na_values=[-999997, -999999.0])
    monthly_outcome = clean_monthly_outcome(monthly_outcome)
    monthly_outcome.to_csv("data/clean/monthly_outcome.csv")

    monthly_outcome_view = sweetviz.analyze(monthly_outcome)
    monthly_outcome_view.show_html(filepath="data/reports/monthly_outcome_view.html")

    application = pd.read_csv("data/raw/application.csv", index_col=0, na_values=[-999997, -999999.0])
    application = clean_application(application)
    application.to_csv("data/clean/application.csv")

    application_view = sweetviz.analyze(application)
    application_view.show_html(filepath="data/reports/application_view.html")

    fig, ax = plt.subplots(figsize=(16*0.8, 9*0.8))
    monthly_outcome.resample(rule='M', on='date')["status"].count().plot.bar(ax=ax, color="#8abb9c")
    ax.set_xticklabels([txt.get_text().split(" ")[0] for txt in ax.get_xticklabels()])
    plt.title("Monthy number of loans")
    plt.tight_layout()
    plt.savefig("data/reports/monthy_number_of_loans.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(16*0.8, 9*0.8))
    application.resample(rule='M', on='origination_date')["origination_date"].count().plot.bar(ax=ax, color="#8abb9c")
    ax.set_xticklabels([txt.get_text().split(" ")[0] for txt in ax.get_xticklabels()])
    plt.tight_layout()
    plt.savefig("data/reports/application_origination_date.png")
    plt.title("Application Origination Date")
    plt.close()
