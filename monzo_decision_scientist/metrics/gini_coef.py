"""
Part 2. Model validation
In this part let's assume we are still at the model development stage and look at the development sample only
(see definition at the start). We will skip the model training part here (which is too much fun to finish in 3 hours),
and assume that we already trained two candidate models. These are of course probabilistic classification model, which
you can find their scores in application.csv as columns model_1 and model_2.

We need to compare their performance and decide which one to use in production. The winner model, once deployed, will
be used for decisions of

* Loan approval: the score must be above certain threshold (which can be adjusted during operation) for the application
  to be approved.
* Loss estimate: for each approved loan, we use the model output to predict the probability of default.
* Pricing: based on the loss estimate, we decide the interest rate to be charged in order to cover potential losses.

Task 2.1: classification power
A common metric used in the credit risk modelling world is the Gini coefficient, which can be linearly mapped to ROCAUC
if that's a term you are more familiar with. Could you please compare the Gini's between the two models as a first step?

An extended question: assuming that classification power is all we care about, what are the other reasons to not pick
the model with highest Gini? It's enough to just write down your thoughts.
## Task 2.2: classification power in segments

As the population of future business might have different distributions from the development sample, we would ideally
want the chosen model to be performant in all segments. For simplicity let's stick with univariate segments only.

Could you please compare the Gini's between the two models in the segments of all the variables? Feel free to define
the segments as you see appropriate.
"""
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from monzo_decision_scientist.definitions import APPLICATION_DTYPES, MONTHLY_OUTCOME_DTYPES


def gini_score(y_true, score):
    auc = roc_auc_score(y_true, score)
    return 2*auc - 1


def compare_gini_from_two_models(y_true, score_1, score_2, index_label="Gini coefficient"):
    GINI_MODEL_1 = gini_score(y_true, score_1)
    GINI_MODEL_2 = gini_score(y_true, score_2)
    GINI = pd.DataFrame([[GINI_MODEL_1, GINI_MODEL_2]], columns=["Model 1", "Model 2"], index=[index_label])
    return GINI


def plot_comparison_gini_comparison(y_true, score_1, score_2):
    ROC_FPR_MODEL_1, ROC_TPR_MODEL_1, ROC_THRESHOLDS_MODEL_1 = roc_curve(y_true, score_1)
    ROC_FPR_MODEL_2, ROC_TPR_MODEL_2, ROC_THRESHOLDS_MODEL_2 = roc_curve(y_true, score_2)
    PRECISION_MODEL_1, RECALL_MODEL_1, PR_THRESHOLDS_MODEL_1 = precision_recall_curve(y_true, score_1)
    PRECISION_MODEL_2, RECALL_MODEL_2, PR_THRESHOLDS_MODEL_2 = precision_recall_curve(y_true, score_2)

    GINI = compare_gini_from_two_models(y_true, score_1, score_2)
    GINI_MODEL_1 = GINI["Model 1"][0]
    GINI_MODEL_2 = GINI["Model 2"][0]
    AUC_MODEL_1 = (GINI_MODEL_1 + 1) / 2.
    AUC_MODEL_2 = (GINI_MODEL_2 + 1) / 2.

    auc_model_1_label = f"$AUC_{{model\\; 1}} = $ {round(AUC_MODEL_1, 3):0.3f}"
    auc_model_2_label = f"$AUC_{{model\\; 2}} = $ {round(AUC_MODEL_2, 3):0.3f}"
    pr_model_1_label = "$PR_{{model\\; 1}}$"
    pr_model_2_label = "$PR_{{model\\; 2}}$"
    gini_model_1_label = f"$Gini_{{model\\; 1}} = $ {round(GINI_MODEL_1, 3):0.3f}"
    gini_model_2_label = f"$Gini_{{model\\; 2}} = $ {round(GINI_MODEL_2, 3):0.3f}"
    fig, axis = plt.subplots(ncols=2, figsize=(16*0.8, 9*0.8))
    x = np.linspace(0, 1)
    axis[0].plot(ROC_FPR_MODEL_1, ROC_TPR_MODEL_1, label=auc_model_1_label, color="#327887", lw=3, alpha=0.9)
    axis[0].plot(ROC_FPR_MODEL_2, ROC_TPR_MODEL_2, label=auc_model_2_label, color="#dfc586", lw=3, alpha=0.9)
    axis[0].plot(x, x, label="", color="#e83860", lw=3, alpha=0.9, ls=":")
    axis[0].legend(fontsize=20)
    axis[0].tick_params(labelsize=20)
    axis[0].set_ylabel("True positive Rate\n(positive: is_bad_12m = 1)", fontsize=20)
    axis[0].set_xlabel("False positive Rate", fontsize=20)
    axis[0].set_ylim(-0.01, 1.01)
    axis[0].set_xlim(-0.01, 1.01)
    axis[0].minorticks_on()
    axis[0].grid()
    axis[1].plot(RECALL_MODEL_1, PRECISION_MODEL_1, label=pr_model_1_label, color="#327887", lw=3, alpha=0.9)
    axis[1].plot(RECALL_MODEL_2, PRECISION_MODEL_2, label=pr_model_2_label, color="#dfc586", lw=3, alpha=0.9)
    axis[1].tick_params(labelsize=20, rotation=0)
    axis[1].set_ylabel("Precision", fontsize=20)
    axis[1].set_xlabel("Recall", fontsize=20)
    axis[1].legend([gini_model_1_label, gini_model_2_label], fontsize=20)
    axis[1].set_ylim(-0.01, 1.01)
    axis[1].set_xlim(-0.01, 1.01)
    axis[1].minorticks_on()
    axis[1].grid()
    plt.tight_layout()
    return fig


def get_limits_grafically(df):
    all_variables = [
        'origination_date',
        'loan_term',
        'loan_amount',
        'age_oldest_account',
        'total_value_of_mortgage',
        'current_utilisation',
        'months_since_2_payments_missed',
        'number_of_credit_searches_last_3_months',
    ]
    variable_splits = pd.DataFrame([np.zeros(len(all_variables))], columns=all_variables, index=["arbritary split"])
    variable_splits.loc["arbritary split", "origination_date"] = df.origination_date.mean()
    variable_splits.loc["arbritary split", "loan_term"] = 40
    variable_splits.loc["arbritary split", "loan_amount"] = 4650
    variable_splits.loc["arbritary split", "age_oldest_account"] = 112
    variable_splits.loc["arbritary split", "total_value_of_mortgage"] = 87000
    variable_splits.loc["arbritary split", "current_utilisation"] = 165
    variable_splits.loc["arbritary split", "months_since_2_payments_missed"] = 28
    variable_splits.loc["arbritary split", "number_of_credit_searches_last_3_months"] = 2
    fig, axis = plt.subplots(nrows=4, ncols=2, figsize=(16*0.7, 9*0.7))
    for i, variable in enumerate(all_variables):
        sns.kdeplot(data=df, x=variable, hue="is_bad_12m", ax=axis.flatten()[i],
                    common_norm=False, palette=["#8abb9c", "#e83860"])
        axis.flatten()[i].set_xlim(*df[variable].quantile([0.001, 0.99]))
        axis.flatten()[i].vlines(variable_splits[variable], *axis.flatten()[i].get_ylim(), color="k")
        if i > 0:
            axis.flatten()[i].get_legend().remove()
    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.99, hspace=0.47, wspace=0.15)
    plt.savefig("data/reports/kdeplot_application_by_is_bad_12m.png", dpi=180)
    plt.show()
    return variable_splits


if __name__ == '__main__':
    monthly_outcome = pd.read_csv("data/clean/monthly_outcome.csv", index_col=0).astype(MONTHLY_OUTCOME_DTYPES)
    application = pd.read_csv("data/processed/application_with_is_bad_3m.csv", index_col=0).astype(APPLICATION_DTYPES)

    # filter development
    monthly_outcome_dev = monthly_outcome[monthly_outcome.date < np.datetime64("2019-08-01")]
    application_dev = application.loc[monthly_outcome_dev.index.unique()]

    y_true = application_dev.is_bad_12m.astype(int)
    score_1 = application_dev.model_1
    score_2 = application_dev.model_2

    GINI_ALL = compare_gini_from_two_models(y_true, score_1, score_2, index_label="development")
    plot_comparison_gini_comparison(y_true, score_1, score_2)
    plt.savefig("data/reports/Ginis_coefficient.png", dpi=180)
    plt.show()

    # filter out-of-time
    monthly_outcome_out_of_time = monthly_outcome[monthly_outcome.date.between(np.datetime64("2019-08-01"),
                                                                               np.datetime64("2020-01-01"))]
    application_out_of_time = application.loc[monthly_outcome_out_of_time.index.unique()]
    y_true_out_of_time = application_out_of_time.is_bad_12m.fillna(0).astype(int)
    score_1_out_of_time = application_out_of_time.model_1
    score_2_out_of_time = application_out_of_time.model_2
    GINI_ALL = pd.concat([GINI_ALL, compare_gini_from_two_models(y_true_out_of_time,
                                                                 score_1_out_of_time,
                                                                 score_2_out_of_time,
                                                                 index_label="out-of-time")])

    # filter post-deployment
    monthly_outcome_post_deployment = monthly_outcome[monthly_outcome.date.lt(np.datetime64("2020-01-01"))]
    application_post_deployment = application.loc[monthly_outcome_post_deployment.index.unique()]
    y_true_post_deployment = application_post_deployment.is_bad_12m.fillna(0).astype(int)
    score_1_post_deployment = application_post_deployment.model_1
    score_2_post_deployment = application_post_deployment.model_2
    GINI_ALL = pd.concat([GINI_ALL, compare_gini_from_two_models(y_true_post_deployment,
                                                                 score_1_post_deployment,
                                                                 score_2_post_deployment,
                                                                 index_label="post-deployment")])

    # Segment the data
    VARIABLE_SPLITS = get_limits_grafically(application)
    for variable in VARIABLE_SPLITS.columns:
        lower_split = application[variable] < VARIABLE_SPLITS[variable][0]
        y_true_low = application[lower_split].is_bad_12m.fillna(0).astype(int)
        score_1_low = application[lower_split].model_1
        score_2_low = application[lower_split].model_2
        gini_low = compare_gini_from_two_models(y_true_low, score_1_low, score_2_low, index_label=f"low {variable}")
        y_true_high = application[~lower_split].is_bad_12m.fillna(0).astype(int)
        score_1_high = application[~lower_split].model_1
        score_2_high = application[~lower_split].model_2
        gini_high = compare_gini_from_two_models(
            y_true_high, score_1_high, score_2_high, index_label=f"high {variable}")
        GINI_ALL = pd.concat([GINI_ALL, gini_low, gini_high])
