"""
# Part 3. Model monitoring

The training and validation of a model is just part of the story. A large part of our work is to understand how our
models perform in real life deicisioning and how we adapt to the changing market. In this part we will look into the
monitoring sample (see definition at the start).

Now let's assume that we have choosen *model_1* and deployed it to production since 1st Jan 2020. On that day, our
decision engine started to use that model, and since then only approved applications with *model_1*<0.05.
"""
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monzo_decision_scientist.definitions import APPLICATION_DTYPES, MONTHLY_OUTCOME_DTYPES

if __name__ == '__main__':
    monthly_outcome = pd.read_csv("data/clean/monthly_outcome.csv", index_col=0).astype(MONTHLY_OUTCOME_DTYPES)
    application = pd.read_csv("data/processed/application_with_is_bad_3m.csv", index_col=0).astype(APPLICATION_DTYPES)

    # filter out-of-time
    monthly_outcome_out_of_time = monthly_outcome[monthly_outcome.date.between(np.datetime64("2019-08-01"),
                                                                               np.datetime64("2020-01-01"))]
    application_out_of_time = application.loc[monthly_outcome_out_of_time.index.unique()]

    # filter post-deployment
    monthly_outcome_post_deployment = monthly_outcome[monthly_outcome.date.lt(np.datetime64("2020-01-01"))]
    application_post_deployment = application.loc[monthly_outcome_post_deployment.index.unique()]

    y_true_out_of_time = application_out_of_time.is_bad_12m.fillna(0).astype(int)
    score_1_out_of_time = application_out_of_time.model_1

    y_true_post_deployment = application_post_deployment.is_bad_12m.fillna(0).astype(int)
    score_1_post_deployment = application_post_deployment.model_1

    threshold = 0.05
    y_pred_out_of_time = score_1_out_of_time > threshold
    y_pred_post_deployment = score_1_post_deployment > threshold
    balanced_accuracy_out_of_time = balanced_accuracy_score(y_true_out_of_time, y_pred_out_of_time)
    precision_out_of_time = precision_score(y_true_out_of_time, y_pred_out_of_time)
    recall_out_of_time = recall_score(y_true_out_of_time, y_pred_out_of_time)

    balanced_accuracy_post_deployment = balanced_accuracy_score(y_true_post_deployment, y_pred_post_deployment)
    precision_post_deployment = precision_score(y_true_post_deployment, y_pred_post_deployment)
    recall_post_deployment = recall_score(y_true_post_deployment, y_pred_post_deployment)

    RESULTS = pd.DataFrame([
        [balanced_accuracy_out_of_time, precision_out_of_time, recall_out_of_time],
        [balanced_accuracy_post_deployment, precision_post_deployment, recall_post_deployment],
    ],
        index=["Out-Of-Time", "Post-Deployment"],
        columns=["Balanced Accuracy", "Precisions", "Recall"])

    fig, ax = plt.subplots(figsize=(16*0.7, 9*0.7))
    RESULTS.plot.bar(color=["#8abb9c", "#dfc586", "#e83860"], ax=ax)
    # plt.setp(ax.get_xticklabels(), rotation=0)
    plt.tick_params(labelsize=18, rotation=0)
    ax.minorticks_on()
    ax.grid()
    plt.tight_layout()
    plt.savefig("data/reports/metrics_comparison_before_and_after_deployment.png", dpi=180)
