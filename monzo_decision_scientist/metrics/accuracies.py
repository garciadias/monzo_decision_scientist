"""
# Task 2.3: accuracy
As we want to use our model for loss estimates and pricing of each customer, could you please check whether the scores
(as probabilistic predictions) are accurate with respect to the actual "bad rates" (i.e. the fraction of *is_bad_12m*=1
among customers of similar scores)
"""
from sklearn.metrics import balanced_accuracy_score, precision_recall_curve, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from monzo_decision_scientist.definitions import APPLICATION_DTYPES, MONTHLY_OUTCOME_DTYPES


def balanced_accuracy_threshold(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    balanced_accuracies = np.array([balanced_accuracy_score(y_true, y_score > threshold) for threshold in thresholds])
    column_labels = ["balanced_accuracy", "precision", "recall"]
    results = pd.DataFrame(np.concatenate([balanced_accuracies[:, None],
                                           precision[1:, None], recall[1:, None]], axis=1),
                           index=thresholds, columns=[column_labels])
    return results


def compare_accuracy_from_two_models(metrics_model_1, metrics_model_2):
    fig = plt.figure(figsize=(16*0.7, 9*0.7))
    plt.plot(metrics_model_1.index, metrics_model_1["balanced_accuracy"].values,
             color="#327887", ls="-", label="Model 1 - Balanced Accuracy")
    plt.plot(metrics_model_1.index, metrics_model_1["precision"].values,
             color="#327887", ls=":", label="Model 1 - Precision")
    plt.plot(metrics_model_1.index, metrics_model_1["recall"].values,
             color="#327887", ls="--", label="Model 1 - Recall")
    plt.plot(metrics_model_2.index, metrics_model_2["balanced_accuracy"].values,
             color="#dfc586", ls="-", label="Model 2 - Balanced Accuracy")
    plt.plot(metrics_model_2.index, metrics_model_2["precision"].values,
             color="#dfc586", ls=":", label="Model 2 - Precision")
    plt.plot(metrics_model_2.index, metrics_model_2["recall"].values,
             color="#dfc586", ls="--", label="Model 2 - Recall")
    plt.legend()
    plt.xlabel("Threshold")
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.ylabel("Score")
    return fig


if __name__ == '__main__':
    monthly_outcome = pd.read_csv("data/clean/monthly_outcome.csv", index_col=0).astype(MONTHLY_OUTCOME_DTYPES)
    application = pd.read_csv("data/processed/application_with_is_bad_3m.csv", index_col=0).astype(APPLICATION_DTYPES)

    # filter development
    monthly_outcome_dev = monthly_outcome[monthly_outcome.date < np.datetime64("2019-08-01")]
    application_dev = application.loc[monthly_outcome_dev.index.unique()]

    y_true_dev = application_dev.is_bad_12m.astype(int)
    score_1_dev = application_dev.model_1
    score_2_dev = application_dev.model_2

    accuracies_dev_1 = balanced_accuracy_threshold(y_true_dev, score_1_dev)
    accuracies_dev_2 = balanced_accuracy_threshold(y_true_dev, score_2_dev)
    compare_accuracy_from_two_models(accuracies_dev_1, accuracies_dev_2)
    plt.title("Development")
    plt.savefig("data/reports/metrics_comparison_dev.png", dpi=180)
    plt.close()

    # filter out-of-time
    monthly_outcome_out_of_time = monthly_outcome[monthly_outcome.date.between(np.datetime64("2019-08-01"),
                                                                               np.datetime64("2020-01-01"))]
    application_out_of_time = application.loc[monthly_outcome_out_of_time.index.unique()]
    y_true_out_of_time = application_out_of_time.is_bad_12m.fillna(0).astype(int)
    score_1_out_of_time = application_out_of_time.model_1
    score_2_out_of_time = application_out_of_time.model_2

    accuracies_out_of_time_1 = balanced_accuracy_threshold(y_true_out_of_time, score_1_out_of_time)
    accuracies_out_of_time_2 = balanced_accuracy_threshold(y_true_out_of_time, score_2_out_of_time)
    compare_accuracy_from_two_models(accuracies_out_of_time_1, accuracies_out_of_time_2)
    plt.title("Out-Of-Time")
    plt.savefig("data/reports/metrics_comparison_out_of_time.png", dpi=180)
    plt.close()

    # filter post-deployment
    monthly_outcome_post_deployment = monthly_outcome[monthly_outcome.date.gt(np.datetime64("2020-01-01"))]
    application_post_deployment = application.loc[monthly_outcome_post_deployment.index.unique()]
    y_true_post_deployment = application_post_deployment.is_bad_12m.fillna(0).astype(int)
    score_1_post_deployment = application_post_deployment.model_1
    score_2_post_deployment = application_post_deployment.model_2

    accuracies_post_deployment_1 = balanced_accuracy_threshold(y_true_post_deployment, score_1_post_deployment)
    accuracies_post_deployment_2 = balanced_accuracy_threshold(y_true_post_deployment, score_2_post_deployment)
    compare_accuracy_from_two_models(accuracies_post_deployment_1, accuracies_post_deployment_2)
    plt.title("Post-Deployment")
    plt.savefig("data/reports/metrics_comparison_post_deployment.png", dpi=180)
    plt.close()
