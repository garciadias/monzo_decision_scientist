"""
## Task 2.4: calibration

We also want to monitor the early risk indicator *is_bad_3m* in case something really bad happens (e.g. a pandemic).
For that we need to calibrate our scores to the probability of such short-term outcome. Could you please create the
calibrated scores for the two models and validate them? (Hint: if this is not a topic you are familiar with,
scikit-learn has some handy utilities)
"""
from sklearn.calibration import calibration_curve
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monzo_decision_scientist.definitions import APPLICATION_DTYPES, MONTHLY_OUTCOME_DTYPES, COLORS

if __name__ == '__main__':
    monthly_outcome = pd.read_csv("data/clean/monthly_outcome.csv", index_col=0).astype(MONTHLY_OUTCOME_DTYPES)
    application = pd.read_csv("data/processed/application_with_is_bad_3m.csv", index_col=0).astype(APPLICATION_DTYPES)
    BLUE, GREEN, YELLOW, RED = COLORS
    SEED = 24042020

    # filter development
    monthly_outcome_dev = monthly_outcome[monthly_outcome.date < np.datetime64("2019-08-01")]
    application_dev = application.loc[monthly_outcome_dev.index.unique()]
    y_true_dev_12m = application_dev.is_bad_12m.astype(int)
    y_true_dev_3m = application_dev.is_bad_3m.astype(int)
    model_1_score_dev = application_dev.model_1

    # filter out-of-time
    monthly_outcome_out_of_time = monthly_outcome[monthly_outcome.date.between(np.datetime64("2019-08-01"),
                                                                               np.datetime64("2020-01-01"))]
    application_out_of_time = application.loc[monthly_outcome_out_of_time.index.unique()]
    y_true_out_of_time_12m = application_out_of_time.is_bad_12m.fillna(0).astype(int)
    y_true_out_of_time_3m = application_out_of_time.is_bad_3m.fillna(0).astype(int)
    model_1_score_out_of_time = application_out_of_time.model_1

    clf = LogisticRegression(random_state=SEED)
    clf.fit(model_1_score_dev.values[:, None], y_true_dev_3m)
    calibrated_model_1 = clf.predict_proba(model_1_score_out_of_time.values[:, None])[:, 1]
    application_out_of_time.loc[:, ("calibrated_model_1")] = calibrated_model_1

    # reliability diagram
    fig, ax = plt.subplots(figsize=(16*0.6, 9*0.6))
    fop, mpv = calibration_curve(y_true_out_of_time_3m, model_1_score_out_of_time)
    fop_calib, mpv_calib = calibration_curve(y_true_out_of_time_3m, calibrated_model_1)
    plt.plot([0, 0.4], [0, 0.4], linestyle='--', color=BLUE, label="Perfect Calibration")
    plt.plot(mpv, fop, marker='.', color=RED, label="Model 1")
    plt.plot(mpv_calib, fop_calib, marker='.', color=GREEN, label="Model 1 Calibrated")
    plt.legend()
    ax.set(xlabel="Mean predicted probability", ylabel="Count")
    plt.savefig("data/reports/reliability_diagram_is_bad_3m_calibration.png", dpi=180)
    plt.close()

    threshold = 0.02
    y_pred_out_of_time = model_1_score_out_of_time > threshold
    y_pred_post_deployment = calibrated_model_1 > threshold
    balanced_accuracy_out_of_time = balanced_accuracy_score(y_true_out_of_time_3m, y_pred_out_of_time)
    precision_out_of_time = precision_score(y_true_out_of_time_3m, y_pred_out_of_time)
    recall_out_of_time = recall_score(y_true_out_of_time_3m, y_pred_out_of_time)

    balanced_accuracy_post_deployment = balanced_accuracy_score(y_true_out_of_time_3m, y_pred_post_deployment)
    precision_post_deployment = precision_score(y_true_out_of_time_3m, y_pred_post_deployment)
    recall_post_deployment = recall_score(y_true_out_of_time_3m, y_pred_post_deployment)

    RESULTS = pd.DataFrame([
        [balanced_accuracy_out_of_time, precision_out_of_time, recall_out_of_time],
        [balanced_accuracy_post_deployment, precision_post_deployment, recall_post_deployment],
    ],
        index=["3m Model 1", "3m Model 1 Calibrated"],
        columns=["Balanced Accuracy", "Precisions", "Recall"])
    fig, ax = plt.subplots(figsize=(16*0.7, 9*0.7))
    RESULTS.plot.bar(color=[GREEN, YELLOW, RED], ax=ax)
    # plt.setp(ax.get_xticklabels(), rotation=0)
    plt.tick_params(labelsize=18, rotation=0)
    ax.minorticks_on()
    ax.grid()
    plt.tight_layout()
    plt.savefig("data/reports/metrics_comparison_before_and_after_calibration.png", dpi=180)
    plt.close()
