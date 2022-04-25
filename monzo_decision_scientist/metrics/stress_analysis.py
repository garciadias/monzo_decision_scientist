"""
## Task 3.3: new variable

You might have noticed that a new variable ***stress_score*** has become available since late 2019. Can you figure out
whether there is additional classification power from this variable over our models?

If so, how would you incorporate it into our decision model?
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score

from monzo_decision_scientist.definitions import APPLICATION_DTYPES


if __name__ == '__main__':
    application = pd.read_csv("data/processed/application_with_is_bad_3m.csv", index_col=0).astype(APPLICATION_DTYPES)

    # See if there are potential on stress_score
    application_stress = application.dropna(subset=["stress_score"]).copy()
    application_stress.loc[:, ("stress_score")] = application_stress.stress_score.astype(float)
    application_stress.loc[:, ("is_bad_12m")] = application_stress.is_bad_12m.fillna(0).astype(int)
    sns.kdeplot(data=application_stress, x="stress_score", hue="is_bad_12m",
                common_norm=False, palette=["#8abb9c",  "#e83860"])
    application_stress[["stress_score", "is_bad_12m", "model_1"]].corr()
    plt.tight_layout()
    plt.savefig("data/reports/kdeplot_stress_score.png", dpi=180)
    plt.close()

    sns.heatmap(application_stress[["stress_score", "is_bad_12m", "model_1"]].corr(),
                cmap="Reds_r", annot=True, square=True)
    plt.savefig("data/reports/correlation_stress.png", dpi=180)
    plt.close()

    # implement simple ideas for using the stress_score
    y_true = application_stress["is_bad_12m"]
    score_1_model_1 = application_stress["model_1"]
    score_1_model_3 = MinMaxScaler().fit_transform(application_stress[["stress_score", "model_1"]]).mean(axis=1)
    application_stress.loc[:, ("model_3")] = score_1_model_3

    fig, ax = plt.subplots()
    ax.vlines(0.30, 0, 4.2, label="Threshold = 0.30", color="#327887")
    plt.text(0.32, 4.0, "Threshold = 0.30", color="#327887")
    ax2 = sns.kdeplot(data=application_stress, x="model_3", hue="is_bad_12m",
                      common_norm=False, palette=["#8abb9c",  "#e83860"], ax=ax)
    plt.tight_layout()
    plt.savefig("data/reports/kdeplot_model_based_on_stress_score.png", dpi=180)
    plt.close()

    threshold_model_1 = 0.05
    threshold_model_2 = 0.30
    y_pred_model_1 = score_1_model_1 > threshold_model_1
    y_pred_model_3 = score_1_model_3 > threshold_model_2
    balanced_accuracy_model_1 = balanced_accuracy_score(y_true, y_pred_model_1)
    precision_model_1 = precision_score(y_true, y_pred_model_1)
    recall_model_1 = recall_score(y_true, y_pred_model_1)

    balanced_accuracy_model_3 = balanced_accuracy_score(y_true, y_pred_model_3)
    precision_model_3 = precision_score(y_true, y_pred_model_3)
    recall_model_3 = recall_score(y_true, y_pred_model_3)

    RESULTS = pd.DataFrame([
        [balanced_accuracy_model_1, precision_model_1, recall_model_1],
        [balanced_accuracy_model_3, precision_model_3, recall_model_3],
    ],
        index=["Model 1", "Model 3"],
        columns=["Balanced Accuracy", "Precisions", "Recall"])
