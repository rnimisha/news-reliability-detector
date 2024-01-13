from pyspark.sql import DataFrame
from sklearn.metrics import auc, roc_curve


def plot_roc_curve(ax, predictions: DataFrame, label_col, title):
    fpr, tpr, _ = roc_curve(
        predictions.select(label_col).collect(),
        predictions.select("probability").rdd.map(lambda row: row[0][1]).collect(),
    )
    roc_auc = auc(fpr, tpr)

    ax.plot(
        fpr,
        tpr,
        color="cadetblue",
        lw=2,
        label=f"ROC curve (area = {roc_auc:.2f})",
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{title} ROC Curve")
    ax.legend(loc="lower right")
