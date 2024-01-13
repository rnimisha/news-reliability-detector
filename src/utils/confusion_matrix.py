import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def get_confusion_matrix(predictions, target_column):
    y_true = predictions.select(target_column).rdd.map(lambda x: x[0])
    y_pred = predictions.select("prediction").rdd.map(lambda x: x[0])

    labels = sorted(
        predictions.select(target_column).distinct().rdd.map(lambda x: x[0]).collect()
    )

    return confusion_matrix(y_true.collect(), y_pred.collect(), labels=labels)


def plot_confusion_matrix(ax, conf_matrix, classes, title):
    sns.set(font_scale=1.2)
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="g",
        cmap="PuBuGn",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
