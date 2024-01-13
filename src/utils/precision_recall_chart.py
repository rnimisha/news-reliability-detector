import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


def plot_precision_recall_curve(ax, predictions, label_col, title):
    y_true = predictions.select(label_col).rdd.map(lambda x: x[0])
    y_scores = predictions.select("probability").rdd.map(lambda x: x[0][1])

    precision, recall, _ = precision_recall_curve(y_true.collect(), y_scores.collect())

    ax.plot(recall, precision, color="cadetblue", label="Precision-Recall Curve")
    ax.set_title(f"{title} Precision & Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.grid(True)
