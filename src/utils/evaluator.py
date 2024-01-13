from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def evaluate_model(predictions, target_column):
    evaluator = MulticlassClassificationEvaluator(
        labelCol=target_column, predictionCol="prediction"
    )

    precision = evaluator.evaluate(
        predictions, {evaluator.metricName: "weightedPrecision"}
    )
    recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})

    return precision, recall, accuracy
