from pyspark.ml import Pipeline
from pyspark.ml.feature import IDF, HashingTF, Tokenizer, VectorAssembler
from pyspark.sql import DataFrame, SparkSession


def load_processed_data(spark: SparkSession) -> DataFrame:
    data = spark.read.csv("data/processed/processed.csv", header=True, inferSchema=True)
    return data


def create_pipeline() -> Pipeline:
    tokenizer = Tokenizer(inputCol="content", outputCol="tokenized")
    hashingTf = HashingTF(inputCol="tokenized", outputCol="tfhashed", numFeatures=100)
    idf = IDF(inputCol="tfhashed", outputCol="features")
    feature_assembler = VectorAssembler(
        inputCols=["features"], outputCol="features_vector"
    )

    pipeline = Pipeline(stages=[tokenizer, hashingTf, idf, feature_assembler])

    return pipeline


def tokenize_data(spark: SparkSession) -> (DataFrame, DataFrame):
    pipeline = create_pipeline()
    data = load_processed_data(spark)

    # split data
    train_data, test_data = data.randomSplit(weights=[0.8, 0.2], seed=100)

    pipeline_model = pipeline.fit(train_data)

    train_transformed = pipeline_model.transform(train_data)
    test_transformed = pipeline_model.transform(test_data)

    return train_transformed, test_transformed
