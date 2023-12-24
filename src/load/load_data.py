import os

from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import lit

current_directory = os.path.abspath("")
os.chdir(current_directory)


def load_fake_data(spark: SparkSession) -> DataFrame:
    """Load fake data csv as add target column as 1

    Returns:
        DataFrame: fake dataframe with target
    """
    fake_data = spark.read.csv("data/raw/Fake.csv", header=True, inferSchema=True)
    fake_data = fake_data.withColumn("target", lit(1))  # true
    return fake_data


def load_true_data(spark: SparkSession) -> DataFrame:
    """Load true data csv as add target column as 0

    Returns:
        DataFrame: true dataframe with target
    """
    true_data = spark.read.csv("data/raw/True.csv")
    true_data = true_data.withColumn("target", lit(0))  # false
    return true_data


def load(spark: SparkSession) -> DataFrame:
    """Concatenates fake and true data into one dataframe

    Returns:
        DataFrame: concatenated dataframe
    """
    fake_data = load_fake_data(spark)
    true_data = load_true_data(spark)
    dataset = fake_data.union(true_data)
    return dataset
