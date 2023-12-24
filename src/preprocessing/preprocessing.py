from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, isnan, rand, when


def print_null_rows(data: DataFrame):
    data.select(
        [count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data.columns]
    ).show()


def drop_missing_text_rows(data: DataFrame) -> DataFrame:
    clean_data = data.na.drop(subset=["text"])
    return clean_data


def shuffle_rows(data: DataFrame) -> DataFrame:
    # generate row with random values
    shuffled = data.withColumn("rand", rand())
    shuffled = shuffled.orderBy("rand").drop("rand")
    return shuffled
