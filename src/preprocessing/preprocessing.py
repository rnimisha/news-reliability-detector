import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, isnan, rand, when

nltk.download("stopwords", quiet=True)
ps = PorterStemmer()


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


def remove_stop_words(text: str) -> str:
    words = text.split()
    stopwords_eng = stopwords.words("english")

    without_stopwords = [x for x in words if x not in (stopwords_eng)]
    return " ".join(without_stopwords)


def stem_words(text: str) -> str:
    words = text.split()
    stemmed = [ps.stem(word) for word in words]
    return " ".join(stemmed)
