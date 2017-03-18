from __future__ import print_function
from pyspark.ml.feature import NGram
from pyspark.sql import SparkSession

if __name__ == '__main__':
    spark = SparkSession\
        .builder\
        .appName('NGram')\
        .getOrCreate()

wordDataFrame = spark.createDataFrame([
        (0, ["Hi", "I", "heard", "about", "Spark"]),
        (1, ["I", "wish", "Java", "could", "use", "case", "classes"]),
        (2, ["Logistic", "regression", "models", "are", "neat"])
    ], ["id", "words"])

ngram = NGram(n=2, inputCol='words', outputCol='ngrams')
ngramDataFrame = ngram.transform(wordDataFrame)
ngramDataFrame.select('ngrams').show(truncate=False)

spark.stop()
