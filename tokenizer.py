from __future__ import print_function
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession

if __name__ == '__main__':
    spark = SparkSession\
        .builder\
        .appName('Tokenizer')\
        .getOrCreate()

    sentenceDataFrame = spark.createDataFrame([
        (0, "Hi I heard about Spark"),
        (1, "I wish Java could use case classes"),
        (2, "Logistic,regression,models,are,neat")
    ], ["id", "sentence"])

    tokenizer = Tokenizer(inputCol = 'sentence', outputCol='words')
    regexTokenizer = RegexTokenizer(inputCol='sentence', outputCol='words', pattern = "\\W")
    countTokens = udf(lambda words: len(words), IntegerType())
    tokenized = tokenizer.transform(sentenceDataFrame)
    tokenized.select('sentence','words')\
        .withColumn('tokens', countTokens(col('words'))).show(truncate=False)

    regexTokenized = regexTokenizer.transform(sentenceDataFrame)
    regexTokenized.select('sentence', 'words') \
        .withColumn('tokens', countTokens(col('words'))).show(truncate=False)

    spark.stop
