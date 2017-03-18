from __future__ import print_function
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql import SparkSession

if __name__ == '__main__':
    spark = SparkSession\
        .builder\
        .appName('TF_IDF')\
        .getOrCreate()

    sentenceData = spark.createDataFrame([
        (0.0, "Hi I heard about Spark"),
        (0.0, "I wish Java could use case classes"),
        (1.0, "Logistic regression models are neat")
    ], ["label", "sentence"])

    tokenizer = Tokenizer(inputCol='sentence', outputCol='words')
    wordsData = tokenizer.transform(sentenceData)
    hashingTF = HashingTF(inputCol='words', outputCol='rawFeatures', numFeatures = 20)
    featurizedData = hashingTF.transform(wordsData)

    idf = IDF(inputCol='rawFeatures', outputCol='features')
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)
    rescaledData.select('label','features').show()

    spark.stop()
