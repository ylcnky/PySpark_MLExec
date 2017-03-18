from __future__ import print_function
from pyspark.ml.feature import Word2Vec
from pyspark.sql import SparkSession

if __name__ == '__main__':
    spark = SparkSession\
        .builder\
        .appName('Word2Vec')\
        .getOrCreate()

    # Input data: Each row is a bag of words from a sentence or documents
    documentDF = spark.createDataFrame([
        ("Hi I heard about Spark".split(" "), ),
        ("I wish Java could use case classes".split(" "), ),
        ("Logistic regression models are neat".split(" "), )
    ], ["text"])

    #Learn a mapping from words to Vectors
    word2Vec = Word2Vec(vectorSize = 3, minCount=0, inputCol='text', outputCol='result')
    model = word2Vec.fit(documentDF)

    result = model.transform(documentDF)
    for row in result.collect():
        text, vector = row
        print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))

    spark.stop()
