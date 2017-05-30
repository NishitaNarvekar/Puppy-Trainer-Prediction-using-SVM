import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import spark.implicits._
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils


//import csv
val df = spark.read.format("csv").option("header", "true").option("mode", "DROPMALFORMED").csv("3b.csv")

//the tokenizer is used to break the sentences to individual words
val tokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("words")

//the stop words remover is used to remove the stop words as they are not relevant to our problem
val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")

//calculating the term frequency
val hashingTF = new HashingTF().setInputCol("filtered").setOutputCol("features").setNumFeatures(1000)

//calculating the inverse document frequency
val idf = new IDF().setInputCol("features").setOutputCol("features1")

val pipeline = new Pipeline().setStages(Array(tokenizer, remover, hashingTF, idf))

val model = pipeline.fit(df)

val pipedata = model.transform(df)

val data = pipedata.select('status, 'features1)

data.show()



