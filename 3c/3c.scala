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
val df = spark.read.format("csv").option("header", "true").option("mode", "DROPMALFORMED").csv("3c.csv")

//the tokenizer is used to break the sentences to individual words
val tokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("words")

//the stop words remover is used to remove the stop words as they are not relevant to our problem
val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")

//calculating the term frequency
val hashingTF = new HashingTF().setInputCol("filtered").setOutputCol("features").setNumFeatures(1000)

//calculating the inverse document frequency
val idf = new IDF().setInputCol("features").setOutputCol("features1")

//inserting stages into pipeline
val pipeline = new Pipeline().setStages(Array(tokenizer, remover, hashingTF, idf))

val model = pipeline.fit(df)

val pipedata = model.transform(df)

val data = pipedata.select('status, 'features1)

//1100 stands for status
val data1 = data.toDF("1100","features")

val toDouble = udf[Double, String]( _.toDouble)
val dataint = data1.withColumn("1100", toDouble(data1("1100"))).select("1100", "features")

val labeled = dataint.rdd.map(row => LabeledPoint(row.getAs[Double]("1100"),org.apache.spark.mllib.linalg.Vectors.fromML(row.getAs[org.apache.spark.ml.linalg.SparseVector]("features"))))

val splits = labeled.randomSplit(Array(0.8, 0.2))
val training = splits(0).cache()
val test = splits(1)

val svm_model = new SVMWithSGD().run(training)

val scoreAndLabelstrain = training.map { point =>
  val score = svm_model.predict(point.features)
  (score, point.label)
}

val scoreAndLabelstest = test.map { point =>
  val score = svm_model.predict(point.features)
  (score, point.label)
}



var trainingError = scoreAndLabelstrain.filter(r => r._1 == r._2).count.toDouble / training.count
printf("Prediction reliability on training data = %.2f%%\n", (100*trainingError))

println("\n");
var testingError = scoreAndLabelstest.filter(r => r._1 == r._2).count.toDouble / test.count
printf("Prediction reliability on testing data = %.2f%%\n", (100*testingError))

println("\n");





