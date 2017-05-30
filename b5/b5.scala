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
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

val df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").option("mode", "DROPMALFORMED").csv("b5.csv")



val tokenizer = new RegexTokenizer().setInputCol("foodtype").setOutputCol("words")
val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")
val hashingTF = new HashingTF().setInputCol("filtered").setOutputCol("features").setNumFeatures(1000)
val idf = new IDF().setInputCol("features").setOutputCol("foodfeatures")

val pipeline = new Pipeline().setStages(Array(tokenizer, remover, hashingTF, idf))

val model = pipeline.fit(df)

val pipedata = model.transform(df)

val data = pipedata.select('status,'exercisetype,'foodfeatures,'text,'Age,'Sex,'GoodAppetite,'EarCleaning,'NailCutting,'AttendsClasses,'BehavesWellClass,'AttendsHomeSwitches)



val tokenizer1 = new RegexTokenizer().setInputCol("exercisetype").setOutputCol("words")
val remover1 = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")
val hashingTF1 = new HashingTF().setInputCol("filtered").setOutputCol("features").setNumFeatures(1000)
val idf1 = new IDF().setInputCol("features").setOutputCol("exercisefeatures")

val pipeline1 = new Pipeline().setStages(Array(tokenizer1, remover1, hashingTF1, idf1))

val model1 = pipeline1.fit(data)

val pipedata1 = model1.transform(data)

val data1 = pipedata1.select('status,'exercisefeatures,'foodfeatures,'text,'Age,'Sex,'GoodAppetite,'EarCleaning,'NailCutting,'AttendsClasses,'BehavesWellClass,'AttendsHomeSwitches)


val tokenizer2 = new RegexTokenizer().setInputCol("text").setOutputCol("words")
val remover2 = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")
val hashingTF2 = new HashingTF().setInputCol("filtered").setOutputCol("features").setNumFeatures(1000)
val idf2 = new IDF().setInputCol("features").setOutputCol("textfeatures")

val pipeline2 = new Pipeline().setStages(Array(tokenizer2, remover2, hashingTF2, idf2))

val model2 = pipeline2.fit(data1)

val pipedata2 = model2.transform(data1)

val data2 = pipedata2.select('status,'exercisefeatures,'foodfeatures,'textfeatures,'Age,'Sex,'GoodAppetite,'EarCleaning,'NailCutting,'AttendsClasses,'BehavesWellClass,'AttendsHomeSwitches)




val assembler = new VectorAssembler().setInputCols(Array("exercisefeatures","foodfeatures","textfeatures","Age","Sex","GoodAppetite","EarCleaning","NailCutting","AttendsClasses","BehavesWellClass","AttendsHomeSwitches")).setOutputCol("features")

val output = assembler.transform(data2)

val labeled = output.rdd.map(row => LabeledPoint(row.getAs[Double]("status"),org.apache.spark.mllib.linalg.Vectors.fromML(row.getAs[org.apache.spark.ml.linalg.SparseVector]("features"))))

val splits = labeled.randomSplit(Array(0.8, 0.2),seed=11l)
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

