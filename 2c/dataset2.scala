import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.optimization.{LBFGS, LogisticGradient, SquaredL2Updater}
// Load training data in LIBSVM format.
var trainsum=0.0;
var testsum=0.0;

val data = MLUtils.loadLibSVMFile(sc,"Milestone1.txt")
var x=0;
//For(x <-1 to 20 )
//{



// Split data into training (60%) and test (40%).
val splits = data.randomSplit(Array(0.8, 0.2))
val training = splits(0).cache()
val test = splits(1).cache()

// Run training algorithm to build the model
val numIterations = 100
val model = SVMWithSGD.train(training, numIterations)
//val svmAlg = new SVMWithSGD()
//svmAlg.optimizer.setStepSize(0.5).setNumIterations(500).setRegParam(0.11).setMiniBatchFraction(0.7).setUpdater(new L1Updater())
//val model = svmAlg.run(training)
model.setThreshold(0.5)
//modelL1.clearThreshold()
//modelL1.clearThreshold()

// Compute raw scores on the test set.
//val modelL1 = sc.broadcast(model)
// Clear the default threshold.
//model.clearThreshold()
//model.setThreshold(0.5)

//val scModel = sc.broadcast(model)

// Compute raw scores on the test set.
val scoreAndLabelstrain = training.map { point =>
  val score = model.predict(point.features)
  (score, point.label)
}

val scoreAndLabelstest = test.map { point =>
  val score = model.predict(point.features)
  (score, point.label)
}



var trainingError = scoreAndLabelstrain.filter(r => r._1 == r._2).count.toDouble / training.count
printf("Prediction reliability on training data = %.2f%%\n", (100*trainingError))
//trainsum= trainsum + 100*trainingError
println("\n");
var testingError = scoreAndLabelstest.filter(r => r._1 == r._2).count.toDouble / test.count
printf("Prediction reliability on testing data = %.2f%%\n", (100*testingError))
//testsum= testsum + 100*testingError
println("\n");

//val MSE = scoreAndLabelstest.map{ case(v, p) => math.pow((v - p), 2)}.mean()
//printf("training Mean Squared Error for testing data: %f\n",MSE)
//}
//printf("Prediction reliability on training data = %.2f%%\n", (trainsum/20))
//println("\n");
//printf("Prediction reliability on testing data = %.2f%%\n", (testsum/20))
//println("\n");
