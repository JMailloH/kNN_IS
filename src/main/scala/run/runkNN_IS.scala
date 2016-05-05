package org.apache.spark.run

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.log4j.Logger
import utils.keel.KeelParser
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{ IndexToString, StringIndexer }
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.kNN_IS.kNN_IS
import java.util.Scanner
import org.apache.spark.sql.functions.{ concat_ws, col }
import org.apache.spark.sql.Row
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
import org.apache.spark.annotation.{ Experimental, Since }
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.ml.util.{ DefaultParamsReadable, MetadataUtils, Identifiable }
import org.apache.spark.ml.{ PipelineModel, PredictorParams }
import org.apache.spark.ml.param._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.Predictor
import org.apache.spark.mllib.classification.{ kNN_IS => OldkNN_IS }
import org.apache.spark.mllib.classification.{ kNN_IS => OldkNN_ISModel }
import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib.linalg.{ DenseVector, SparseVector, Vectors, Vector }
import scala.collection.mutable.ListBuffer
import org.apache.log4j.Logger
import keel.Algorithms.Lazy_Learning.LazyAlgorithm
import keel.Algorithms.Lazy_Learning.KNN
import keel.Algorithms.Lazy_Learning.KNN.KNN
import utils.keel.KeelParser
import org.apache.spark.mllib.feature.PCAModel

object runkNN_IS extends Serializable {

  var sc: SparkContext = null

  def main(arg: Array[String]) {

    var logger = Logger.getLogger(this.getClass())

    if (arg.length == 9) {
      logger.error("=> wrong parameters number")
      System.err.println("Parameters \n\t<path-to-header>\n\t<path-to-train>\n\t<path-to-test>\n\t<number-of-neighbors>\n\t<number-of-partition>\n\t<number-of-reducers-task>\n\t<path-to-output>\n\t<maximum-weight-per-task>")
      System.exit(1)
    }

    //Reading parameters
    val pathHeader = arg(0)
    val pathTrain = arg(1)
    val pathTest = arg(2)
    val K = arg(3).toInt
    val distanceType = 2
    val numPartitionMap = arg(4).toInt
    val numReduces = arg(5).toInt
    val numIterations = arg(6).toInt //if == -1, auto-setting
    val pathOutput = arg(7)
    var maxWeight = 0.0
    if (numIterations == -1) {
      maxWeight = arg(8).toDouble
    }
    val version = arg(9)

    //Clean pathOutput for set the jobName
    var outDisplay: String = pathOutput

    //Basic setup
    val jobName = "Maillo - KnnMR -> " + outDisplay + " K = " + K

    //Spark Configuration
    val conf = new SparkConf().setAppName(jobName)
    sc = new SparkContext(conf)

    logger.info("=> jobName \"" + jobName + "\"")
    logger.info("=> pathToHeader \"" + pathHeader + "\"")
    logger.info("=> pathToTrain \"" + pathTrain + "\"")
    logger.info("=> pathToTest \"" + pathTest + "\"")
    logger.info("=> NumberNeighbors \"" + K + "\"")
    logger.info("=> NumberMapPartition \"" + numPartitionMap + "\"")
    logger.info("=> NumberReducePartition \"" + numReduces + "\"")
    logger.info("=> DistanceType \"" + distanceType + "\"")
    logger.info("=> pathToOuput \"" + pathOutput + "\"")
    if (numIterations == -1) {
      logger.info("=> maxWeight \"" + maxWeight + "\"")
    }
    logger.info("=> version \"" + version + "\"")

    //Reading header of the dataset and dataset
    val converter = new KeelParser(sc, pathHeader)
    val trainRaw = sc.textFile(pathTrain: String, numPartitionMap)
    val train = trainRaw.map { line =>
      val parsed = converter.parserToDouble(line)
      val featureVector = Vectors.dense(parsed.init)
      val label = parsed.last
      LabeledPoint(label, featureVector)
    }.persist

    val test = sc.textFile(pathTest: String).map { line =>
      val parsed = converter.parserToDouble(line)
      val featureVector = Vectors.dense(parsed.init)
      val label = parsed.last
      LabeledPoint(label, featureVector)
    }.persist

    val numClass = converter.getNumClassFromHeader()
    val numFeatures = converter.getNumFeaturesFromHeader()

    if (version == "mllib") {
      val knn = kNN_IS.setup(train, test, K, distanceType, converter, numPartitionMap, numReduces, numIterations, maxWeight)
      knn.predict(sc)
      knn.writeResults(sc, pathOutput)
    } else {
      val sqlContext = new org.apache.spark.sql.SQLContext(train.context)
      import sqlContext.implicits._

      var outPathArray: Array[String] = new Array[String](1)
      outPathArray(0) = pathOutput
      val knn = new org.apache.spark.ml.classification.kNN_IS.kNN_ISClassifier()
        .setLabelCol("label")
        .setFeaturesCol("features")
        .setK(K)
        .setDistanceType(distanceType)
        .setConverter(converter)
        .setNumPartitionMap(numPartitionMap)
        .setNumReduces(numReduces)
        .setNumIter(numIterations)
        .setMaxWeight(maxWeight)
        .setNumSamplesTest(test.count.toInt)
        .setOutPath(outPathArray)

      // Chain indexers and tree in a Pipeline
      val pipeline = new Pipeline().setStages(Array(knn))

      // Train model.  This also runs the indexers.
      val model = pipeline.fit(train.toDF())
      val predictions = model.transform(test.toDF())
    }

  }
}
