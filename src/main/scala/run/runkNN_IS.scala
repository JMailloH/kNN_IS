package org.apache.spark.run

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.classification.kNN_IS.kNN_IS
import org.apache.log4j.Logger
import utils.keel.KeelParser
import scala.collection.mutable.ListBuffer

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
    val numClass = converter.getNumClassFromHeader()
    val numFeatures = converter.getNumFeaturesFromHeader()
    var timeBeg: Long = 0l
    var timeEnd: Long = 0l
    var predictions: RDD[(Double, Double)] = null

    timeBeg = System.nanoTime
    if (version == "mllib") {
      val train = sc.textFile(pathTrain: String, numPartitionMap).map(line => converter.parserToLabeledPoint(line)).persist
      val test = sc.textFile(pathTest: String).map(line => converter.parserToLabeledPoint(line)).persist

      val knn = kNN_IS.setup(train, test, K, distanceType, numClass, numFeatures, numPartitionMap, numReduces, numIterations, maxWeight)
      predictions = knn.predict(sc)

    } else {
      val sqlContext = new org.apache.spark.sql.SQLContext(sc)
      import sqlContext.implicits._
      val train = sc.textFile(pathTrain: String, numPartitionMap).map(line => converter.parserToLabeledPoint(line)).toDF().persist
      val test = sc.textFile(pathTest: String).map(line => converter.parserToLabeledPoint(line)).toDF().persist

      var outPathArray: Array[String] = new Array[String](1)
      outPathArray(0) = pathOutput
      val knn = new org.apache.spark.ml.classification.kNN_IS.kNN_ISClassifier()

      knn
        .setLabelCol("label")
        .setFeaturesCol("features")
        .setK(K)
        .setDistanceType(distanceType)
        .setNumClass(numClass)
        .setNumFeatures(numFeatures)
        .setNumPartitionMap(numPartitionMap)
        .setNumReduces(numReduces)
        .setNumIter(numIterations)
        .setMaxWeight(maxWeight)
        .setNumSamplesTest(test.count.toInt)
        .setOutPath(outPathArray)

      // Chain indexers and tree in a Pipeline
      val pipeline = new Pipeline().setStages(Array(knn))

      // Train model.  This also runs the indexers.
      val model = pipeline.fit(train)
      predictions = model.transform(test).map(line => (line.get(0).asInstanceOf[Number].doubleValue(), line.get(1).asInstanceOf[Number].doubleValue()))
    }
    predictions.count
    timeEnd = System.nanoTime

    val metrics = new MulticlassMetrics(predictions)
    val precision = metrics.precision
    val cm = metrics.confusionMatrix

    var writerReport = new ListBuffer[String]
    writerReport += "***Report.txt ==> Contain: Confusion Matrix; Precision; Total Runtime***\n"
    writerReport += "@ConfusionMatrix\n" + cm
    writerReport += "\n@Precision\n" + precision
    writerReport += "\n@TotalRuntime\n" + (timeEnd - timeBeg) / 1e9
    val Report = sc.parallelize(writerReport, 1)
    Report.saveAsTextFile(pathOutput + "/Report.txt")

  }
}
