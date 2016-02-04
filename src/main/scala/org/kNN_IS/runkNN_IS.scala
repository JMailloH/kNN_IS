package org.KnnMR

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.log4j.Logger
import org.KnnMR.Utils.KeelParser
import java.util.StringTokenizer
import scala.collection.mutable.ListBuffer
import keel.Algorithms.Lazy_Learning.LazyAlgorithm
import keel.Algorithms.Lazy_Learning.KNN
import keel.Algorithms.Lazy_Learning.KNN.KNN
import java.io._
import org.apache.spark.broadcast.Broadcast
import org.apache.hadoop.fs.Path
import org.apache.hadoop.fs.FSDataInputStream
import java.util.Scanner
import org.apache.hadoop.conf.Configuration
import org.apache.spark.rdd.PairRDDFunctions
import org.apache.spark.rdd.OrderedRDDFunctions
import org.apache.spark.Partitioner;
import org.apache.spark.RangePartitioner

object KnnMR extends Serializable {

  private def broadcastTest(test: Array[Array[Double]], context: SparkContext) = context.broadcast(test)

  var sc: SparkContext = null

  def main(arg: Array[String]) {

    var logger = Logger.getLogger(this.getClass())

    if (arg.length < 8) {
      logger.error("=> wrong parameters number")
      System.err.println("Parameters \n\t<path-to-header>\n\t<path-to-train>\n\t<path-to-test>\n\t<number-of-neighbors>\n\t<number-of-partition>\n\t<number-of-reducers-task>\n\t<path-to-output>\n\t<maximum-weight-per-task>")
      System.exit(1)
    }

    //Taking the initial time.
    val timeBeg = System.nanoTime

    //Reading parameters
    val pathHeader = arg(0)
    val pathTrain = arg(1)
    val pathTest = arg(2)
    val K = arg(3).toInt
    val distanceType = 2
    val numPartition = arg(4).toInt
    val numReduces = arg(5).toInt
    val numReducePartitions = arg(6).toInt //if == -1, auto-setting
    val pathOutput = arg(7)
    var maxWeight = 0.0
    if (numReducePartitions == -1) {
      maxWeight = arg(8).toDouble
    }

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
    logger.info("=> NumberMapPartition \"" + numPartition + "\"")
    logger.info("=> NumberReducePartition \"" + numReduces + "\"")
    logger.info("=> DistanceType \"" + distanceType + "\"")
    logger.info("=> pathToOuput \"" + pathOutput + "\"")

    //Reading header of the dataset and dataset
    val converter = new KeelParser(sc, pathHeader)
    val trainRaw = sc.textFile(pathTrain: String, numPartition)
    val train = trainRaw.map(line => (converter.parserToDouble(line))).cache

    val testRaw = sc.textFile(pathTest: String).zipWithIndex().map(line => (line._2.toInt, converter.parserToDouble(line._1))).sortByKey().cache //Obligo a una particion para asegurar la linealidad de los datos.

    //Count the samples of each data set and the number of classes
    val numSamplesTrain = train.count()
    val numSamplesTest = testRaw.count()
    val numClass = converter.getNumClassFromHeader()
    val numFeatures = converter.getNumFeaturesFromHeader()

    //Setting Iterative MapReduce
    var inc = 0
    var subdel = 0
    var topdel = 0

    var weightTrain = 0.0
    var weightTest = 0.0

    var numIterations = 0

    if (numReducePartitions == -1) { //Auto-setting the minimum partition test.
      weightTrain = (8 * numSamplesTrain * numFeatures) / (numPartition * 1024.0 * 1024.0)
      weightTest = (8 * numSamplesTest * numFeatures) / (1024.0 * 1024.0)
      println("\n\nPeso total: " + (weightTrain + weightTest) + "  |  Peso training: " + weightTrain + "  |  Limite por task: " + (maxWeight * 1024.0))
      if (weightTrain + weightTest < maxWeight * 1024.0) { //It can be run with one iteration
        numIterations = 1
      } else {
        if (weightTrain >= maxWeight * 1024.0) {
          println("Train wight bigger than lim-task. Abort")
          System.exit(0)
        }
        numIterations = (1 + (weightTest / ((maxWeight * 1024.0) - weightTrain)).toInt)
      }

    } else {
      numIterations = numReducePartitions
    }

    logger.info("=> NumberIterations \"" + numIterations + "\"")

    inc = (numSamplesTest / numIterations).toInt
    subdel = 0
    topdel = inc
    if (numIterations == 1) { //If only one partition
      topdel = numSamplesTest.toInt + 1
    }

    var rightPredictedClasses: Array[Array[Array[Int]]] = new Array[Array[Array[Int]]](numIterations)
    var mapTimesArray: Array[Double] = new Array[Double](numIterations)
    var reduceTimesArray: Array[Double] = new Array[Double](numIterations)
    var iterativeTimesArray: Array[Double] = new Array[Double](numIterations)

    var test: Broadcast[Array[Array[Double]]] = null

    val rightClasses = testRaw.map(line => line._2(line._2.length - 1)).collect

    for (i <- 0 to numIterations - 1) {

      //Taking the iterative initial time.
      val timeBegIterative = System.nanoTime

      if (i == numIterations - 1) {
        //test = broadcastTest(testRaw.filter { case (key, value) => key >= subdel }.map(line => line._2).collect, sc)
        test = broadcastTest(testRaw.filterByRange(subdel, topdel * 2).map(line => line._2).collect, sc)
        print("\nNumber of samples on each partition ( the last one ) " + test.value.length + "\n\n")

      } else {
        //test = broadcastTest(testRaw.filter { case (key, value) => key >= subdel && key < topdel }.map(line => line._2).collect, sc)
        test = broadcastTest(testRaw.filterByRange(subdel, topdel).map(line => line._2).collect, sc)
        print("\nNumber of samples on each partition " + test.value.length + "\n\n")
      }

      println("\nTime of filterByRange and .collect: " + (System.nanoTime - timeBegIterative) / 1e9 + "****\n\n")

      //Taking the map initial time.
      val timeBegMap = System.nanoTime
      //Calling KNN (Map Phase)    
      var resultKNNPartitioned = train.mapPartitions(line => MR_KNN.knn(line, test, numClass, K, distanceType, subdel)).cache //.collect
      resultKNNPartitioned.count

      var mapTimes = resultKNNPartitioned.filter(line => line._1 == -1).map(line => MR_KNN.getMapTimes(line)).collect

      var maxTime = mapTimes(0)

      for (i <- 1 to mapTimes.length - 1) {
        //if(mapTimes(i) > maxTime){
        maxTime = maxTime + mapTimes(i)
        //}
      }

      mapTimesArray(i) = maxTime/mapTimes.length

      //Taking the reduce initial time.
      val timeBegRed = System.nanoTime
      //Reduce phase
      var result = resultKNNPartitioned.reduceByKey(MR_KNN.combine(_, _), numReduces).filter(line => line._1 != -1).collect
      reduceTimesArray(i) = (System.nanoTime - timeBegRed) / 1e9

      rightPredictedClasses(i) = MR_KNN.calculateRightPredictedClasses(rightClasses, result, numClass)

      subdel = subdel + inc
      topdel = topdel + inc

      test.destroy

      iterativeTimesArray(i) = (System.nanoTime - timeBegIterative) / 1e9
    }

    //Taking the end time.
    val timeEnd = System.nanoTime

    //Write the Predictions file
    var writerPredictions = new ListBuffer[String]
    writerPredictions += "***Predictions.txt ==> 1th column predicted class; 2on column right class***"

    for (i <- 0 to numIterations - 1) {
      var size = rightPredictedClasses(i).length - 1
      for (j <- 0 to size) {
        writerPredictions += rightPredictedClasses(i)(j)(0).toString + "\t" + rightPredictedClasses(i)(j)(1).toString
      }
    }
    val predictionsTxt = sc.parallelize(writerPredictions, 1)
    predictionsTxt.saveAsTextFile(pathOutput + "/Predictions.txt")

    //Write the Result file
    val confusionMatrix = MR_KNN.calculateConfusionMatrix(rightPredictedClasses, numClass)
    var writerResult = new ListBuffer[String]
    writerResult += "***Results.txt ==> Contain: Confusion Matrix; Accuracy; Time of the run***\n"
    for (i <- 0 to numClass - 1) {
      var auxString = ""
      for (j <- 0 to numClass - 1) {
        auxString = auxString + confusionMatrix(i)(j).toString + "\t"
      }
      writerResult += auxString
    }

    val accuracy = MR_KNN.calculateAccuracy(confusionMatrix)
    val AUC = MR_KNN.calculateAUC(confusionMatrix)
    writerResult += "\n" + accuracy
    writerResult += "\n" + AUC
    writerResult += "\n" + (timeEnd - timeBeg) / 1e9 + " seconds\n"

    val resultTxt = sc.parallelize(writerResult, 1)
    resultTxt.saveAsTextFile(pathOutput + "/Result.txt")

    var writerTimes = new ListBuffer[String]
    writerTimes += "***Times.txt ==> Contain: run maps time; run reduce time; run clean up reduce time; Time of complete the run***"
    var sumTimesAux = 0.0
    for (i <- 0 to numIterations - 1) {
      sumTimesAux = sumTimesAux + mapTimesArray(i)
    }
    writerTimes += "\n@mapTime\n" + sumTimesAux / numIterations
    sumTimesAux = 0.0
    for (i <- 0 to numIterations - 1) {
      sumTimesAux = sumTimesAux + reduceTimesArray(i)
    }
    writerTimes += "\n@reduceTime\n" + sumTimesAux / numIterations
    sumTimesAux = 0.0
    for (i <- 0 to numIterations - 1) {
      sumTimesAux = sumTimesAux + iterativeTimesArray(i)
    }
    writerTimes += "\n@iterativeTime\n" + sumTimesAux / numIterations

    writerTimes += "\n@totalRunTime\n" + (timeEnd - timeBeg) / 1e9

    val timesTxt = sc.parallelize(writerTimes, 1)
    timesTxt.saveAsTextFile(pathOutput + "/Times.txt")
    
  }
}
