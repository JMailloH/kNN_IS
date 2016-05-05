package org.apache.spark.mllib.classification.kNN_IS

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import scala.collection.mutable.ListBuffer
import org.apache.log4j.Logger
import keel.Algorithms.Lazy_Learning.LazyAlgorithm
import keel.Algorithms.Lazy_Learning.KNN
import keel.Algorithms.Lazy_Learning.KNN.KNN
import utils.keel.KeelParser

/**
 * @author Jesus Maillo
 */

class kNN_IS (train: RDD[LabeledPoint], test: RDD[LabeledPoint], k: Int, distanceType: Int, converter: KeelParser, numPartitionMap: Int, numReduces: Int, numIterations: Int, maxWeight: Double) extends Serializable {

  //Count the samples of each data set and the number of classes
  private val numSamplesTrain = train.count()
  private val numSamplesTest = test.count()
  private val numClass = converter.getNumClassFromHeader()
  private val numFeatures = converter.getNumFeaturesFromHeader()
  private val rightClasses = test.map(line => line.label).collect

  //Setting Iterative MapReduce
  private var inc = 0
  private var subdel = 0
  private var topdel = 0
  private var numIter = numIterations
  private def broadcastTest(test: Array[LabeledPoint], context: SparkContext) = context.broadcast(test)

  //Time variables
  private val timeBeg = System.nanoTime
  private var timeEnd: Long = 0
  private var mapTimesArray: Array[Double] = null
  private var reduceTimesArray: Array[Double] = null
  private var iterativeTimesArray: Array[Double] = null

  def getTrain: RDD[LabeledPoint] = train
  def getTest: RDD[LabeledPoint] = test
  def getK: Int = k
  def getDistanceType: Int = distanceType
  def getConverter: KeelParser = converter
  def getNumPartitionMap: Int = numPartitionMap
  def getNumReduces: Int = numReduces
  def getMaxWeight: Double = maxWeight
  def getNumIterations: Int = numIterations

  //Output
  private var rightPredictedClasses: Array[Array[Array[Int]]] = null

  def getRightPredictedClasses(): Array[Array[Array[Int]]] = {
    rightPredictedClasses
  }

  def getMapTimes(): Array[Double] = {
    mapTimesArray
  }

  def getReduceTimes(): Array[Double] = {
    reduceTimesArray
  }

  def getIterativeTimes(): Array[Double] = {
    iterativeTimesArray
  }

  //private var testBroadcast: Broadcast[Array[Array[Double]]] = null

  /**
   * Initial setting necessary.
   */
  def setup(): kNN_IS = {

    //Starting logger
    var logger = Logger.getLogger(this.getClass())

    //Setting Iterative MapReduce
    var weightTrain = 0.0
    var weightTest = 0.0

    numIter = 0
    if (numIterations == -1) { //Auto-setting the minimum partition test.
      weightTrain = (8 * numSamplesTrain * numFeatures) / (numPartitionMap * 1024.0 * 1024.0)
      weightTest = (8 * numSamplesTest * numFeatures) / (1024.0 * 1024.0)
      if (weightTrain + weightTest < maxWeight * 1024.0) { //It can be run with one iteration
        numIter = 1
      } else {
        if (weightTrain >= maxWeight * 1024.0) {
          logger.error("=> Train wight bigger than lim-task. Abort")
          System.exit(1)
        }
        numIter = (1 + (weightTest / ((maxWeight * 1024.0) - weightTrain)).toInt)
      }

    } else {
      numIter = numIterations
    }

    logger.info("=> NumberIterations \"" + numIter + "\"")

    inc = (numSamplesTest / numIter).toInt
    subdel = 0
    topdel = inc
    if (numIterations == 1) { //If only one partition
      topdel = numSamplesTest.toInt + 1
    }

    mapTimesArray = new Array[Double](numIter)
    reduceTimesArray = new Array[Double](numIter)
    iterativeTimesArray = new Array[Double](numIter)
    rightPredictedClasses = new Array[Array[Array[Int]]](numIter)

    this

  }

  /**
   * Predict.
   */
  def predict(sc: SparkContext) {
    val testWithKey = test.zipWithIndex().map { line => (line._2.toInt, line._1)}.sortByKey().cache
    var logger = Logger.getLogger(this.getClass())
    var testBroadcast: Broadcast[Array[LabeledPoint]] = null

    for (i <- 0 to numIter - 1) {

      //Taking the iterative initial time.
      val timeBegIterative = System.nanoTime

      if (i == numIter - 1) {
        testBroadcast = broadcastTest(testWithKey.filterByRange(subdel, topdel * 2).map(line => line._2).collect, sc)
      } else {
        testBroadcast = broadcastTest(testWithKey.filterByRange(subdel, topdel).map(line => line._2).collect, sc)
      }


      //Taking the map initial time.
      val timeBegMap = System.nanoTime
      //Calling KNN (Map Phase)    
      var resultKNNPartitioned = train.mapPartitions(line => knn(line, testBroadcast, subdel)).cache //.collect
      resultKNNPartitioned.count

      var mapTimes = resultKNNPartitioned.filter(line => line._1 == -1).map(line => getMapTimes(line)).collect

      var maxTime = mapTimes(0)

      for (i <- 1 to mapTimes.length - 1) {
        //if(mapTimes(i) > maxTime){
        maxTime = maxTime + mapTimes(i)
        //}
      }

      mapTimesArray(i) = maxTime / mapTimes.length

      //Taking the reduce initial time.
      val timeBegRed = System.nanoTime
      //Reduce phase
      var result = resultKNNPartitioned.reduceByKey(combine(_, _), numReduces).filter(line => line._1 != -1).collect
      reduceTimesArray(i) = (System.nanoTime - timeBegRed) / 1e9

      rightPredictedClasses(i) = calculateRightPredictedClasses(rightClasses, result, numClass)

      subdel = subdel + inc
      topdel = topdel + inc

      testBroadcast.destroy
      iterativeTimesArray(i) = (System.nanoTime - timeBegIterative) / 1e9

      timeEnd = System.nanoTime
    }

  }

  /**
   * Write the results in HDFS
   *
   */
  def writeResults(sc: SparkContext, pathOutput: String) = {

    var logger = Logger.getLogger(this.getClass())

    //Write the Predictions file
    var writerPredictions = new ListBuffer[String]
    writerPredictions += "***Predictions.txt ==> 1th column predicted class; 2on column right class***"

    for (i <- 0 to numIter - 1) {
      var size = rightPredictedClasses(i).length - 1
      for (j <- 0 to size) {
        writerPredictions += rightPredictedClasses(i)(j)(0).toString + "\t" + rightPredictedClasses(i)(j)(1).toString
      }
    }
    val predictionsTxt = sc.parallelize(writerPredictions, 1)
    predictionsTxt.saveAsTextFile(pathOutput + "/Predictions.txt")

    //Write the Result file
    val confusionMatrix = calculateConfusionMatrix(rightPredictedClasses, numClass)
    var writerResult = new ListBuffer[String]
    writerResult += "***Results.txt ==> Contain: Confusion Matrix; Accuracy; Time of the run***\n"
    for (i <- 0 to numClass - 1) {
      var auxString = ""
      for (j <- 0 to numClass - 1) {
        auxString = auxString + confusionMatrix(i)(j).toString + "\t"
      }
      writerResult += auxString
    }

    val accuracy = calculateAccuracy(confusionMatrix)
    val AUC = calculateAUC(confusionMatrix)
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
    writerTimes += "\n@mapTime\n" + sumTimesAux / numIter
    sumTimesAux = 0.0
    for (i <- 0 to numIterations - 1) {
      sumTimesAux = sumTimesAux + reduceTimesArray(i)
    }
    writerTimes += "\n@reduceTime\n" + sumTimesAux / numIter
    sumTimesAux = 0.0
    for (i <- 0 to numIterations - 1) {
      sumTimesAux = sumTimesAux + iterativeTimesArray(i)
    }
    writerTimes += "\n@iterativeTime\n" + sumTimesAux / numIter

    writerTimes += "\n@totalRunTime\n" + (timeEnd - timeBeg) / 1e9

    val timesTxt = sc.parallelize(writerTimes, 1)
    timesTxt.saveAsTextFile(pathOutput + "/Times.txt")
  }

  /**
   * @brief Calculate the K nearest neighbor from the test set over the train set.
   *
   * @param iter Data that iterate the RDD of the train set
   * @param testSet The test set in a broadcasting
   * @param classes number of label for the objective class
   * @param numNeighbors Value of the K nearest neighbors to calculate
   * @param distanceType MANHATTAN = 1 ; EUCLIDEAN = 2 ; HVDM = 3
   */
  def knn[T](iter: Iterator[LabeledPoint], testSet: Broadcast[Array[LabeledPoint]], subdel: Int): Iterator[(Int, Array[Array[Float]])] = { //Si queremos un Pair RDD el iterador seria iter: Iterator[(Long, Array[Double])]
    var begTime = System.nanoTime();
    var auxTime = System.nanoTime();

    // Initialization
    var acu = new ListBuffer[Array[Double]]
    var trainClass = new ListBuffer[Int]
    var trainData = new ListBuffer[Array[Double]]

    //val len = testSet.value(0).length

    //Parser to right structures
    while (iter.hasNext) {
      val cur = iter.next
      trainData += cur.features.toArray
      trainClass += cur.label.toInt
    }
    auxTime = System.nanoTime();

    //Create object KNN with the necesary information
    var outKNN = new KNN(trainData.toArray, trainClass.toArray, numClass, k, distanceType)

    // Initialization with the first sample to get de data structure
    val isCategorical = converter.getIfCategorical()
    var result = outKNN.getNeighbors(testSet.value(0).features.toArray, isCategorical)

    //Calculate the K nearest neighbors for each instance of the test set
    for (i <- 1 to testSet.value.length - 1) {
      result.add(outKNN.getNeighbors(testSet.value(i).features.toArray, isCategorical).get(0))
    }

    //Change to static array and set the key
    val size = result.size()
    var res = new Array[(Int, Array[Array[Float]])](size + 1)

    //res(0) = new Array[Array[Array[Double]]](size+1)
    var auxSubDel = subdel
    for (i <- 0 to size - 1) {
      val aux = new Array[Array[Float]](k)
      //res(i) = (i, aux)
      for (j <- 0 to k - 1) {
        aux(j) = new Array[Float](2)
        aux(j)(0) = result.get(i).get(j).getFirst().toFloat
        aux(j)(1) = result.get(i).get(j).getSecond().toFloat

      }
      res(i) = (auxSubDel, aux)
      auxSubDel = auxSubDel + 1
    }
    auxTime = System.nanoTime();

    val aux = new Array[Array[Float]](k)
    for (j <- 0 to k - 1) {
      aux(j) = new Array[Float](2)
      aux(j)(0) = ((System.nanoTime - begTime) / 1e9).toFloat
      aux(j)(1) = -1
    }
    res(size) = (-1, aux)

    res.iterator

  }

  /**
   * @brief Join the result of the map
   *
   * @param mapOut1 A element of the RDD to join
   * @param mapOut2 Another element of the RDD to join
   */
  def combine(mapOut1: Array[Array[Float]], mapOut2: Array[Array[Float]]): Array[Array[Float]] = {

    //val timeBegRed = System.nanoTime

    //var acumEquals = 0.0
    val numNeighbors = mapOut1.length
    var itOut1 = 0

    for (j <- 0 to numNeighbors - 1) { //Loop for the k neighbors
      if (mapOut1(itOut1)(0) <= mapOut2(j)(0)) { // Update the matrix taking the k nearest neighbors
        mapOut2(j)(0) = mapOut1(itOut1)(0)
        mapOut2(j)(1) = mapOut1(itOut1)(1)
        itOut1 = itOut1 + 1
      }
    }

    //val timeEndRed = System.nanoTime

    mapOut2
  }

  /**
   * @brief Recolect the run map time include on the class - distance matrix
   *
   * @param mapOut line of the RDD. before need a filter where key = -1
   */
  def getMapTimes[T](mapOut: (Int, Array[Array[Float]])): Float = {
    val res = mapOut._2(0)(0)
    res
  }

  /**
   * @brief Calculate majority vote and do a matrix with predicted and right class.
   *
   * @param test Test set for get the right class
   * @param predictedNeigh Class - distance matrix with the K nearest neighbor. Calculate the predicted class
   * @param numClass Number of label for the objetive clas
   */
  def calculateRightPredictedClasses(test: Array[Double], predictedNeigh: Array[(Int, Array[Array[Float]])], numClass: Int): Array[Array[Int]] = {
    val size = predictedNeigh.length
    val numNeighbors = predictedNeigh(0)._2.length
    //val numFeatures = test.value.length
    var rightPredictedClasses = new Array[Array[Int]](size)

    var t = 0
    for (i <- 0 to size - 1) {
      //Auxiliaris variables for majority vote
      var auxClas = new Array[Int](numClass)
      for (x <- 0 to numClass - 1) {
        auxClas(x) = 0
      }

      var clas = 0
      var numVotes = 0
      for (j <- 0 to numNeighbors - 1) {
        auxClas(predictedNeigh(i)._2(j)(1).toInt) = auxClas(predictedNeigh(i)._2(j)(1).toInt) + 1
        if (auxClas(predictedNeigh(i)._2(j)(1).toInt) > numVotes) {
          clas = predictedNeigh(i)._2(j)(1).toInt
          numVotes = auxClas(predictedNeigh(i)._2(j)(1).toInt)
        }

      }
      rightPredictedClasses(t) = new Array[Int](2)
      rightPredictedClasses(t)(0) = test(predictedNeigh(i)._1).toInt
      rightPredictedClasses(t)(1) = clas
      t = t + 1
    }

    rightPredictedClasses
  }

  /**
   * @brief Calculate the confusion matrix
   *
   * @param rightPredictedClas Array of int with right and predicted class
   * @param numClass Number of label for the objective class
   */
  def calculateConfusionMatrix(rightPredictedClas: Array[Array[Array[Int]]], numClass: Int): Array[Array[Int]] = {
    //Create and initializate the confusion matrix
    var confusionMatrix = new Array[Array[Int]](numClass)
    for (i <- 0 to numClass - 1) {
      confusionMatrix(i) = new Array[Int](numClass)
      for (j <- 0 to numClass - 1) {
        confusionMatrix(i)(j) = 0
      }
    }

    val numPartitionReduces = rightPredictedClas.length
    for (i <- 0 to numPartitionReduces - 1) {
      var size = rightPredictedClas(i).length - 1
      for (j <- 0 to size) {
        confusionMatrix(rightPredictedClas(i)(j)(0))(rightPredictedClas(i)(j)(1)) = confusionMatrix(rightPredictedClas(i)(j)(0))(rightPredictedClas(i)(j)(1)) + 1

      }
    }

    confusionMatrix
  }

  /**
   * @brief Calculate the accuracy with confusion matrix
   *
   * @param confusionMatrix
   */
  def calculateAccuracy(confusionMatrix: Array[Array[Int]]): Double = {
    var right = 0.0
    var total = 0.0

    val size = confusionMatrix.length

    for (i <- 0 to size - 1) {
      for (j <- 0 to size - 1) {
        if (i == j) {
          right = right + confusionMatrix(i)(j)
        }
        total = total + confusionMatrix(i)(j)
      }
    }

    val accuracy = right / total
    accuracy
  }

  /**
   * @brief Calculate the AUC with confusion matrix
   *
   * @param confusionMatrix
   */
  def calculateAUC(confusionMatrix: Array[Array[Int]]): Double = {
    var positive = 0.0
    var negative = 0.0
    val size = confusionMatrix.length

    if (size == 2) {
      positive = confusionMatrix(0)(0) / confusionMatrix(1)(0)
      negative = confusionMatrix(1)(1) / confusionMatrix(0)(1)
    }

    val AUC = (positive + negative) / 2.0
    AUC
  }
}

object kNN_IS {
  /**
   * Initial setting necessary.
   *
   * @param train Data that iterate the RDD of the train set
   * @param test The test set in a broadcasting
   * @param k number of neighbors
   * @param distanceType MANHATTAN = 1 ; EUCLIDEAN = 2 ; HVDM = 3
   * @param converter Dataset's information read from the header
   * @param numPartitionMap Number of partition. Number of map tasks
   * @param numReduces Number of reduce tasks
   * @param numIterations Autosettins = -1. Number of split in the test set and number of iterations
   */
  def setup(train: RDD[LabeledPoint], test: RDD[LabeledPoint], k: Int, distanceType: Int, converter: KeelParser, numPartitionMap: Int, numReduces: Int, numIterations: Int, maxWeight: Double) = {
    new kNN_IS(train, test, k, distanceType, converter, numPartitionMap, numReduces, numIterations, maxWeight).setup()
  }
}