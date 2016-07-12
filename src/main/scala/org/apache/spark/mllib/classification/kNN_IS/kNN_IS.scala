package org.apache.spark.mllib.classification.kNN_IS

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import scala.collection.mutable.ArrayBuffer
import org.apache.log4j.Logger
import utils.keel.KeelParser

/**
 * Distributed kNN class.
 *
 *
 * @author Jesus Maillo
 */

class kNN_IS(train: RDD[LabeledPoint], test: RDD[LabeledPoint], k: Int, distanceType: Int, numClass: Int, numFeatures: Int, numPartitionMap: Int, numReduces: Int, numIterations: Int, maxWeight: Double) extends Serializable {

  //Count the samples of each data set and the number of classes
  private val numSamplesTrain = train.count()
  private val numSamplesTest = test.count()

  //Setting Iterative MapReduce
  private var inc = 0
  private var subdel = 0
  private var topdel = 0
  private var numIter = numIterations
  private def broadcastTest(test: Array[LabeledPoint], context: SparkContext) = context.broadcast(test)

  //Getters
  def getTrain: RDD[LabeledPoint] = train
  def getTest: RDD[LabeledPoint] = test
  def getK: Int = k
  def getDistanceType: Int = distanceType
  def getNumClass: Int = numClass
  def getNumFeatures: Int = numFeatures
  def getNumPartitionMap: Int = numPartitionMap
  def getNumReduces: Int = numReduces
  def getMaxWeight: Double = maxWeight
  def getNumIterations: Int = numIterations

  /**
   * Initial setting necessary. Auto-set the number of iterations and load the data sets and parameters.
   *
   * @return Instance of this class. *this*
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

    this

  }

  /**
   * Predict. kNN
   *
   * @return RDD[(Double, Double)]. First column Predicted class. Second Column Right class.
   */
  def predict(): RDD[(Double, Double)] = {
    val testWithKey = test.zipWithIndex().map { line => (line._2.toInt, line._1) }.sortByKey().cache
    var logger = Logger.getLogger(this.getClass())
    var testBroadcast: Broadcast[Array[LabeledPoint]] = null
    var output: RDD[(Double, Double)] = null

    for (i <- 0 until numIter) {

      //Taking the iterative initial time.
      val timeBegIterative = System.nanoTime

      if (numIter == 1)
        testBroadcast = broadcastTest(test.collect, test.sparkContext)
      else
        testBroadcast = broadcastTest(testWithKey.filterByRange(subdel, topdel).map(line => line._2).collect, testWithKey.sparkContext)

      if (output == null) {
        output = testWithKey.join(train.mapPartitions(split => knn(split, testBroadcast, subdel)).reduceByKey(combine)).map(sample => calculatePredictedRightClassesFuzzy(sample)).cache
      } else {
        output = output.union(testWithKey.join(train.mapPartitions(split => knn(split, testBroadcast, subdel)).reduceByKey(combine)).map(sample => calculatePredictedRightClassesFuzzy(sample))).cache
      }
      output.count

      //Update the pairs of delimiters
      subdel = topdel + 1
      topdel = topdel + inc + 1
      testBroadcast.destroy
    }

    output

  }

  /**
   * Calculate the K nearest neighbor from the test set over the train set.
   *
   * @param iter Iterator of each split of the training set.
   * @param testSet The test set in a broadcasting
   * @param subdel Int needed for take order when iterative version is running
   * @return K Nearest Neighbors for this split
   */
  def knn[T](iter: Iterator[LabeledPoint], testSet: Broadcast[Array[LabeledPoint]], subdel: Int): Iterator[(Int, Array[Array[Float]])] = {
    // Initialization
    var train = new ArrayBuffer[LabeledPoint]
    val size = testSet.value.length

    var dist: Distance.Value = null
    //Distance MANHATTAN or EUCLIDEAN
    if (distanceType == 1)
      dist = Distance.Manhattan
    else
      dist = Distance.Euclidean

    //Join the train set
    while (iter.hasNext)
      train.append(iter.next)

    var knnMemb = new KNN(train, k, dist, numClass)

    var auxSubDel = subdel
    var result = new Array[(Int, Array[Array[Float]])](size)

    for (i <- 0 until size) {
      result(i) = (auxSubDel, knnMemb.neighbors(testSet.value(i).features))
      auxSubDel = auxSubDel + 1
    }

    result.iterator

  }

  /**
   * Join the result of the map taking the nearest neighbors.
   *
   * @param mapOut1 A element of the RDD to join
   * @param mapOut2 Another element of the RDD to join
   * @return Combine of both element with the nearest neighbors
   */
  def combine(mapOut1: Array[Array[Float]], mapOut2: Array[Array[Float]]): Array[Array[Float]] = {

    var itOut1 = 0
    var itOut2 = 0
    var out: Array[Array[Float]] = new Array[Array[Float]](k)

    var i = 0
    while (i < k) {
      out(i) = new Array[Float](2)
      if (mapOut1(itOut1)(0) <= mapOut2(itOut2)(0)) { // Update the matrix taking the k nearest neighbors
        out(i)(0) = mapOut1(itOut1)(0)
        out(i)(1) = mapOut1(itOut1)(1)
        if (mapOut1(itOut1)(0) == mapOut2(itOut2)(0)) {
          i += 1
          if (i < k) {
            out(i) = new Array[Float](2)
            out(i)(0) = mapOut2(itOut2)(0)
            out(i)(1) = mapOut2(itOut2)(1)
            itOut2 = itOut2 + 1
          }
        }
        itOut1 = itOut1 + 1

      } else {
        out(i)(0) = mapOut2(itOut2)(0)
        out(i)(1) = mapOut2(itOut2)(1)
        itOut2 = itOut2 + 1
      }
      i += 1
    }

    out
  }

  /**
   * Calculate majority vote and return the predicted and right class for each instance.
   *
   * @param sample Real instance of the test set and his nearest neighbors
   * @return predicted and right class.
   */
  def calculatePredictedRightClassesFuzzy(sample: (Int, (LabeledPoint, Array[Array[Float]]))): (Double, Double) = {

    val size = sample._2._2.length
    //val numFeatures = test.value.length
    val rightClass: Int = sample._2._1.label.toInt
    val predictedNeigh = sample._2._2

    var auxClas = new Array[Int](numClass)
    var clas = 0
    var numVotes = 0
    for (j <- 0 until k) {
      auxClas(predictedNeigh(j)(1).toInt) = auxClas(predictedNeigh(j)(1).toInt) + 1
      if (auxClas(predictedNeigh(j)(1).toInt) > numVotes) {
        clas = predictedNeigh(j)(1).toInt
        numVotes = auxClas(predictedNeigh(j)(1).toInt)
      }

    }

    (clas.toDouble, rightClass.toDouble)
  }

}

/**
 * Distributed kNN class.
 *
 *
 * @author Jesus Maillo
 */
object kNN_IS {
  /**
   * Initial setting necessary.
   *
   * @param train Data that iterate the RDD of the train set
   * @param test The test set in a broadcasting
   * @param k number of neighbors
   * @param distanceType MANHATTAN = 1 ; EUCLIDEAN = 2
   * @param converter Dataset's information read from the header
   * @param numPartitionMap Number of partition. Number of map tasks
   * @param numReduces Number of reduce tasks
   * @param numIterations Autosettins = -1. Number of split in the test set and number of iterations
   */
  def setup(train: RDD[LabeledPoint], test: RDD[LabeledPoint], k: Int, distanceType: Int, numClass: Int, numFeatures: Int, numPartitionMap: Int, numReduces: Int, numIterations: Int, maxWeight: Double) = {
    new kNN_IS(train, test, k, distanceType, numClass, numFeatures, numPartitionMap, numReduces, numIterations, maxWeight).setup()
  }
}