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
import org.apache.hadoop.conf.Configuration;

/**
 * @author Jesus Maillo
 */
object MR_KNN extends Serializable {

  /**
   * @brief Calculate the K nearest neighbor from the test set over the train set.
   *
   * @param iter Data that iterate the RDD of the train set
   * @param testSet The test set in a broadcasting
   * @param classes number of label for the objective class
   * @param numNeighbors Value of the K nearest neighbors to calculate
   * @param distanceType MANHATTAN = 1 ; EUCLIDEAN = 2 ; HVDM = 3
   */
  def knn[T](iter: Iterator[Array[Double]], testSet: Broadcast[Array[Array[Double]]], classes: Int, numNeighbors: Int, distanceType: Int, subdel: Int): Iterator[(Int, Array[Array[Float]])] = { //Si queremos un Pair RDD el iterador seria iter: Iterator[(Long, Array[Double])]
    var begTime = System.nanoTime();
    var auxTime = System.nanoTime();

    // Initialization
    var acu = new ListBuffer[Array[Double]]
    var trainClass = new ListBuffer[Int]
    var trainData = new ListBuffer[Array[Double]]

    val len = testSet.value(0).length

    //Parser to right structures
    while (iter.hasNext) {
      val cur = iter.next
      trainData += cur.dropRight(1)
      var auxClass: Array[Double] = new Array[Double](1)
      auxClass = cur.drop(len - 1)
      //print("Size auxClass: " + auxClass.length + "\n\n")
      trainClass += auxClass(0).toInt
    }
    println("\n\nInizialing the structures: " + ((System.nanoTime - auxTime) / 1e9).toFloat + "\n")
    auxTime = System.nanoTime();

    //Create object KNN with the necesary information
    var outKNN = new KNN(trainData.toArray, trainClass.toArray, classes, numNeighbors, distanceType)

    // Initialization with the first sample to get de data structure
    var result = outKNN.getNeighbors(testSet.value(0).dropRight(1))

    //Calculate the K nearest neighbors for each instance of the test set
    for (i <- 1 to testSet.value.length - 1) {
      result.add(outKNN.getNeighbors(testSet.value(i).dropRight(1)).get(0))
    }

    println("\n\nRunning KNN: " + ((System.nanoTime - auxTime) / 1e9).toFloat + "\n")

    //Change to static array and set the key
    val size = result.size()
    var res = new Array[(Int, Array[Array[Float]])](size + 1)

    //res(0) = new Array[Array[Array[Double]]](size+1)
    var auxSubDel = subdel
    for (i <- 0 to size - 1) {
      val aux = new Array[Array[Float]](numNeighbors)
      //res(i) = (i, aux)
      for (j <- 0 to numNeighbors - 1) {
        aux(j) = new Array[Float](2)
        aux(j)(0) = result.get(i).get(j).getFirst().toFloat
        aux(j)(1) = result.get(i).get(j).getSecond().toFloat

      }
      res(i) = (auxSubDel, aux)
      auxSubDel = auxSubDel + 1
    }

    println("\n\nTo KEY-VALUE: " + ((System.nanoTime - auxTime) / 1e9).toFloat + "\n")
    auxTime = System.nanoTime();

    println("\n***TEST SIZE: " + testSet.value.length + "   ***")
    println("***TRAIN SIZE: " + trainClass.length + "   ***\n")

    val aux = new Array[Array[Float]](numNeighbors)
    for (j <- 0 to numNeighbors - 1) {
      aux(j) = new Array[Float](2)
      aux(j)(0) = ((System.nanoTime - begTime) / 1e9).toFloat
      aux(j)(1) = -1
    }
    res(size) = (-1, aux)

    println("\nTOTAL MAP TIME: " + ((System.nanoTime - begTime) / 1e9).toFloat + "\n")

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

    //print("\n\n\n" + size + "\n\n\n" + numNeighbors + "\n\n\n")
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
    
    if(size==2){
      positive = confusionMatrix(0)(0)/confusionMatrix(1)(0)
      negative = confusionMatrix(1)(1)/confusionMatrix(0)(1)
    }

    val AUC = (positive+negative) / 2.0
    AUC
  }

}