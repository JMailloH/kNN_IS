package org.apache.spark.ml.classification.kNN_IS

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
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row
import org.apache.spark.mllib.linalg.{ DenseVector, SparseVector, Vectors, Vector }
import scala.collection.mutable.ListBuffer
import org.apache.log4j.Logger
import keel.Algorithms.Lazy_Learning.LazyAlgorithm
import keel.Algorithms.Lazy_Learning.KNN
import keel.Algorithms.Lazy_Learning.KNN.KNN
import utils.keel.KeelParser
import org.apache.spark.sql.DataFrame

/**
 * @author Jesus Maillo
 */

private[ml] trait kNN_ISParams extends PredictorParams {

  // Obligatory parameters
  final val K: IntParam = new IntParam(this, "K", "Number of neighbors.", ParamValidators.gtEq(1))
  final def getK: Int = $(K)
  final val distanceType: IntParam = new IntParam(this, "distanceType", "Distance Type: MANHATTAN = 1 ; EUCLIDEAN = 2 ; HVDM = 3.", ParamValidators.gtEq(1))
  final def getDistanceType: Int = $(distanceType)
  final val converter: Param[KeelParser] = new Param(this, "converter", "Information extracted of the dataset.")
  final def getConverter: KeelParser = $(converter)
  final val numPartitionMap: IntParam = new IntParam(this, "numPartitionMap", "Number of partition.", ParamValidators.gtEq(1))
  final def getNumPartitionMap: Int = $(numPartitionMap)
  final val numReduces: IntParam = new IntParam(this, "numReduces", "Number of reduces.", ParamValidators.gtEq(1))
  final def getNumReduces: Int = $(numReduces)
  final var numIter: IntParam = new IntParam(this, "numIterations", "Number of Iteration to classify the test.")
  final def getNumIter: Int = $(numIter)
  final val maxWeight: DoubleParam = new DoubleParam(this, "maxWeight", "Available main memory.")
  final def getMaxWeight: Double = $(maxWeight)
  final val numSamplesTest: IntParam = new IntParam(this, "numSamplesTest", "Number of instance in the test set.", ParamValidators.gtEq(1))
  final val outPath: StringArrayParam = new StringArrayParam(this, "outPath", "Output path.")

  //Count the samples of each data set and the number of classes. Array with the right classes of the test set.
  final var numSamplesTrain: IntParam = new IntParam(this, "numSamplesTrain", "Number of instance in the training set.", ParamValidators.gtEq(1))
  final def getNumSamplesTrain: Int = $(numSamplesTrain)
  final def getNumSamplesTest: Int = $(numSamplesTest)
  final var numClass: IntParam = new IntParam(this, "numClass", "Number of classes.", ParamValidators.gtEq(1))
  final def getNumClass: Int = $(numClass)
  final var numFeatures: IntParam = new IntParam(this, "numFeatures", "Number of features.", ParamValidators.gtEq(1))
  final def getNumFeatures: Int = $(numFeatures)
  final var rightClasses: Param[Array[Double]] = new Param(this, "rightClasses", "Array with the right classes.")
  final def getRightClasses: Array[Double] = $(rightClasses)

  //Setting Iterative MapReduce
  final var inc: IntParam = new IntParam(this, "inc", "Increment used to the iterative behavior.")
  final def getInc: Int = $(inc)
  final var subdel: IntParam = new IntParam(this, "subdel", "Sub-delimiter used to the iterative behavior.")
  final def getSubdel: Int = $(subdel)
  final var topdel: IntParam = new IntParam(this, "topdel", "Top-delimiter used to the iterative behavior.")
  final def getTopdel: Int = $(topdel)

  //Time variables
  final var timeBeg: LongParam = new LongParam(this, "timeBeg", "Start time of the run.")
  final var timeEnd: LongParam = new LongParam(this, "timeEnd", "End time of the run.")
  /*  final var mapTimesArray: Param[Array[Double]] = new Param(this, "mapTimesArray", "Array with the runtimes of each map.")
  final var reduceTimesArray: Param[Array[Double]] = new Param(this, "reduceTimesArray", "Array with the runtimes of reduce stage.")
  final var iterativeTimesArray: Param[Array[Double]] = new Param(this, "reduceTimesArray", "Array with the runtimes of each iterations.")
  final var rightPredictedClasses: Param[Array[Array[Array[Int]]]] = new Param(this, "rightPredictedClasses", "Array with the right and predicted classes.")
*/
}

@Since("1.6.1")
@Experimental
class kNN_ISClassifier(@Since("1.6.1") override val uid: String)
    extends Classifier[Vector, kNN_ISClassifier, kNN_ISClassificationModel]
    with kNN_ISParams {

  @Since("1.6.1")
  def this() = this(Identifiable.randomUID("kNN_IS"))

  //Obligatory parameters
  @Since("1.6.1") def setK(value: Int): this.type = set(K, value) setDefault (K -> 1)
  @Since("1.6.1") def setDistanceType(value: Int): this.type = set(distanceType, value) setDefault (distanceType -> 1)
  @Since("1.6.1") def setConverter(value: KeelParser): this.type = set(converter, value) setDefault (converter -> new KeelParser)
  @Since("1.6.1") def setNumPartitionMap(value: Int): this.type = set(numPartitionMap, value) setDefault (numPartitionMap -> 100)
  @Since("1.6.1") def setNumReduces(value: Int): this.type = set(numReduces, value) setDefault (numReduces -> 20)
  @Since("1.6.1") def setMaxWeight(value: Double): this.type = set(maxWeight, value) setDefault (maxWeight -> -1.0)
  @Since("1.6.1") def setNumSamplesTest(value: Int): this.type = set(numSamplesTest, value) setDefault (numSamplesTest -> 1)
  @Since("1.6.1") def setOutPath(value: Array[String]): this.type = set(outPath, value) setDefault (outPath -> null)

  //Count the samples of each data set and the number of classes. Array with the right classes of the test set.
  @Since("1.6.1") def setNumSamplesTrain(value: Int): this.type = set(numSamplesTrain, value) setDefault (numSamplesTrain -> 1)
  @Since("1.6.1") def setNumClass(value: Int): this.type = set(numClass, value) setDefault (numClass -> 1)
  @Since("1.6.1") def setNumFeatures(value: Int): this.type = set(numFeatures, value) setDefault (numFeatures -> 1)
  @Since("1.6.1") def setRightClasses(value: Array[Double]): this.type = set(rightClasses, value) setDefault (rightClasses -> null)

  //Setting Iterative MapReduce
  @Since("1.6.1") def setInc(value: Int): this.type = set(inc, value) setDefault (inc -> 0)
  @Since("1.6.1") def setTopdel(value: Int): this.type = set(topdel, value) setDefault (topdel -> 0)
  @Since("1.6.1") def setSubdel(value: Int): this.type = set(subdel, value) setDefault (subdel -> 0)
  @Since("1.6.1") def setNumIter(value: Int): this.type = set(numIter, value) setDefault (numIter -> -1)

  //Time variables
  @Since("1.6.1") def setTimeBeg(value: Long): this.type = set(timeBeg, value) setDefault (timeBeg -> 0)
  @Since("1.6.1") def setTimeEnd(value: Long): this.type = set(timeEnd, value) setDefault (timeEnd -> 0)
  /*  @Since("1.6.1") def setMapTimesArray(value: Array[Double]): this.type = set(mapTimesArray, value) setDefault (mapTimesArray -> null)
  @Since("1.6.1") def setReduceTimesArray(value: Array[Double]): this.type = set(reduceTimesArray, value) setDefault (reduceTimesArray -> null)
  @Since("1.6.1") def setIterativeTimesArray(value: Array[Double]): this.type = set(iterativeTimesArray, value) setDefault (iterativeTimesArray -> null)
  @Since("1.6.1") def setRightPredictedClasses(value: Array[Array[Array[Int]]]): this.type = set(rightPredictedClasses, value) setDefault (rightPredictedClasses -> null)
*/
  override protected def train(dataset: DataFrame): kNN_ISClassificationModel = {

    val train: RDD[LabeledPoint] = extractLabeledPoints(dataset)
    this.setNumSamplesTrain(train.count().toInt)
    //this.setNumSamplesTest(test.count().toInt) Know by parameter
    this.setNumClass($(converter).getNumClassFromHeader())
    this.setNumFeatures($(converter).getNumFeaturesFromHeader())
    //this.setNumIter($(numIterations))
    this.setTimeBeg(System.nanoTime)
    this.setTimeEnd(0)

    //Starting logger
    var logger = Logger.getLogger(this.getClass())

    //Setting Iterative MapReduce
    var weightTrain = 0.0
    var weightTest = 0.0

    var numIterAux = 0
    if ($(numIter) == -1) { //Auto-setting the minimum partition test.
      weightTrain = (8 * $(numSamplesTrain) * $(numFeatures)) / ($(numPartitionMap) * 1024.0 * 1024.0)
      weightTest = (8 * $(numSamplesTest) * $(numFeatures)) / (1024.0 * 1024.0)
      if (weightTrain + weightTest < $(maxWeight) * 1024.0) { //It can be run with one iteration
        numIterAux = 1
      } else {
        if (weightTrain >= $(maxWeight) * 1024.0) {
          logger.error("=> Train wight bigger than lim-task. Abort")
          System.exit(1)
        }
        numIterAux = (1 + (weightTest / (($(maxWeight) * 1024.0) - weightTrain)).toInt)
      }

    } else {
      numIterAux = $(numIter)
    }

    this.setInc(($(numSamplesTest) / $(numIter)).toInt)
    this.setSubdel(0)
    this.setTopdel($(inc))
    if ($(numIter) == 1) { //If only one partition
      this.setTopdel($(numSamplesTest).toInt + 1)
    }

    if (this.getTopdel < 0) {
      this.setTopdel((-1) * (this.getTopdel))
    }

    this.setNumIter(numIterAux)
    logger.info("=> NumberIterations \"" + $(numIter) + "\"")

    val adios = kNN_ISClassificationModel.setup(train,
      $(K),
      $(distanceType),
      $(converter),
      $(numPartitionMap),
      $(numReduces),
      $(numIter),
      $(maxWeight),
      $(numSamplesTrain),
      $(numSamplesTest),
      $(numClass),
      $(numFeatures),
      $(inc),
      $(subdel),
      $(topdel),
      $(numIter),
      $(timeBeg),
      $(timeEnd),
      $(outPath),
      this)

    adios
  }

  @Since("1.6.1")
  override def copy(extra: ParamMap): kNN_ISClassifier = defaultCopy(extra)
}

@Since("1.6.0")
object kNN_ISClassifier extends DefaultParamsReadable[kNN_ISClassifier] {

  @Since("1.6.0")
  override def load(path: String): kNN_ISClassifier = super.load(path)
}

@Since("1.6.1")
@Experimental
class kNN_ISClassificationModel private[ml] (override val uid: String,
                                             override val train: RDD[LabeledPoint],
                                             override val k: Int,
                                             override val distanceType: Int,
                                             override val converter: KeelParser,
                                             override val numPartitionMap: Int,
                                             override val numReduces: Int,
                                             override val numIterations: Int,
                                             override val maxWeight: Double,
                                             val numSamplesTrain: Int,
                                             val numSamplesTest: Int,
                                             val numClass: Int,
                                             val numFeaturesAux: Int,
                                             var inc: Int,
                                             var subdel: Int,
                                             var topdel: Int,
                                             var numIter: Int,
                                             var timeBeg: Long,
                                             var timeEnd: Long,
                                             val outPath: Array[String])
    extends ClassificationModel[Vector, kNN_ISClassificationModel]
    with kNN_ISEnsembleModel with Serializable {

  @Since("1.6.1")
  override val numFeatures: Int = numFeaturesAux

  @Since("1.6.1")
  override val numClasses: Int = numClass

  private[ml] def this(train: RDD[LabeledPoint],
                       k: Int,
                       distanceType: Int,
                       converter: KeelParser,
                       numPartitionMap: Int,
                       numReduces: Int,
                       numIterations: Int,
                       maxWeight: Double,
                       numSamplesTrain: Int,
                       numSamplesTest: Int,
                       numClass: Int,
                       numFeatures: Int,
                       inc: Int,
                       subdel: Int,
                       topdel: Int,
                       numIter: Int,
                       timeBeg: Long,
                       timeEnd: Long,
                       outPath: Array[String]) = this(Identifiable.randomUID("kNN_IS"),
    train,
    k,
    distanceType,
    converter,
    numPartitionMap,
    numReduces,
    numIterations,
    maxWeight,
    numSamplesTrain,
    numSamplesTest,
    numClass,
    numFeatures,
    inc,
    subdel,
    topdel,
    numIter,
    timeBeg,
    timeEnd,
    outPath)
  var mapTimesArray = new Array[Double](numIter)
  var reduceTimesArray = new Array[Double](numIter)
  var iterativeTimesArray = new Array[Double](numIter)
  var rightPredictedClasses = new Array[Array[Array[Int]]](numIter)

  private def broadcastTest(test: Array[LabeledPoint], context: SparkContext) = context.broadcast(test)

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
    writerResult += "***Results.txt ==> Contain: Confusion Matrix; Accuracy; AUC (if it is posible); Time of the run***\n"
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

  override def transform(dataset: DataFrame): DataFrame = {
    val test = { dataset.select($(labelCol), $(featuresCol)).map { case Row(label: Double, features: Vector) => LabeledPoint(label, features) } }
    //val test1: RDD[LabeledPoint] = extractLabeledPoints(dataset)
    var numSamplesTest = test.count().toInt
    var rightClasses = test.map(line => line.label).collect

    val testWithKey = test.zipWithIndex().map { line => (line._2.toInt, line._1) }.sortByKey().cache
    var logger = Logger.getLogger(this.getClass())
    var testBroadcast: Broadcast[Array[LabeledPoint]] = null
    val sc = testWithKey.context

    for (i <- 0 to numIterations - 1) {

      //Taking the iterative initial time.
      val timeBegIterative = System.nanoTime

      if (i == numIterations - 1) {
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
        maxTime = maxTime + mapTimes(i)
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

    writeResults(sc, outPath(0))

    var sizeOut = 0
    for (i <- 0 to numIterations - 1) {
      sizeOut = sizeOut + rightPredictedClasses(i).length
    }
    var outRightPredictedClasses = new Array[Array[Int]](sizeOut)
    for (i <- 0 to numIterations - 1) {
      for (j <- 0 to rightPredictedClasses(i).length - 1) {
        outRightPredictedClasses(j) = rightPredictedClasses(i)(j)
      }
    }

    val output = sc.parallelize(outRightPredictedClasses, numPartitionMap)
    val out = output.map { line =>
      val featureVector = Vectors.dense(line(0))
      val label = line(1)
      LabeledPoint(label, featureVector)
    }

    val sqlContext = new org.apache.spark.sql.SQLContext(out.context)
    import sqlContext.implicits._
    out.toDF()
  }

  override protected def predict(features: Vector): Double = {
    //Do not use
    var nullOut: Double = 0.0
    nullOut
  }

  override protected def predictRaw(features: Vector): Vector = {
    //Do not use
    var nullOut: Vector = null
    nullOut
  }

  @Since("1.6.1")
  override def copy(extra: ParamMap): kNN_ISClassificationModel = {
    copyValues(new kNN_ISClassificationModel(
      train: RDD[LabeledPoint],
      k: Int,
      distanceType: Int,
      converter: KeelParser,
      numPartitionMap: Int,
      numReduces: Int,
      numIterations: Int,
      maxWeight: Double,
      numSamplesTrain: Int,
      numSamplesTest: Int,
      numClass: Int,
      numFeatures: Int,
      inc: Int,
      subdel: Int,
      topdel: Int,
      numIter: Int,
      timeBeg: Long,
      timeEnd: Long,
      outPath: Array[String]))
  }

  @Since("1.6.1")
  override def toString: String = {
    s"kNN_ISClassificationModel (uid=$uid) with $k nearest neighbor(s)"
  }
}

private[ml] object kNN_ISClassificationModel {

  def setup(train: RDD[LabeledPoint],
            k: Int,
            distanceType: Int,
            converter: KeelParser,
            numPartitionMap: Int,
            numReduces: Int,
            numIterations: Int,
            maxWeight: Double,
            numSamplesTrain: Int,
            numSamplesTest: Int,
            numClass: Int,
            numFeatures: Int,
            inc: Int,
            subdel: Int,
            topdel: Int,
            numIter: Int,
            timeBeg: Long,
            timeEnd: Long,
            outPath: Array[String],
            parent: kNN_ISClassifier): kNN_ISClassificationModel = {
    val uid = if (parent != null) parent.uid else Identifiable.randomUID("kNN_IS")

    new kNN_ISClassificationModel(uid,
      train,
      k,
      distanceType,
      converter,
      numPartitionMap,
      numReduces,
      numIterations,
      maxWeight,
      numSamplesTrain,
      numSamplesTest,
      numClass,
      numFeatures,
      inc: Int,
      subdel: Int,
      topdel: Int,
      numIter: Int,
      timeBeg: Long,
      timeEnd: Long,
      outPath: Array[String])
  }
}

/**
 * Abstraction for kNN_IS Ensemble models.
 *
 * TODO: Add support for predicting probabilities and raw predictions  SPARK-3727
 */
private[ml] trait kNN_ISEnsembleModel {

  def train: RDD[LabeledPoint]
  def k: Int
  def distanceType: Int
  def converter: KeelParser
  def numPartitionMap: Int
  def numReduces: Int
  def numIterations: Int
  def maxWeight: Double

  /** Summary of the model */
  override def toString: String = {
    // Implementing classes should generally override this method to be more descriptive.
    s"kNN_ISModel with $k nearest neighbor(s)"
  }

  /** Full description of model */
  def toDebugString: String = {
    val header = toString + "\n"
    header
  }
}


