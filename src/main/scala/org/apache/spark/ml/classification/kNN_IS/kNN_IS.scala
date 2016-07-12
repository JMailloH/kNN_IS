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
import scala.collection.mutable.ArrayBuffer
import org.apache.log4j.Logger
import org.apache.spark.mllib.classification.kNN_IS.KNN
import utils.keel.KeelParser
import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib.classification.kNN_IS.KNN
import org.apache.spark.mllib.classification.kNN_IS.Distance

/**
 * @author Jesus Maillo
 */

private[ml] trait kNN_ISParams extends PredictorParams {

  // Obligatory parameters
  final val K: IntParam = new IntParam(this, "K", "Number of neighbors.", ParamValidators.gtEq(1))
  final def getK: Int = $(K)
  final val distanceType: IntParam = new IntParam(this, "distanceType", "Distance Type: MANHATTAN = 1 ; EUCLIDEAN = 2 ; HVDM = 3.", ParamValidators.gtEq(1))
  final def getDistanceType: Int = $(distanceType)
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

  /**
   * Initial setting necessary. Auto-set the number of iterations and load the data sets and parameters.
   *
   * @return Instance of this class. *this*
   */
  override protected def train(dataset: DataFrame): kNN_ISClassificationModel = {

    val train: RDD[LabeledPoint] = extractLabeledPoints(dataset)
    this.setNumSamplesTrain(train.count().toInt)
    this.setNumClass($(numClass))
    this.setNumFeatures($(numFeatures))

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

    this.setNumIter(numIterAux)
    logger.info("=> NumberIterations \"" + $(numIter) + "\"")

    val knn = kNN_ISClassificationModel.setup(train,
      $(K),
      $(distanceType),
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
      $(outPath),
      this)

    knn
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
class kNN_ISClassificationModel private[ml] (override val uid: String,
                                             override val train: RDD[LabeledPoint],
                                             override val k: Int,
                                             override val distanceType: Int,
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
                       outPath: Array[String]) = this(Identifiable.randomUID("kNN_IS"),
    train,
    k,
    distanceType,
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
    outPath)
  var mapTimesArray = new Array[Double](numIter)
  var reduceTimesArray = new Array[Double](numIter)
  var iterativeTimesArray = new Array[Double](numIter)
  var rightPredictedClasses = new Array[Array[Array[Int]]](numIter)

  private def broadcastTest(test: Array[LabeledPoint], context: SparkContext) = context.broadcast(test)

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

  /**
   * Predict. kNN
   *
   * @return RDD[(Double, Double)]. First column Predicted class. Second Column Right class.
   */
  override def transform(dataset: DataFrame): DataFrame = {
    val test = { dataset.select($(labelCol), $(featuresCol)).map { case Row(label: Double, features: Vector) => LabeledPoint(label, features) } }
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

    val sqlContext = new org.apache.spark.sql.SQLContext(output.context)
    import sqlContext.implicits._
    output.toDF()
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
            outPath: Array[String],
            parent: kNN_ISClassifier): kNN_ISClassificationModel = {
    val uid = if (parent != null) parent.uid else Identifiable.randomUID("kNN_IS")

    new kNN_ISClassificationModel(uid,
      train,
      k,
      distanceType,
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
      outPath: Array[String])
  }
}

/**
 * Abstraction for kNN_IS.
 *
 */
private[ml] trait kNN_ISEnsembleModel {

  def train: RDD[LabeledPoint]
  def k: Int
  def distanceType: Int
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


