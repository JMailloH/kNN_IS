package utils.keel

import org.apache.spark.SparkContext

/**
 * Get the information from the header and normalize the dataset.
 *
 * @author Jesus Maillo
 */
class KeelParser extends Serializable {

  var conv: Array[Map[String, Double]] = null
  var isCategorical: Array[java.lang.Boolean] = null
  var numClass: Int = 0
  var numFeatures: Int = 0

  /**
   * Constructor.
   *
   * @param sc SparkContext
   * @param file Path to the file. String.
   */
  def this(sc: SparkContext, file: String) {
    this()
    calculateParserFromHeader(sc, file)
  }

  /**
   * Get the labels of a feature or the main class as Array[String].
   *
   * @param str string to parser
   * @author Jesus Maillo
   */
  def getLabels(str: String): Array[String] = {
    var result = str.substring(str.indexOf("{") + 1, str.indexOf("}")).replaceAll(" ", "").split(",")
    result
  }

  /**
   * Get the min and max of a feature as a Array[Double].
   *
   * @param str string to parser
   * @author Jesus Maillo
   */
  def getRange(str: String): Array[Double] = {
    var aux = str.substring(str.indexOf("[") + 1, str.indexOf("]")).replaceAll(" ", "").split(",")
    var result = new Array[Double](2)
    result(0) = aux(0).toDouble
    result(1) = aux(1).toDouble
    result
  }

  /**
   * Calculate the information necesary for parser with function parserToDouble.
   *
   * @param sc The SparkContext
   * @param file path of the header
   * @author Jesus Maillo
   */
  def calculateParserFromHeader(sc: SparkContext, file: String) = {
    //Reading header. Each element is a line
    val header = sc.textFile(file)
    var linesHeader = header.toArray()

    //Calculate number of featires + 1 for the class
    numFeatures = 0
    for (i <- 0 to (linesHeader.length - 1)) {
      if (linesHeader(i).toUpperCase().contains("@INPUTS")) {
        numFeatures = linesHeader(i).length - linesHeader(i).replaceAllLiterally(",", "").length + 2
      } //end if
    } //end for

    //Calculate transformation to normalize and erase categorical features
    conv = new Array[Map[String, Double]](numFeatures)
    isCategorical = new Array[java.lang.Boolean](numFeatures)

    for (i <- 0 to numFeatures - 1) {
      conv(i) = Map()
      isCategorical(i) = false
    }

    var auxParserClasses = 0.0
    var auxNumFeature = 0
    for (i <- 0 to (linesHeader.length - 1)) {
      if (linesHeader(i).toUpperCase().contains("@ATTRIBUTE CLASS")) {
        numClass = linesHeader(i).length - linesHeader(i).replaceAllLiterally(",", "").length + 1
        val labelsClasses = getLabels(linesHeader(i)) //Array of String with the labels of the objetive variable
        for (key <- labelsClasses) { //Calculate map for parser label classes
          conv(numFeatures - 1) += (key -> auxParserClasses)
          isCategorical(auxNumFeature) = true
          auxParserClasses = auxParserClasses + 1
        }
      } else if (linesHeader(i).toUpperCase().contains("[")) { //Real or integer feature
        val range = getRange(linesHeader(i)) //Min and max of the feature
        conv(auxNumFeature) += ("min" -> range(0), "max" -> range(1)) //Do the parser for this feature
        isCategorical(auxNumFeature) = false
        auxNumFeature = auxNumFeature + 1 //Increase for the next feature
      } else if (linesHeader(i).toUpperCase().contains("{") && !(linesHeader(i).toUpperCase().contains("@ATTRIBUTE CLASS"))) {
        val labelsClasses = getLabels(linesHeader(i)) //Array String with the labels of the feature
        val size = labelsClasses.length

        //Calculate the increase. If categorical variable only have a value (WTF) it must do 0 and the increase 1. Dont /0
        var inc: Double = 0.0
        if (size == 1) {
          inc = 1.0
        } else {
          inc = 1.0 / (size - 1.0)
        }

        for (i <- 0 to labelsClasses.length - 1) { //Map to parser the label class
          conv(auxNumFeature) += (labelsClasses(i) -> i * inc)
        }
        isCategorical(auxNumFeature) = true

        auxNumFeature = auxNumFeature + 1
      }
    } //end for

  }

  /**
   * Parser a line to a Array[Double].
   *
   * @param conv Array[Map] with the information to parser
   * @param line The string to be parsed
   * @author Jesus Maillo
   */
  def parserToDouble(line: String): Array[Double] = {
    val size = conv.length
    var result: Array[Double] = new Array[Double](size)

    //Change the line to Array[String]
    val auxArray = line.split(",")

    //Iterate over the array parsing to double each element with the knwolegde of the header
    for (i <- 0 to size - 1) {
      if(auxArray(i) == "?"){
        result(i) = -1
      }else if (conv(i).contains("min") && conv(i).contains("max") && (conv(i).size == 2)) { //If dictionary have like key (only) min and max is real or integer, else, categorical
        result(i) = (auxArray(i).toDouble - conv(i).get("min").get) / (conv(i).get("max").get - conv(i).get("min").get)
      } else {
        result(i) = conv(i).get(auxArray(i)).get
      }
    }

    result
  }

  /**
   * Return the number of label from the objetive class.
   *
   * @param conv Array[Map] with the information to parser
   * @param line The string to be parsed
   * @author Jesus Maillo
   */
  def getNumClassFromHeader(): Int = {
    numClass
  }

  /**
   * Return the number of features.
   *
   * @author Jesus Maillo
   */
  def getNumFeaturesFromHeader(): Int = {
    numFeatures
  }

  /**
   * Return structure necessary for parser with function parserToDouble.
   *
   * @param sc The SparkContext
   * @param file path of the header
   * @author Jesus Maillo
   */
  def getParserFromHeader(): Array[Map[String, Double]] = {
    conv
  }

  /**
   * Return structure necessary for parser with function parserToDouble.
   *
   * @param sc The SparkContext
   * @param file path of the header
   * @author Jesus Maillo
   */
  def getIfCategorical(): Array[java.lang.Boolean] = {
    isCategorical
  }

}