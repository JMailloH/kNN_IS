#kNN-IS: An Iterative Spark-based design of the k-Nearest Neighbors classifier for big data.

This is an open-source Spark package about an exact k-nearest neighbors classification based on Apache Spark. We take advantage of its in-memory operations to simultaneously classify big amounts of unseen cases against a big training dataset. The map phase computes the k-nearest neighbors in different splits of the training data. Afterwards, multiple reducers process the definitive neighbors from the list obtained in the map phase. The key point of this proposal lies on the management of the test set, maintaining it in memory when it is possible. Otherwise, this is split into a minimum number of pieces, applying a MapReduce per chunk, using the caching skills of Spark to reuse the previously partitioned training set. 

###Cite this software as:
 **J. Maillo, S. Ramírez-Gallego, I. Triguero, F. Herrera.** _kNN-IS: An Iterative Spark-based design of the k-Nearest Neighbors classifier for big data._ Knowledge-Based Systems, in press. [doi: 10.1016/j.knosys.2016.06.012](doi: 10.1016/j.knosys.2016.06.012)





#How to use

##Pre-requiriments
The following software have to get installed:
- Sacala. Version 2.10
- Spark. Version 1.6.0
- Maven. Version 3.3.3
- JVM. Java Virtual Machine. Version 1.7.0 because Sacala run over it.

##Download and build with maven
- Download source code: It is host on GitHub. To get the sources and compile them we will need git instructions. The specifically command is:
```git clone https://github.com/JMailloH/kNN_IS.git ```
- Build jar file: Once we have the sources, we need maven and we change to root directory of the project. Thus, with the following instructions, jar file will build:
```mvn package -Dmaven.test.skip=true. ```
It generates the jar file under /target/ directory
- Software version: The approach was development with scala 2.10 and run over Spark 1.6.0.

Another alternative is download by spark-package at [https://spark-packages.org/package/JMailloH/kNN_IS](https://spark-packages.org/package/JMailloH/kNN_IS)


##How to run

A generic sample of run could be: 

```/opt/spark-1.6.0/bin/spark-submit --master "URL" --executor-memory "XG" --total-executor-cores "number-of-cores" --class org.apache.spark.run.runkNN_IS kNN_IS-2.0.jar "path-to-header" "path-to-train" "path-to-test" "number-of-neighbors" "number-of-maps" "number-of-reduces" "number-of-iterations" "path-to-output" ["maximum-mem-per-node-on-GB"] "mllib or ml"```

- Parameters of spark: ```--master "URL" | --executor-memory "XG" | --total-executor-cores "number-of-cores"```. They can be usefull for launch with diferent settings and datasets.
- ```--class org.apache.spark.run.runkNN_IS kNN_IS-2.0.jar``` Determine the jar file to be run.
- ```"path-to-header"``` Path from HDFS to header.
- ```"path-to-train"``` Path from HDFS to training set.
- ```"path-to-test"``` Path from HDFS to test set.
- ```"number-of-neighbors"```Number of neighbors. The value of k.
- ```"number-of-maps"``` Number of map tasks.
- ```"number-of-reduces"``` Number of reduce tasks.
- ```"number-of-iterations"``` Number of iterations. -1 to autosetting
- ```"path-to-output"``` Path from HDFS to output directory. 
- ```["maximum-mem-per-node-on-GB"]``` Optional parameter. Limit on GB for each map task. Used to autosetting the number of iterations.
- ```"mllib or ml"``` Execute the MLlib version or the MLbase version.

##Output
Returns an RDD[(Double,Double)] with predicted class as 1st column and right class as 2on column.

The output with this example is a directory store in HDFS:
 - Report.txt: contains only a file, part-00000. It shows the confusion matrix, precision and total runtime.
 
##Example MLlib

```scala
    val train = sc.textFile(pathTrain: String, numPartitionMap).map(line => converter.parserToLabeledPoint(line)).persist
    val test = sc.textFile(pathTest: String).map(line => converter.parserToLabeledPoint(line)).persist

    val knn = kNN_IS.setup(train, test, K, distanceType, numClass, numFeatures, numPartitionMap, numReduces, numIterations, maxWeight)
    predictions = knn.predict(sc)
```

##Example ML

```scala
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
```

##MulticlassMetrics example

```scala
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
```
