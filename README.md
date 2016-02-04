#kNN-IS.

This is an open-source Spark package about an exact k-nearest neighbors classification based on Apache Spark. We take advantage of its in-memory operations to simultaneously classify big amounts of unseen cases against a big training dataset. The map phase computes the k-nearest neighbors in different splits of the training data. Afterwards, multiple reducers process the definitive neighbors from the list obtained in the map phase. The key point of this proposal lies on the management of the test set, maintaining it in memory when it is possible. Otherwise, this is split into a minimum number of pieces, applying a MapReduce per chunk, using the caching skills of Spark to reuse the previously partitioned training set. 

##How to use

###Pre-requiriments
The following software have to get installed:
- Sacala. Version 2.10
- Spark. Version 1.6.0
- Maven. Version 3.3.3
- JVM. Java Virtual Machine. Version 1.7.0 because Sacala run over it.

###Download and build with maven
- Download source code: It is host on GitHub. To get the sources and compile them we will need git instructions. The specifically command is:
```git clone https://github.com/JMailloH/kNN_IS.git ```
- Build jar file: Once we have the sources, we need maven and we change to root directory of the project. Thus, with the following instructions, jar file will build:
```mvn package -Dmaven.test.skip=true. ```
It generates the jar file under /target/ directory
- Software version: The approach was development with scala 2.10 and run over Spark 1.5.1.

###How to run

A generic sample of run could be: 

```/opt/spark-1.5.1/bin/spark-submit --master "URL" --executor-memory "XG" --total-executor-cores "number-of-cores" --class org.KnnMR.KnnMR KnnMR-beta.jar "path-to-header" "path-to-train" "path-to-test" "number-of-neighbors" "number-of-maps" "number-of-reduces" "number-of-iterations" "path-to-output" ["maximum-mem-per-node-on-GB"]```

- Parameters of spark: ```--master "URL" | --executor-memory "XG" | --total-executor-cores "number-of-cores"```. They can be usefull for launch with diferent settings and datasets.
- ```--class org.KnnMR.KnnMR KnnMR-beta.jar``` Determine the jar file to be run.
- ```"path-to-header"``` Path from HDFS to header. Its contains information for normalize data set. Directory datasets/poker-5-fold/ contain a example dataset obtained from [UCI Repository](http://archive.ics.uci.edu/ml/)
- ```"path-to-train"``` Path from HDFS to training set.
- ```"path-to-test"``` Path from HDFS to test set.
- ```"number-of-neighbors"```Number of neighbors. The value of k.
- ```"number-of-maps"``` Number of map tasks.
- ```"number-of-reduces"``` Number of reduce tasks.
- ```"number-of-iterations"``` Number of iterations. -1 to autosetting
- ```"path-to-output"``` Path from HDFS to output directory. 
- ```["maximum-mem-per-node-on-GB"]``` Optional parameter. Limit on GB for each map task. Used to autosetting the number of iterations.

###Output
The output directory will store in HDFS. It contains 3 sub-directory with the result:
 - Predictions.txt: contains only a file, part-00000. It shows the predicted and right class in two column.
 - Results.txt: contains only a file, part-00000- It contains the confusion matrix, the accuracy and the total runtime.
 - Times.txt: contains only a file, part-00000. It presents the time by map, reduce, each iterations and the total runtime.

##Classes and method

- **src/main/scala/org/KnnMR/KnnMR.scala** The main class. Set the Spark enviroment and the method. Launch all map and reduce, if is needed iterative-MapReduce. Finally, write the result in HDFS.
- **src/main/scala/org/KnnMR/MR_KNN.scala** It contains Map and Reduce funtions. 
    - *knn* compute k-nearest neighbors for train and test splits. 
    - *combine* join the result by KEY taking the k nearest.
    - *calculateRightPredictedClasses* Calculate the predicted class computing majority voting and get the right class from the original test set.
    - *calculateConfusionMatrix* Return the confusion matrix obtained from the output of *calculateRightPredictedClasses*
    - *calculateAccuracy* Return the accuracy from the confusion matrix.
- **src/main/scala/org/KnnMR/Utils/KeelParser.scala** This class obtain usefull information from the header.
    - *KeelParser* Defaul constructor has path-Header as param. Get the number of classes and a Map to normalize and parser the dataset.
