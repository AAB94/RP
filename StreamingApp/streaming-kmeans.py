from __future__ import print_function
from pyspark import SparkContext, SparkConf
from pyspark.streaming.kafka import KafkaUtils
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.clustering import StreamingKMeans
from pyspark.streaming import StreamingContext
from pyspark.mllib.clustering import KMeans

if __name__ == "__main__":

    conf = SparkConf().set("spark.jars", "/home/benjamin/lib/spark-2.2.0-bin-hadoop2.7/jars/spark-streaming-kafka-0-8-assembly_2.11-2.2.1.jar")

    sc = SparkContext(master="local[4]", appName="Streaming-KMeans", conf=conf)

    ssc = StreamingContext(sc, 5)

    # Kafka Stream
    ks = KafkaUtils.createDirectStream(ssc, ["test"], {"metadata.broker.list": "localhost:9092"})

    # training data for File
    trainingData = sc.textFile("data/datatraining.txt")\
        .map(lambda line: line.split(',')[2:-1]).map(lambda arr: Vectors.dense([float(x) for x in arr]))

    # Supplied to Streaming KMeans as the centers by StreamingKmeans are not giving good predictions
    init_centers = KMeans.train(trainingData, 2).centers

    # We create a model with random clusters and specify the number of clusters to find
    model = StreamingKMeans(k=2, decayFactor=0.4)\
        .setInitialCenters(init_centers, [1.0, 1.0, 1.0, 1.0, 1.0])

    model.trainOn(ssc.queueStream([trainingData]))

    def parse(lp):
        arr = lp.split(',')[2:-1]
        label = lp.split(',')[0]
        vec = Vectors.dense([float(x) for x in arr])
        return LabeledPoint(label, vec)

    test_stream = ks.map(lambda x: x[1]).map(parse)

    result = model.predictOnValues(test_stream.map(lambda lp: (lp.label, lp.features)))

    # Prints Prediction Prediction and Cluster Centers
    def current_centers(time, rdd):
        print("\n--------------------- %s --------------------------" % str(time))
        stream_pred = rdd.collect()
        centers = model.latestModel().centers
        print("Prediction = ", stream_pred, "  Centers ",centers)
        print("\n---------------------------------------------------------------")

    result.foreachRDD(current_centers)

    ssc.start()
    ssc.awaitTermination(240)

    print(model.latestModel().clusterCenters)


