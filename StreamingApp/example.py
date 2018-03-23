from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.clustering import StreamingKMeans
from pyspark.streaming import StreamingContext
from pyspark.mllib.clustering import KMeans
from pyspark import SparkContext

# we make an input stream of vectors for training,
# as well as a stream of vectors for testing


sc = SparkContext();
ssc = StreamingContext(sc, 5);


trainingData = sc.textFile("data/datatraining.txt")\
    .map(lambda line: line.split(',')[2:-1]).map(lambda arr: Vectors.dense([float(x) for x in arr]))

centers = KMeans.train(trainingData, 2).centers


trainingQueue = [trainingData]


trainingStream = ssc.queueStream(trainingQueue)


# We create a model with random clusters and specify the number of clusters to find
model = StreamingKMeans(k=2, decayFactor=0.3)#.setRandomCenters(5, 1.0, 0)
model.setInitialCenters( centers, [1.0,1.0,1.0,1.0,1.0])
# Now register the streams for training and testing and start the job,
# printing the predicted cluster assignments on new data points as they arrive.
model.trainOn(trainingStream)

def parse(lp):
    #label = float(lp[lp.find('(') + 1: lp.find(')')])
    #vec = Vectors.dense(lp[lp.find('[') + 1: lp.find(']')].split(','))
    arr = lp.split(',')[2:-1]
    label = lp.split(',')[0]
    label = label[1:-1]
    vec = Vectors.dense([float(x) for x in arr])
    print(model.latestModel().centers)
    return LabeledPoint(label, vec)


testingData = sc.textFile("data/dataset.txt").map(parse)
testingQueue = [testingData]

testingStream = ssc.queueStream(testingQueue)

result = model.predictOnValues(testingStream.map(lambda lp: (lp.label, lp.features)))
result.pprint()

ssc.start()
#ssc.stop(stopSparkContext=True, stopGraceFully=True)
ssc.awaitTermination()
