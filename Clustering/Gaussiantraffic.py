from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import GaussianMixture
from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("traffic-gaussian").getOrCreate()

'''
Create Dataframe From CSV 
'''

train_df = spark.read.csv("/home/iotsys1/Benjamin/DATASET/trafficData.csv",
	header=True,
	inferSchema=True,
	)


'''
Convert Data to Vector as Standardised Scaler 
and Kmeans can only operate on that data
'''

vectorAssembler = VectorAssembler(
		inputCols=["avgMeasuredTime","avgSpeed","vehicleCount"],
		outputCol="features"
	)


train_df = vectorAssembler.transform(train_df)



'''
Done to standardise data 
'''

stand_scaled = StandardScaler(
		inputCol="features",
		outputCol= "sfeatures",
		withStd=True,
		withMean=True
	)


scaled_model = stand_scaled.fit(train_df)

train_df = scaled_model.transform(train_df)

gm = GaussianMixture(featuresCol="sfeatures",k=2, seed=2, maxIter=20)

gmodel = gm.fit(train_df)

if gmodel.hasSummary:
    print("Cluster sizes", gmodel.summary.clusterSizes)
    print("Clsuters ",gmodel.summary.k)

transformed = gmodel.transform(train_df)

transformed.show(10)
