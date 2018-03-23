from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import GaussianMixture
from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("Gaussian").getOrCreate()
'''
Create Dataframe From CSV 
'''
train_df = spark.read.csv("/home/iotsys1/Benjamin/MyCODES/datatraining.csv",
	header=True,
	inferSchema=True,
	)


test1_df = spark.read.csv("/home/iotsys1/Benjamin/MyCODES/datatest.txt",
	    inferSchema=True,
	    header=True
	)

test2_df = spark.read.csv("/home/iotsys1/Benjamin/MyCODES/datatest2.txt",
	    inferSchema=True,
	    header=True
	)


'''
Convert Data to Vector as Standardised Scaler 
and Kmeans can only operate on that data
'''

vectorAssembler = VectorAssembler(
		inputCols=["Temperature","Humidity",
		"Light","CO2","HumidityRatio"],
		outputCol="DenseVector"
	)


train_df = vectorAssembler.transform(train_df)

test1_df = vectorAssembler.transform(test1_df)

test2_df = vectorAssembler.transform(test2_df)


'''
Done to standardise data 
'''

stand_scaled = StandardScaler(
		inputCol="DenseVector",
		outputCol= "features",
		withStd=True,
		withMean=True
	)


scaled_model = stand_scaled.fit(train_df)

train_df = scaled_model.transform(train_df)

scaled_model = stand_scaled.fit(test1_df)

test1_df = scaled_model.transform(test1_df)

scaled_model = stand_scaled.fit(test2_df)

test2_df = scaled_model.transform(test2_df)

gm = GaussianMixture(featuresCol="features",k=2, seed=2, maxIter=20)

gmodel = gm.fit(train_df)

if gmodel.hasSummary:
    print("Cluster sizes", gmodel.summary.clusterSizes)
    print("Clsuters ",gmodel.summary.k)


test1_df = gmodel.transform(test1_df)
test1_df.select("features","Occupancy","prediction").show(5)

test2_df = gmodel.transform(test2_df)
test2_df.select("features","Occupancy","prediction").show(5)

count1 = test1_df.filter(" prediction!=Occupancy").count()
total1 = test1_df.count()

count2 = test2_df.filter(" prediction!=Occupancy").count()
total2 = test2_df.count()

total = total1+total2
tc = count1+count2
ans = float(tc)/float(total)
print(ans)

