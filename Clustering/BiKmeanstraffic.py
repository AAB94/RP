from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import BisectingKMeans
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.patches as mpatches

spark = SparkSession.builder.appName("Bi-Kmeans-traffic").getOrCreate()
'''
Create Dataframe From CSV 
'''
train_df = spark.read.csv("/home/iotsys1/Benjamin/MyCODES/Data/trafficData.csv",
	header=True,
	inferSchema=True,
	)


'''
Removing Columns that are not needed from DataFrame
'''
#cluster_df = cluster_df.drop("id","date")


'''
Convert Data to Vector as Standardised Scaler 
and Kmeans can only operate on that data
'''
vectorAssembler = VectorAssembler(
		inputCols=["avgMeasuredTime","avgSpeed","vehicleCount"],
		outputCol="DenseVector"
	)



train_df = vectorAssembler.transform(train_df)


'''
Done to standardise data 
'''

stand_scaled = StandardScaler(
		inputCol="DenseVector",
		outputCol= "features",
		withStd=True,
		withMean=True
	)

'''
outputCol must be named Features as Spark KMeans will only use that column as input
'''

scaled_model = stand_scaled.fit(train_df)

train_df = scaled_model.transform(train_df)

bkmeans = BisectingKMeans().setK(2)
bkmeans = bkmeans.setSeed(1)
bkmodel = bkmeans.fit(train_df)
bkcenters = bkmodel.clusterCenters()

if bkmodel.hasSummary:
	print(bkmodel.summary.clusterSizes)
	print(bkmodel.clusterCenters())


predict_df = bkmodel.transform(train_df)

predict_df = predict_df.select("avgMeasuredTime","avgSpeed","vehicleCount","prediction")

predict_df.show(2)




c1 = mpatches.Patch(color="green",label="No Traffic")

c2 = mpatches.Patch(color="red",label="Traffic")

df = predict_df.limit(100).toPandas()

colors = {0:"red",1:"green"}

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



ax.scatter(df["avgSpeed"],df["avgMeasuredTime"],df["vehicleCount"],c=df["prediction"].apply(lambda x: colors[x]),s=100,marker="o")

plt.title("BiKmeans Traffic Data")
ax.set_xlabel('avgMeasuredTime')
ax.set_ylabel('avgSpeed')
ax.set_zlabel('vehicleCount')

plt.legend(handles=[c1,c2])

plt.show()


