from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D




spark = SparkSession.builder.appName("traffic-Kmeans").getOrCreate()

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

kmeans = KMeans(featuresCol="features").setK(2)   # set number of clusters
kmeans = kmeans.setSeed(1)  # set start point
kmodel = kmeans.fit(train_df)

if kmodel.hasSummary:
    print("KMeans Cluster Size ", kmodel.summary.clusterSizes)
    print("KMeans Cluster Centers", kmodel.clusterCenters())
    print("K Value", kmodel.summary.k)

# Kmeans prediction on Data

predict_df = kmodel.transform(train_df)

predict_df = predict_df.select("avgMeasuredTime","avgSpeed","vehicleCount","prediction")

predict_df.select("avgSpeed", "avgMeasuredTime","vehicleCount","prediction").filter("prediction=1").show(2)

#### Visualise Data ####

c1 = mpatches.Patch(color="green",label="Cluster 1")

c2 = mpatches.Patch(color="red",label="Cluster 2")

colors = {0:"red",1:"green"}

df = predict_df.limit(100).toPandas()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df["avgMeasuredTime"], df["avgSpeed"], df["vehicleCount"], c=df["prediction"].apply(lambda x: colors[x]), s=100, marker="o")

plt.title("Kmeans Traffic Data")

ax.set_xlabel('avgMeasuredTime')
ax.set_ylabel('avgSpeed')
ax.set_zlabel('vehicleCount')

plt.legend(handles=[c1,c2])

plt.show()

