from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.sql.types import *
from pyspark.sql.functions import udf, col
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

spark = SparkSession.builder.appName("Kmeans").getOrCreate()
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


kmeans = KMeans().setK(2)  # set number of clusters
kmeans = kmeans.setSeed(1)  # set start point
kmodel = kmeans.fit(train_df)
centers = kmodel.clusterCenters()


print(centers)

test1_df = kmodel.transform(test1_df)
test1_df.select("features","Occupancy","prediction").show(5)

test2_df = kmodel.transform(test2_df)
test2_df.select("features","Occupancy","prediction").show(5)

count1 =  test1_df.filter(" prediction!=Occupancy").count()
total1 = test1_df.count()

count2 =  test2_df.filter(" prediction!=Occupancy").count()
total2 = test2_df.count()

total = total1+total2
tc = count1+count2
ans = float(tc)/float(total)
print(ans)

#### Convert to PCA ####

pca = PCA(k=2, inputCol="features", outputCol="pcaFeatures")

pcamodel = pca.fit(train_df)

# Features of Data Set 1

pca_ds1_features = test1_df.select("features","prediction")

# Features of Data Set 2

pca_ds2_features = test2_df.select("features","prediction")

# Transform Data

pca_ds1_features = pcamodel.transform(pca_ds1_features)

pca_ds2_features = pcamodel.transform(pca_ds2_features)

def pcaxaxis_(col):
    return float(col[0])

def pcayaxis_(col):
    return float(col[1])

pcaXaxis = udf(pcaxaxis_,FloatType())
pcaYaxis = udf(pcayaxis_,FloatType())


res_ds1 = pca_ds1_features.withColumn("x", pcaXaxis(col("pcaFeatures")))\
                            .withColumn("y", pcaYaxis(col("pcaFeatures"))).select("x","y","prediction")


res_ds2 = pca_ds2_features.withColumn("x", pcaXaxis(col("pcaFeatures")))\
                            .withColumn("y", pcaYaxis(col("pcaFeatures"))).select("x","y","prediction")


#### Visualise Data ####

c1 = mpatches.Patch(color="green",label="Occupied")

c2 = mpatches.Patch(color="red",label="Not Occupied")

df = res_ds1.limit(1000).toPandas()


fig, ax = plt.subplots()

colors = {0: 'red', 1: 'green'}

ax.scatter(df["x"],df["y"],c=df["prediction"].apply(lambda x: colors[x]),s=100,marker="o")

plt.xlabel("x")
plt.ylabel("y")
plt.title("KMeans Occupation Data")
plt.legend(handles=[c1,c2])
plt.show()

df = res_ds2.limit(1000).toPandas()

fig, ax = plt.subplots()

colors = {0: 'red', 1: 'green'}

ax.scatter(df["x"],df["y"],c=df["prediction"].apply(lambda x: colors[x]),s=100,marker="o")

plt.xlabel("x")

plt.ylabel("y")

plt.title("KMeans Occupation Data")

plt.legend(handles=[c1,c2])

plt.show()

