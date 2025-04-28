from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
import matplotlib.pyplot as plt

# Start Spark session
spark = SparkSession.builder.appName("ElbowMethod").getOrCreate()

# Load the dataset
path = "C:/minor project/visualization/obis_cluster_stats.csv"
df = spark.read.option("header", True).option("inferSchema", True).csv(path)

# Feature columns for elbow method
features_cols = [
    'avg_lat', 'avg_lon', 'depth_mean',
    'sst_mean', 'sss_mean', 'density',
    'bathymetry_mean', 'shoredistance_mean'
]

# Filter out rows with missing values in selected features
df_clean = df.select(features_cols).dropna()

# Assemble the features into a single vector
vec_assembler = VectorAssembler(inputCols=features_cols, outputCol="features")
df_kmeans = vec_assembler.transform(df_clean)

# Elbow Method: compute WSSSE for different k
cost = []
k_values = list(range(2, 11))

for k in k_values:
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
    model = kmeans.fit(df_kmeans)
    cost.append(model.summary.trainingCost)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, cost, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within Set Sum of Squared Errors (WSSSE)')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.grid(True)
plt.tight_layout()
plt.show()
