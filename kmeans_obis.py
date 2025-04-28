from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Start Spark session
spark = SparkSession.builder.appName("KMeansClustering").getOrCreate()

# Load the dataset
path = "C:/minor project/visualization/obis_cluster_stats.csv"
df = spark.read.option("header", True).option("inferSchema", True).csv(path)

# Select relevant numeric features for clustering
features_cols = [
    'avg_lat', 'avg_lon', 'depth_mean',
    'sst_mean', 'sss_mean', 'density',
    'bathymetry_mean', 'shoredistance_mean'
]

# Drop rows with any null values in the selected columns
df_clean = df.select(features_cols).dropna()

# Assemble features into a vector column
vec_assembler = VectorAssembler(inputCols=features_cols, outputCol="features")
df_kmeans = vec_assembler.transform(df_clean)

# Set your optimal number of clusters (replace with the value you got from the Elbow method)
optimal_k = 4  # Change this if your elbow method result was different

# Train the KMeans model
kmeans = KMeans().setK(optimal_k).setSeed(1).setFeaturesCol("features").setPredictionCol("prediction")
model = kmeans.fit(df_kmeans)

# Make predictions
predictions = model.transform(df_kmeans)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print(f"Silhouette Score: {silhouette}")

# Show sample with predicted cluster
predictions.select(features_cols + ["prediction"]).show(10, truncate=False)
