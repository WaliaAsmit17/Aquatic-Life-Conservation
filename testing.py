from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Remove Missing Entries") \
    .getOrCreate()

# Load the CSV file
input_path = "C:/minor project/visualization/cleaned_taxon.csv"
df = spark.read.option("header", True).csv(input_path)

# Trim whitespaces (optional but good practice)
df_trimmed = df.select([trim(col(c)).alias(c) for c in df.columns])

# Drop rows with any null or empty string values
df_cleaned = df_trimmed.filter(
    " AND ".join([f"{c} IS NOT NULL AND {c} != ''" for c in df_trimmed.columns])
)

# Save the cleaned dataframe to a new CSV file
output_path = "C:/minor project/visualization/cleaned_taxon_filtered.csv"
df_cleaned.coalesce(1).write.mode("overwrite").option("header", True).csv(output_path)
