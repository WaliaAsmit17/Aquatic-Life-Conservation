from pyspark.sql import SparkSession
import pandas as pd
import os

# Initialize Spark session
spark = SparkSession.builder.appName("CleanTaxonCSV").getOrCreate()

# Load the dataset
input_path = "C:/minor project/visualization/cleaned_taxon.csv"
df = spark.read.option("header", True).csv(input_path)

# Drop rows with any missing (null or empty string) values in any column
df_cleaned = df.replace('', None).dropna(how='any')

# Optional: Show number of rows before and after cleaning
print(f"Original rows: {df.count()}, Cleaned rows: {df_cleaned.count()}")

# Convert to pandas DataFrame and save as a single CSV file
output_path = "C:/minor project/visualization/cleaned_taxon_final.csv"
pandas_df = df_cleaned.toPandas()
pandas_df.to_csv(output_path, index=False)

print("Cleaned CSV written to:", output_path)

# Stop the Spark session
spark.stop()