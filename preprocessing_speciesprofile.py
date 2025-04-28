from pyspark.sql import SparkSession
import os

# Step 1: Create Spark session
spark = SparkSession.builder.appName("SpeciesProfilePreprocessing").getOrCreate()

# Step 2: Read the TSV input
input_path = "C:/data/WoRMS_DwC-A/speciesprofile.txt"
df = spark.read.csv(input_path, header=True, inferSchema=False, sep='\t')

# Step 3: Clean column names
df = df.toDF(*[c.strip() for c in df.columns])

# Step 4: Drop rows with null values in all columns
df_cleaned = df.na.drop()

# Step 5: Convert to Pandas and write directly to a text file
final_output_path = "C:/data/output/cleaned_speciesprofile.txt"

# Ensure output directory exists
os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

# Use pandas to write directly to a single file with tab separator
pandas_df = df_cleaned.toPandas()
pandas_df.to_csv(final_output_path, sep='\t', index=False)

print("✅ Cleaned species profile data saved to:", final_output_path)

# Step 6: Stop Spark session
spark.stop()