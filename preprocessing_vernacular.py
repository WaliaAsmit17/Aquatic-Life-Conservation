from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os

# Step 1: Create Spark session
spark = SparkSession.builder.appName("VernacularNamePreprocessing").getOrCreate()

# Step 2: Read the TSV input
input_path = "C:/data/WoRMS_DwC-A/vernacularname.txt"
df = spark.read.csv(input_path, header=True, inferSchema=False, sep='\t')

# Step 3: Clean column names
df = df.toDF(*[c.strip() for c in df.columns])

# Step 4: Select only the required columns
required_columns = ['taxonID', 'vernacularName', 'language']
existing_cols = [c for c in required_columns if c in df.columns]
df_filtered = df.select(*existing_cols)

# Step 5: Filter to keep only English entries
df_filtered = df_filtered.filter(col("language") == "ENG")

# Step 6: Drop nulls in key fields and deduplicate if needed
df_cleaned = df_filtered.na.drop(subset=['taxonID', 'vernacularName'])

# Step 7: Convert to Pandas and write directly to a text file
final_output_path = "C:/data/output/cleaned_vernacularname.txt"

# Ensure output directory exists
os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

# Use pandas to write directly to a single file with tab separator
pandas_df = df_cleaned.toPandas()
pandas_df.to_csv(final_output_path, sep='\t', index=False)

print("✅ Cleaned vernacular name data saved to:", final_output_path)

# Step 8: Stop Spark session
spark.stop()