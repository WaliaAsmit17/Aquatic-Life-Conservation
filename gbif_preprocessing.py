from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder \
    .appName("GBIF Geospatial Extraction for DBSCAN") \
    .getOrCreate()

# Define input and output paths
input_path = "C:/minor project/output/gbif.csv"
output_path = "C:/minor project/output/gbif_geospatial.csv"

try:
    # Read the source CSV
    print(f"Reading data from {input_path}...")
    df = spark.read.option("header", "true") \
               .option("inferSchema", "true") \
               .csv(input_path)
    
    # Print schema to verify column types
    print("Source data schema:")
    df.printSchema()
    
    # Count records before filtering
    total_count = df.count()
    print(f"Total records in source: {total_count}")
    
    # Extract just what's needed for DBSCAN: species and coordinates
    # For DBSCAN, we typically need at minimum:
    # 1. An identifier (species in this case)
    # 2. The coordinates (latitude/longitude)
    geospatial_df = df.select(
        "species",
        col("decimalLatitude").cast("double").alias("latitude"),
        col("decimalLongitude").cast("double").alias("longitude")
    ).filter(
        # Filter out missing coordinates (crucial for DBSCAN)
        col("decimalLatitude").isNotNull() &
        col("decimalLongitude").isNotNull() &
        # Filter out invalid coordinates
        (col("decimalLatitude").cast("double") >= -90) &
        (col("decimalLatitude").cast("double") <= 90) &
        (col("decimalLongitude").cast("double") >= -180) &
        (col("decimalLongitude").cast("double") <= 180)
    )
    
    # Count valid records
    valid_count = geospatial_df.count()
    print(f"Valid geospatial records: {valid_count}")
    print(f"Filtered out {total_count - valid_count} invalid or incomplete records")
    
    # Show a sample of the data
    print("Sample of processed data:")
    geospatial_df.show(5)
    
    # Convert to pandas and write to a single CSV file
    print("Writing geospatial data to CSV...")
    pandas_df = geospatial_df.toPandas()
    pandas_df.to_csv(output_path, index=False)
    
    print(f"✅ Geospatial data successfully extracted and saved to: {output_path}")
    print(f"Total records in output: {len(pandas_df)}")
    print("This data is now ready for DBSCAN clustering.")

except Exception as e:
    print(f"Error processing data: {str(e)}")

finally:
    spark.stop()
    print("Spark session closed")