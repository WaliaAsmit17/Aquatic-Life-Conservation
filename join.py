from pyspark.sql import SparkSession

# Start Spark session
spark = SparkSession.builder.appName("JoinTaxonFiles").getOrCreate()

# Load the CSV files
taxon_df = spark.read.option("header", True).option("inferSchema", True).csv("C:/minor project/visualization/cleaned_taxon_final.csv")
profile_df = spark.read.option("header", True).option("inferSchema", True).csv("C:/minor project/visualization/speciesprofile_clusters.csv")

# Perform inner join on 'taxonID'
joined_df = taxon_df.join(profile_df, on="taxonID", how="inner")

# Optional: Drop duplicates if any
joined_df = joined_df.dropDuplicates(["taxonID"])

# Save the joined DataFrame to CSV
output_path = "C:/minor project/visualization/combined_taxon_profile.csv"
joined_df.coalesce(1).write.option("header", True).csv(output_path)

print("Join completed and saved to:", output_path)
