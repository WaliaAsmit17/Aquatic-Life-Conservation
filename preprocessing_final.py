from pyspark.sql import SparkSession
import os
import pandas as pd

# Step 1: Create Spark session
spark = SparkSession.builder.appName("CombineTaxonData").getOrCreate()

# Step 2: Read the cleaned TSV files
taxon_path = "C:/minor project/output/cleaned_taxon.txt"
vernacular_path = "C:/minor project/output/cleaned_vernacularname.txt" 
species_profile_path = "C:/minor project/output/cleaned_speciesprofile.txt"
distribution_path = "C:/minor project/output/cleaned_distribution.txt"

# Read all files with pandas directly since we're having issues with Spark joins
taxon_df = pd.read_csv(taxon_path, sep='\t')
vernacular_df = pd.read_csv(vernacular_path, sep='\t')
species_df = pd.read_csv(species_profile_path, sep='\t')
distribution_df = pd.read_csv(distribution_path, sep='\t')

# Step 3: Merge all dataframes on taxonID using pandas merge
# Start with taxon as the base
combined_df = taxon_df

# Merge with vernacular data (left join)
if not vernacular_df.empty:
    combined_df = pd.merge(combined_df, vernacular_df, on="taxonID", how="left", suffixes=('', '_vern'))
    # Remove any duplicate taxonID columns
    if "taxonID_vern" in combined_df.columns:
        combined_df = combined_df.drop(columns=["taxonID_vern"])

# Merge with species profile data
if not species_df.empty:
    combined_df = pd.merge(combined_df, species_df, on="taxonID", how="left", suffixes=('', '_species'))
    # Remove any duplicate taxonID columns
    if "taxonID_species" in combined_df.columns:
        combined_df = combined_df.drop(columns=["taxonID_species"])

# Merge with distribution data
if not distribution_df.empty:
    combined_df = pd.merge(combined_df, distribution_df, on="taxonID", how="left", suffixes=('', '_dist'))
    # Remove any duplicate taxonID columns
    if "taxonID_dist" in combined_df.columns:
        combined_df = combined_df.drop(columns=["taxonID_dist"])

# Step 4: Write the combined data to a single file
final_output_path = "C:/minor project/output/combined_taxon_data.txt"

# Ensure output directory exists
os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

# Write to a tab-separated file
combined_df.to_csv(final_output_path, sep='\t', index=False)

print("✅ Combined data saved to:", final_output_path)

# No need to stop Spark session since we're using pandas directly