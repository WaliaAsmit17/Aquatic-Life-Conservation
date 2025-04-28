import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Step 1: Load the dataset
file_path = r"C:\minor project\output\combined_taxon_data.txt"
df = pd.read_csv(file_path, sep='\t')

# Step 2: Drop rows with missing locality or countryCode
df = df.dropna(subset=['locality', 'countryCode'])

# Step 3: Group and count species per locality within each country
grouped = df.groupby(['countryCode', 'locality']).agg({
    'taxonID': 'count'
}).reset_index().rename(columns={'taxonID': 'species_count'})

# Step 4: Encode countryCode into numeric form
le = LabelEncoder()
grouped['country_encoded'] = le.fit_transform(grouped['countryCode'])

# Step 5: Prepare features for clustering
X = grouped[['country_encoded', 'species_count']]

# Step 6: Apply KMeans
k = 3  # You can adjust the number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
grouped['cluster'] = kmeans.fit_predict(X)

# Step 7: Optional – visualize clusters
plt.figure(figsize=(10, 6))
plt.scatter(grouped['species_count'], grouped['country_encoded'], c=grouped['cluster'], cmap='viridis')
plt.xlabel("Species Count")
plt.ylabel("Encoded Country")
plt.title("KMeans Clustering of Localities by Species Count")
plt.show()

# Step 8: Save to CSV for Tableau
output_path = r"C:\minor project\output\locality_clusters.csv"
grouped.to_csv(output_path, index=False)