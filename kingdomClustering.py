import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv("C:/minor project/output/combined_taxon_data.txt", sep="\t")

# Drop rows with missing values for taxonomic information
df = df.dropna(subset=['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'taxonID', 'scientificName', 'vernacularName'])

# Encoding categorical columns using LabelEncoder
le = LabelEncoder()

# Apply label encoding on taxonomic columns
df['kingdom_encoded'] = le.fit_transform(df['kingdom'])
df['phylum_encoded'] = le.fit_transform(df['phylum'])
df['class_encoded'] = le.fit_transform(df['class'])
df['order_encoded'] = le.fit_transform(df['order'])
df['family_encoded'] = le.fit_transform(df['family'])
df['genus_encoded'] = le.fit_transform(df['genus'])

# Now let's create a feature matrix for clustering (encode taxonomical hierarchy)
X = df[['kingdom_encoded', 'phylum_encoded', 'class_encoded', 'order_encoded', 'family_encoded', 'genus_encoded']]

# Perform Agglomerative Clustering
agg_clust = AgglomerativeClustering(n_clusters=7)  # Set the number of clusters (or determine it based on your analysis)
df['cluster'] = agg_clust.fit_predict(X)

# Show the first few rows of the result
print(df[['taxonID', 'scientificName', 'vernacularName', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'cluster']])

# Save the resulting DataFrame to a CSV file
df.to_csv('C:/minor project/output/taxon_clusters.csv', index=False)

# Plot a simple count of the clusters for visualization
sns.countplot(x='cluster', data=df)
plt.title('Cluster Distribution')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()