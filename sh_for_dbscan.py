from sklearn.metrics import silhouette_score
import pandas as pd

# Load your clustered CSV
df = pd.read_csv(r"C:\minor project\visualization\obis_clusters_for_tableau.csv")

# Assume columns: 'decimalLatitude', 'decimalLongitude', 'cluster'
X = df[['decimalLatitude', 'decimalLongitude']]
labels = df['cluster']

# Remove noise points if labeled -1
mask = labels != -1
X_filtered = X[mask]
labels_filtered = labels[mask]

# Calculate silhouette score
score = silhouette_score(X_filtered, labels_filtered)
print("Silhouette Score:", score)