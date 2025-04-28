import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("C:/minor project/output/combined_taxon_data.txt", sep="\t")

# Drop rows with missing country/locality
df = df.dropna(subset=['countryCode', 'locality'])

# Combine for clustering
df['country_locality'] = df['countryCode'] + '_' + df['locality']

# Label encode the combined location
le = LabelEncoder()
df['encoded_location'] = le.fit_transform(df['country_locality'])

# Prepare feature matrix
X = df[['encoded_location']]

# -----------------------------------------
# 1. Elbow Method
# -----------------------------------------
wcss = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K, wcss, 'bo-')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# -----------------------------------------
# 2. Silhouette Score
# -----------------------------------------
print("Silhouette Scores:")
for k in range(2, 11):  # Silhouette score needs at least 2 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"k = {k}, silhouette score = {score:.4f}")