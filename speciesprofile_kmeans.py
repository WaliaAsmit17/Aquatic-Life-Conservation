import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the data
input_path = r"C:\minor project\output\cleaned_speciesprofile.txt"
df = pd.read_csv(input_path, sep='\t')

# Drop missing values
df.dropna(inplace=True)

# Features
features = ['isMarine', 'isFreshwater', 'isTerrestrial', 'isExtinct', 'isBrackish']
X = df[features]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans with chosen number of clusters
optimal_k = 4  # Change this based on elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Save result to CSV
output_path = r"C:\minor project\output\speciesprofile_clusters.csv"
df.to_csv(output_path, index=False)

print(f"✅ KMeans clustering complete. File saved to: {output_path}")