import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import haversine as hs
from haversine import Unit

# Load the geospatial data prepared in the previous step
input_path = "C:/minor project/output/gbif_geospatial.csv"
output_path = "C:/minor project/output/gbif_clusters_for_tableau.csv"

try:
    # Read the geospatial data
    print(f"Loading geospatial data from {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Loaded {len(df)} records with columns: {', '.join(df.columns)}")
    
    # Extract coordinates for clustering
    coords = df[['latitude', 'longitude']].values
    print(f"Extracted coordinates array with shape: {coords.shape}")
    
    # Scale the coordinates (especially important when using Euclidean distance)
    # Note: For geospatial data, haversine distance is often better, but we'll handle that below
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    # Define a function to calculate haversine distance in km
    def haversine_distance(point1, point2):
        """Calculate haversine distance between two points in km"""
        # Convert scaled points back to original coordinates
        p1 = scaler.inverse_transform([point1])[0]
        p2 = scaler.inverse_transform([point2])[0]
        # Calculate distance in km
        return hs.haversine((p1[0], p1[1]), (p2[0], p2[1]), unit=Unit.KILOMETERS)
    
    # Find optimal DBSCAN parameters
    # We'll try a range of eps values and pick the one with the best silhouette score
    best_eps = 0
    best_min_samples = 0
    best_score = -1
    best_n_clusters = 0
    
    print("Searching for optimal DBSCAN parameters...")
    
    # For larger datasets, you might want to sample to speed up parameter search
    sample_size = min(10000, len(coords_scaled))
    sample_indices = np.random.choice(len(coords_scaled), sample_size, replace=False)
    coords_sample = coords_scaled[sample_indices]
    
    # Search for optimal parameters
    eps_values = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3]
    min_samples_values = [5, 10, 15, 20, 30]
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            cluster_labels = dbscan.fit_predict(coords_sample)
            
            # Count number of clusters (excluding noise points with label -1)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            # Only calculate silhouette score if we have at least 2 clusters
            if n_clusters >= 2:
                # Filter out noise points for silhouette calculation
                mask = cluster_labels != -1
                if np.sum(mask) > n_clusters:  # Ensure we have enough points
                    score = silhouette_score(coords_sample[mask], cluster_labels[mask])
                    print(f"  eps={eps}, min_samples={min_samples}: {n_clusters} clusters, score={score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_eps = eps
                        best_min_samples = min_samples
                        best_n_clusters = n_clusters
    
    # If no suitable parameters were found, use reasonable defaults
    if best_eps == 0:
        best_eps = 0.1
        best_min_samples = 10
        print(f"No optimal parameters found in search range. Using defaults: eps={best_eps}, min_samples={best_min_samples}")
    else:
        print(f"Best parameters: eps={best_eps}, min_samples={best_min_samples}, expected clusters: {best_n_clusters}")
    
    # Apply DBSCAN with the best parameters to the full dataset
    print("Applying DBSCAN to the full dataset...")
    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples, n_jobs=-1)
    df['cluster'] = dbscan.fit_predict(coords_scaled)
    
    # Count clusters and noise points
    n_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'] else 0)
    n_noise = list(df['cluster']).count(-1)
    
    print(f"DBSCAN results: {n_clusters} clusters, {n_noise} noise points ({n_noise/len(df)*100:.2f}%)")
    
    # Calculate cluster stats for Tableau visualization
    # Group by cluster and calculate count, centroid, density
    cluster_stats = df.groupby('cluster').agg(
        count=('species', 'count'),
        avg_lat=('latitude', 'mean'),
        avg_lon=('longitude', 'mean'),
        min_lat=('latitude', 'min'),
        max_lat=('latitude', 'max'),
        min_lon=('longitude', 'min'),
        max_lon=('longitude', 'max')
    ).reset_index()
    
    # Calculate cluster radius (approximate, in km)
    def calculate_radius(group):
        # Calculate the maximum distance from any point to the centroid
        if len(group) <= 1:
            return 0
        centroid = (group['latitude'].mean(), group['longitude'].mean())
        max_dist = 0
        for _, row in group.iterrows():
            dist = hs.haversine(centroid, (row['latitude'], row['longitude']), unit=Unit.KILOMETERS)
            max_dist = max(max_dist, dist)
        return max_dist
    
    # Calculate radius for each cluster
    cluster_radii = {}
    for cluster_id in df['cluster'].unique():
        if cluster_id == -1:  # Skip noise points
            cluster_radii[cluster_id] = 0
            continue
        group = df[df['cluster'] == cluster_id]
        cluster_radii[cluster_id] = calculate_radius(group)
    
    # Add radius to cluster stats
    cluster_stats['radius_km'] = cluster_stats['cluster'].map(cluster_radii)
    
    # Calculate cluster density (points per square km)
    cluster_stats['area_km2'] = np.pi * cluster_stats['radius_km'] ** 2
    cluster_stats['density'] = cluster_stats['count'] / cluster_stats['area_km2'].replace(0, 1)  # Avoid division by zero
    
    # For noise points (cluster -1), set density to 0
    cluster_stats.loc[cluster_stats['cluster'] == -1, 'density'] = 0
    
    # Prepare the output dataset for Tableau
    # 1. Original data with cluster assignments
    df_for_tableau = df.copy()
    df_for_tableau['is_noise'] = df['cluster'] == -1
    df_for_tableau['cluster'] = df['cluster'].apply(lambda x: 'Noise' if x == -1 else f'Cluster {x}')
    
    # 2. Add cluster statistics
    cluster_id_to_name = {-1: 'Noise'}
    for i in range(n_clusters):
        cluster_id_to_name[i] = f'Cluster {i}'
    
    # Rename clusters in stats dataframe
    cluster_stats['cluster_name'] = cluster_stats['cluster'].map(cluster_id_to_name)
    
    # Save to CSV for Tableau
    print(f"Saving clustered data to {output_path}...")
    df_for_tableau.to_csv(output_path, index=False)
    
    # Save cluster stats for reference
    stats_output_path = "C:/minor project/output/gbif_cluster_stats.csv"
    cluster_stats.to_csv(stats_output_path, index=False)
    print(f"Cluster statistics saved to {stats_output_path}")
    
    # Generate a simple plot for quick visualization
    plt.figure(figsize=(10, 8))
    
    # Plot regular points
    regular_points = df_for_tableau[~df_for_tableau['is_noise']]
    noise_points = df_for_tableau[df_for_tableau['is_noise']]
    
    # Plot clusters with different colors
    unique_clusters = regular_points['cluster'].unique()
    for cluster in unique_clusters:
        cluster_points = regular_points[regular_points['cluster'] == cluster]
        plt.scatter(
            cluster_points['longitude'], 
            cluster_points['latitude'],
            s=10,
            alpha=0.6,
            label=cluster
        )
    
    # Plot noise points in gray
    if len(noise_points) > 0:
        plt.scatter(
            noise_points['longitude'], 
            noise_points['latitude'],
            s=5,
            alpha=0.3,
            color='gray',
            label='Noise'
        )
    
    plt.title(f'DBSCAN Clustering Results: {n_clusters} clusters identified')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_path = "C:/minor project/output/gbif_clusters_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Preview plot saved to {plot_path}")
    
    print(f"\n✅ DBSCAN clustering complete! The data is ready for visualization in Tableau.")
    print(f"Tableau tips for heatmaps:")
    print(f"1. Connect to the CSV file: {output_path}")
    print(f"2. Create a map visualization")
    print(f"3. Add Latitude and Longitude to Rows and Columns")
    print(f"4. Add 'cluster' to Color mark")
    print(f"5. For a density heatmap, use the Density mark type or add Count to Size")
    print(f"6. Use the cluster_stats.csv file to create additional visualizations of cluster characteristics")

except Exception as e:
    import traceback
    print(f"Error in DBSCAN processing: {str(e)}")
    traceback.print_exc()
    print("Please check if you have the required libraries installed:")
    print("pip install scikit-learn pandas numpy matplotlib haversine")