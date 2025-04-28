import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import haversine as hs
from haversine import Unit
import os

# File paths
input_path = "C:/minor project/output/obis_80000.csv"
output_path = "C:/minor project/output/obis_clusters_for_tableau.csv"
stats_output_path = "C:/minor project/output/obis_cluster_stats.csv"

try:
    # Read the OBIS data
    print(f"Reading OBIS data from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    
    print(f"Loaded {len(df)} records")
    
    # Select relevant columns for DBSCAN and insights
    essential_columns = ['decimalLatitude', 'decimalLongitude']
    
    # Additional columns for enriched analysis (select what's available in your dataset)
    enrichment_columns = [
        'species', 'scientificName', 'depth', 'maximumDepthInMeters', 'minimumDepthInMeters',
        'year', 'date_year', 'phylum', 'class', 'order', 'family', 'genus', 'marine',
        'bathymetry', 'shoredistance', 'sst', 'sss'
    ]
    
    # Check which columns actually exist in the dataset
    available_columns = [col for col in essential_columns + enrichment_columns if col in df.columns]
    
    # Ensure essential columns exist
    if not all(col in available_columns for col in essential_columns):
        missing = [col for col in essential_columns if col not in available_columns]
        raise ValueError(f"Missing essential columns for DBSCAN: {missing}")
    
    # Select available columns
    selected_df = df[available_columns].copy()
    
    # Drop rows with missing coordinates
    initial_count = len(selected_df)
    selected_df = selected_df.dropna(subset=['decimalLatitude', 'decimalLongitude'])
    print(f"Dropped {initial_count - len(selected_df)} rows with missing coordinates")
    
    # Filter out invalid coordinates
    selected_df = selected_df[
        (selected_df['decimalLatitude'] >= -90) & 
        (selected_df['decimalLatitude'] <= 90) &
        (selected_df['decimalLongitude'] >= -180) & 
        (selected_df['decimalLongitude'] <= 180)
    ]
    print(f"Retained {len(selected_df)} rows after filtering invalid coordinates")
    
    # Extract coordinates for clustering
    coords = selected_df[['decimalLatitude', 'decimalLongitude']].values
    
    # Scale the coordinates
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    # Find optimal DBSCAN parameters using a smaller sample for efficiency
    # For large datasets (>10,000 rows), use sampling to speed up parameter search
    sample_size = min(10000, len(coords_scaled))
    sample_indices = np.random.choice(len(coords_scaled), sample_size, replace=False)
    coords_sample = coords_scaled[sample_indices]
    
    print("Searching for optimal DBSCAN parameters...")
    
    # Parameter ranges to search
    eps_values = [0.05, 0.1, 0.15, 0.2, 0.3]
    min_samples_values = [10, 20, 50, 100]
    
    best_eps = 0
    best_min_samples = 0
    best_score = -1
    best_n_clusters = 0
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            try:
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
            except Exception as e:
                print(f"Error testing parameters eps={eps}, min_samples={min_samples}: {str(e)}")
    
    # If no suitable parameters were found, use reasonable defaults
    if best_eps == 0:
        best_eps = 0.1
        best_min_samples = 50
        print(f"No optimal parameters found. Using defaults: eps={best_eps}, min_samples={best_min_samples}")
    else:
        print(f"Best parameters: eps={best_eps}, min_samples={best_min_samples}, expected clusters: {best_n_clusters}")
    
    # Apply DBSCAN with the best parameters to the full dataset
    print("Applying DBSCAN to the full dataset...")
    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples, n_jobs=-1)
    selected_df['cluster'] = dbscan.fit_predict(coords_scaled)
    
    # Count clusters and noise points
    n_clusters = len(set(selected_df['cluster'])) - (1 if -1 in selected_df['cluster'] else 0)
    n_noise = list(selected_df['cluster']).count(-1)
    
    print(f"DBSCAN results: {n_clusters} clusters, {n_noise} noise points ({n_noise/len(selected_df)*100:.2f}%)")
    
    # Calculate additional features for each cluster
    cluster_stats = selected_df.groupby('cluster').agg(
        count=('decimalLatitude', 'count'),
        avg_lat=('decimalLatitude', 'mean'),
        avg_lon=('decimalLongitude', 'mean'),
        min_lat=('decimalLatitude', 'min'),
        max_lat=('decimalLatitude', 'max'),
        min_lon=('decimalLongitude', 'min'),
        max_lon=('decimalLongitude', 'max')
    ).reset_index()
    
    # Calculate cluster radius (approximate, in km)
    def calculate_radius(group):
        # Calculate the maximum distance from any point to the centroid
        if len(group) <= 1:
            return 0
        centroid = (group['decimalLatitude'].mean(), group['decimalLongitude'].mean())
        max_dist = 0
        for _, row in group.iterrows():
            dist = hs.haversine(centroid, (row['decimalLatitude'], row['decimalLongitude']), unit=Unit.KILOMETERS)
            max_dist = max(max_dist, dist)
        return max_dist
    
    # Calculate radius for each cluster
    cluster_radii = {}
    for cluster_id in selected_df['cluster'].unique():
        if cluster_id == -1:  # Skip noise points
            cluster_radii[cluster_id] = 0
            continue
        group = selected_df[selected_df['cluster'] == cluster_id]
        cluster_radii[cluster_id] = calculate_radius(group)
    
    # Add radius to cluster stats
    cluster_stats['radius_km'] = cluster_stats['cluster'].map(cluster_radii)
    
    # Calculate cluster area and density
    cluster_stats['area_km2'] = np.pi * cluster_stats['radius_km'] ** 2
    cluster_stats['density'] = cluster_stats['count'] / cluster_stats['area_km2'].replace(0, 1)  # Avoid division by zero
    
    # For noise points (cluster -1), set density to 0
    cluster_stats.loc[cluster_stats['cluster'] == -1, 'density'] = 0
    
    # Add environmental and taxonomic stats where available
    env_columns = ['depth', 'bathymetry', 'shoredistance', 'sst', 'sss']
    for col in env_columns:
        if col in selected_df.columns:
            aggs = selected_df.groupby('cluster')[col].agg(['mean', 'min', 'max', 'std']).reset_index()
            aggs.columns = ['cluster', f'{col}_mean', f'{col}_min', f'{col}_max', f'{col}_std']
            cluster_stats = pd.merge(cluster_stats, aggs, on='cluster', how='left')
    
    # Add taxonomic diversity stats
    taxonomy_columns = ['phylum', 'class', 'order', 'family', 'genus', 'species']
    for col in taxonomy_columns:
        if col in selected_df.columns:
            # Count unique taxonomic groups per cluster
            unique_counts = selected_df.groupby('cluster')[col].nunique().reset_index()
            unique_counts.columns = ['cluster', f'unique_{col}_count']
            cluster_stats = pd.merge(cluster_stats, unique_counts, on='cluster', how='left')
            
            # Calculate Shannon diversity index for taxonomic groups
            def shannon_diversity(group, col_name):
                if len(group) <= 1:
                    return 0
                taxa_counts = group[col_name].value_counts()
                proportions = taxa_counts / taxa_counts.sum()
                return -np.sum(proportions * np.log(proportions))
            
            diversity = []
            for cluster_id in cluster_stats['cluster']:
                group = selected_df[selected_df['cluster'] == cluster_id]
                if len(group) <= 1:
                    diversity.append(0)
                else:
                    diversity.append(shannon_diversity(group, col))
                    
            cluster_stats[f'{col}_diversity'] = diversity
    
    # Prepare output for Tableau
    # 1. Original data with cluster assignments
    df_for_tableau = selected_df.copy()
    df_for_tableau['is_noise'] = df_for_tableau['cluster'] == -1
    df_for_tableau['cluster_name'] = df_for_tableau['cluster'].apply(lambda x: 'Noise' if x == -1 else f'Cluster {x}')
    
    # 2. Rename clusters in stats dataframe
    cluster_stats['cluster_name'] = cluster_stats['cluster'].apply(lambda x: 'Noise' if x == -1 else f'Cluster {x}')
    
    # Save to CSV for Tableau
    print(f"Saving clustered data to {output_path}...")
    df_for_tableau.to_csv(output_path, index=False)
    
    # Save cluster stats
    print(f"Saving cluster statistics to {stats_output_path}...")
    cluster_stats.to_csv(stats_output_path, index=False)
    
    # Generate basic preview map
    print("Generating preview map...")
    plt.figure(figsize=(12, 8))
    
    # Plot clusters
    regular_points = df_for_tableau[~df_for_tableau['is_noise']]
    noise_points = df_for_tableau[df_for_tableau['is_noise']]
    
    # Plot a max of 20,000 points for performance
    if len(regular_points) > 20000:
        sample_size = 20000
        regular_sample = regular_points.sample(sample_size)
    else:
        regular_sample = regular_points
        
    # Plot noise (max 5,000 points)
    if len(noise_points) > 5000:
        noise_sample = noise_points.sample(5000)
    else:
        noise_sample = noise_points
    
    # Plot clusters with different colors
    unique_clusters = regular_sample['cluster_name'].unique()
    for cluster in unique_clusters:
        cluster_points = regular_sample[regular_sample['cluster_name'] == cluster]
        plt.scatter(
            cluster_points['decimalLongitude'], 
            cluster_points['decimalLatitude'],
            s=10,
            alpha=0.6,
            label=f"{cluster} (n={len(cluster_points)})"
        )
    
    # Plot noise points
    if len(noise_sample) > 0:
        plt.scatter(
            noise_sample['decimalLongitude'], 
            noise_sample['decimalLatitude'],
            s=3,
            alpha=0.2,
            color='gray',
            label=f'Noise (n={len(noise_points)})'
        )
    
    plt.title(f'DBSCAN Clustering of OBIS Marine Data: {n_clusters} clusters identified')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Save the plot
    plot_path = "C:/minor project/output/obis_clusters_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Preview plot saved to {plot_path}")
    
    print(f"\n✅ OBIS data processing and DBSCAN clustering complete!")
    print(f"The data is ready for visualization in Tableau.")

except Exception as e:
    import traceback
    print(f"Error in OBIS data processing: {str(e)}")
    traceback.print_exc()