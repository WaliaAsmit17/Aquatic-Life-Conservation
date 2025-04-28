import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# File paths
input_path = "C:/minor project/output/obis_80000.csv"
output_path = "C:/minor project/output/species_temperature_analysis.csv"

try:
    print(f"Reading OBIS data from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    
    print(f"Loaded {len(df)} records")
    
    # Check if sea surface temperature (sst) column exists
    if 'sst' not in df.columns:
        print("Warning: Sea surface temperature (sst) column not found. Checking for alternatives...")
        
        # Look for alternative temperature columns
        temp_columns = [col for col in df.columns if 'temp' in col.lower()]
        if temp_columns:
            print(f"Found alternative temperature columns: {temp_columns}")
            temp_column = temp_columns[0]  # Use the first one
        else:
            raise ValueError("No temperature data found in the dataset. Cannot proceed with temperature analysis.")
    else:
        temp_column = 'sst'
        
    print(f"Using {temp_column} for temperature analysis")
    
    # Verify taxonomic group columns
    taxonomic_levels = ['phylum', 'class', 'order', 'family', 'genus', 'species']
    available_taxa = [level for level in taxonomic_levels if level in df.columns]
    
    if not available_taxa:
        raise ValueError("No taxonomic classification columns found in the dataset.")
    
    print(f"Available taxonomic levels: {available_taxa}")
    
    # Select relevant columns and clean data
    columns_to_keep = [temp_column] + available_taxa + ['decimalLatitude', 'decimalLongitude']
    
    # Filter to rows that have temperature data
    filtered_df = df[columns_to_keep].dropna(subset=[temp_column])
    print(f"After filtering for temperature data: {len(filtered_df)} records")
    
    # Initialize results list to store temperature statistics for each taxonomic group
    results = []
    
    # Process each taxonomic level, from most specific to most general
    for tax_level in reversed(available_taxa):
        print(f"Analyzing temperature data for taxonomic level: {tax_level}")
        
        # Group by the taxonomic level and calculate temperature statistics
        # Only include groups with enough data points (10+)
        group_counts = filtered_df.groupby(tax_level).size()
        valid_groups = group_counts[group_counts >= 10].index
        
        # For each valid taxonomic group, calculate temperature statistics
        for group_name in valid_groups:
            group_data = filtered_df[filtered_df[tax_level] == group_name]
            
            # Skip if NaN or empty group name
            if pd.isna(group_name) or group_name == '':
                continue
                
            # Calculate temperature statistics
            temp_data = group_data[temp_column].dropna()
            
            if len(temp_data) < 10:
                continue  # Skip if too few points after dropping NaNs
                
            # Basic statistics
            mean_temp = temp_data.mean()
            median_temp = temp_data.median()
            min_temp = temp_data.min()
            max_temp = temp_data.max()
            std_temp = temp_data.std()
            
            # Calculate 5th and 95th percentiles to avoid outlier influence
            p05 = np.percentile(temp_data, 5)
            p95 = np.percentile(temp_data, 95)
            
            # Calculate the mode (most frequent temperature)
            try:
                mode_temp = stats.mode(temp_data, keepdims=False)[0]
            except:
                mode_temp = median_temp  # Fallback if mode calculation fails
            
            # Calculate frequency distribution for histogram
            hist, bin_edges = np.histogram(temp_data, bins=20)
            peak_bin_index = np.argmax(hist)
            peak_temp_range_min = bin_edges[peak_bin_index]
            peak_temp_range_max = bin_edges[peak_bin_index + 1]
            peak_temp = (peak_temp_range_min + peak_temp_range_max) / 2
            
            # Get optimal temperature range (where most observations occur)
            # We'll define this as the range containing the middle 50% of observations
            q1 = np.percentile(temp_data, 25)
            q3 = np.percentile(temp_data, 75)
            
            # Add results to our list
            results.append({
                'taxonomic_level': tax_level,
                'taxonomic_group': group_name,
                'observation_count': len(temp_data),
                'mean_temperature': mean_temp,
                'median_temperature': median_temp,
                'mode_temperature': mode_temp,
                'peak_temperature': peak_temp,  # Temperature with most observations
                'min_temperature': min_temp,
                'max_temperature': max_temp,
                'std_temperature': std_temp,
                'p05_temperature': p05,
                'p95_temperature': p95,
                'q1_temperature': q1,  # 25th percentile - lower optimal bound
                'q3_temperature': q3,  # 75th percentile - upper optimal bound
                'optimal_temp_min': q1,  # Lower bound of optimal range
                'optimal_temp_max': q3,  # Upper bound of optimal range
                'temp_range_width': max_temp - min_temp,
                'optimal_range_width': q3 - q1
            })
            
            # Add higher taxonomy information if available
            for higher_level in available_taxa:
                if higher_level != tax_level and available_taxa.index(higher_level) < available_taxa.index(tax_level):
                    # Get the most common higher taxon for this group
                    higher_taxa = group_data[higher_level].value_counts().index
                    if len(higher_taxa) > 0:
                        results[-1][f'parent_{higher_level}'] = higher_taxa[0]
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by observation count to prioritize groups with more data
    results_df = results_df.sort_values('observation_count', ascending=False)
    
    # Save the results
    print(f"Saving temperature analysis for {len(results_df)} taxonomic groups to {output_path}")
    results_df.to_csv(output_path, index=False)
    
    # Create a summary with top 20 species/groups by observation count
    top_groups = results_df.head(20)
    
    # Generate visualization of temperature ranges for top groups
    plt.figure(figsize=(12, 10))
    
    # Plot range for each of the top groups
    y_positions = range(len(top_groups))
    plt.barh(y_positions, 
             top_groups['optimal_range_width'], 
             left=top_groups['optimal_temp_min'],
             height=0.7, 
             color='skyblue', 
             alpha=0.8)
    
    # Plot the peak temperature point
    plt.scatter(top_groups['peak_temperature'], 
                y_positions, 
                color='darkblue', 
                s=80, 
                zorder=3, 
                label='Peak Temperature')
    
    # Add group names on the y-axis
    plt.yticks(y_positions, 
               [f"{row['taxonomic_group'][:25]} ({row['observation_count']} obs)" 
                for _, row in top_groups.iterrows()])
    
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.title('Optimal Temperature Ranges for Top Marine Species Groups', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Save the plot
    plot_path = "C:/minor project/output/species_temperature_ranges.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Preview plot saved to {plot_path}")
    
    # Create JSON data for more advanced Tableau custom visualization
    # This can be imported into Tableau for interactive visualizations
    viz_data = []
    for _, row in results_df.iterrows():
        # Create histogram data (simplified for brevity)
        group_data = filtered_df[filtered_df[row['taxonomic_level']] == row['taxonomic_group']]
        temp_data = group_data[temp_column].dropna().tolist()
        
        entry = {
            'taxonomic_level': row['taxonomic_level'],
            'taxonomic_group': row['taxonomic_group'],
            'count': row['observation_count'],
            'optimal_min': row['optimal_temp_min'],
            'optimal_max': row['optimal_temp_max'],
            'peak': row['peak_temperature'],
            'all_temperatures': temp_data[:1000]  # Limit to 1000 points for file size
        }
        viz_data.append(entry)
    
    # Save JSON for advanced visualization
    import json
    json_path = "C:/minor project/output/temperature_distributions.json"
    with open(json_path, 'w') as f:
        json.dump(viz_data[:100], f)  # Limit to top 100 groups
    
    print(f"\n✅ Temperature analysis complete! Data is ready for visualization in Tableau.")
    print("Suggested Tableau visualizations:")
    print("1. Bar chart showing optimal temperature range by species")
    print("2. Scatter plot of peak temperature vs latitude")
    print("3. Heat map combining temperature preferences with geographic distribution")
    print("4. Box plots of temperature ranges by taxonomic level")
    print("5. Density plots showing temperature distribution curves by species")

except Exception as e:
    import traceback
    print(f"Error in temperature analysis: {str(e)}")
    traceback.print_exc()