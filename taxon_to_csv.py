import pandas as pd

# Path to your .txt file
input_path = r'C:\minor project\output\cleaned_taxon.txt'
output_path = r'C:\minor project\output\cleaned_taxon.csv'

# Try reading it with tab or other common delimiter
try:
    # Auto-detect the delimiter (assumes it's tab or something else)
    df = pd.read_csv(input_path, sep=None, engine='python')
except Exception as e:
    print("Error reading file:", e)
    # You can manually try a specific delimiter here, like '\t' or '|'
    df = pd.read_csv(input_path, sep='\t')

# Show the first few rows (for confirmation)
print(df.head())

# Export to CSV for Tableau
df.to_csv(output_path, index=False, encoding='utf-8')
print(f"Converted successfully to: {output_path}")