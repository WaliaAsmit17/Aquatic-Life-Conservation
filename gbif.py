import requests
import pandas as pd
import time

def fetch_gbif_data(limit=300, offset=0):
    """
    Fetch GBIF occurrence records with coordinates.
    """
    url = "https://api.gbif.org/v1/occurrence/search"
    params = {
        "limit": limit,
        "offset": offset,
        "hasCoordinate": "true",  # Ensures we get lat/lon
        "basisOfRecord": "HUMAN_OBSERVATION"  # You can remove this if you want all records
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get("results", [])
    else:
        print(f"⚠️ Error {response.status_code} at offset {offset}")
        return []

def save_to_csv(data, filename="gbif_80000_raw.csv"):
    df = pd.DataFrame(data)

    # Filter to useful columns
    columns_to_keep = [
        'species', 'decimalLatitude', 'decimalLongitude',
        'eventDate', 'country', 'kingdom', 'phylum',
        'class', 'order', 'family', 'genus'
    ]
    df = df[[col for col in columns_to_keep if col in df.columns]]

    df.to_csv(filename, index=False)
    print(f"✅ Saved {len(df)} records to {filename}")

def main():
    all_data = []
    offset = 0
    limit = 1000
    target_records = 50000

    while len(all_data) < target_records:
        print(f"📥 Fetching records {offset} to {offset + limit}...")
        results = fetch_gbif_data(limit=limit, offset=offset)
        if not results:
            print("🚫 No more data or request failed.")
            break
        all_data.extend(results)
        offset += limit
        time.sleep(1)  # Be polite to GBIF servers

    save_to_csv(all_data)

if __name__ == "__main__":
    main()
