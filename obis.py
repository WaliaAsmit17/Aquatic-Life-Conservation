import requests
import pandas as pd
import time

def fetch_obis_data(size=1000, from_offset=0):
    """
    Fetches data from OBIS API with pagination.
    """
    url = "https://api.obis.org/v3/occurrence"
    params = {
        "size": size,     # 🔥 Must use 'size', not 'limit' (OBIS uses ElasticSearch syntax)
        "from": from_offset
    }

    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json().get("results", [])
    else:
        print(f"❌ Error {response.status_code} at offset {from_offset}")
        return []

def save_to_csv(data, filename="obis_80000.csv"):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"✅ Saved {len(df)} records to {filename}")

def main():
    all_data = []
    offset = 0
    size = 1000
    target_records = 80000

    print(f"🚀 Starting OBIS data fetch for {target_records} records...\n")

    while len(all_data) < target_records:
        print(f"📥 Fetching {offset} to {offset + size}...")
        results = fetch_obis_data(size=size, from_offset=offset)

        if not results:
            print("⚠️ No more data available or request failed.")
            break

        all_data.extend(results)

        if len(results) < size:
            print("ℹ️ Fewer results returned than requested — possibly end of data.")
            break

        offset += size
        time.sleep(1)

    if len(all_data) > target_records:
        all_data = all_data[:target_records]

    save_to_csv(all_data)

if __name__ == "__main__":
    main()
