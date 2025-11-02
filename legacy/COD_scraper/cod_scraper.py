import os
import requests
import pandas as pd

df = pd.read_csv("matching_compositions.csv")
cod_id_list = df['cod_id'].astype(str).to_list()

downloaded_file = "downloaded_files.txt"

# Read previously downloaded files if the tracking file exists
if os.path.exists(downloaded_file):
    with open(downloaded_file, "r") as f:
        downloaded_ids = set(f.read().splitlines())
else:
    downloaded_ids = set()

cod_id_list = [cod_id for cod_id in cod_id_list if cod_id not in downloaded_ids]

# Define the download function
def download_file(cod_id):
    url = f"https://example.com/files/{cod_id}.cif"  # Replace with actual URL format
    file_path = f"downloads/{cod_id}.cif"  # Save to a 'downloads' folder

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        os.makedirs("downloads", exist_ok=True)  # Ensure directory exists

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Mark file as downloaded
        with open(downloaded_file, "a") as f:
            f.write(f"{cod_id}\n")

        print(f"Downloaded: {cod_id}")

    except requests.RequestException as e:
        print(f"Failed to download {cod_id}: {e}")

# Download files
for cod_id in cod_id_list:
    download_file(cod_id)

print("Download process complete.")
