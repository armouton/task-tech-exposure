import os
import pandas as pd
import numpy as np
import requests
import zipfile
import json
import shutil
from datetime import datetime
from pathlib import Path
import tempfile
from urllib.parse import urljoin
import re

url = 'https://httpbin.org/headers' # A website that echoes back the headers sent
custom_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'

headers = {
    'User-Agent': custom_user_agent
}
response = requests.get(url, headers=headers)
print(response.json()['headers']['User-Agent'])

class TTEDownloader:
    def __init__(self, local_output_dir):
        self.local_output_dir = Path(local_output_dir)
        self.local_output_dir.mkdir(parents=True, exist_ok=True)
        self.local_manifest_path = self.local_output_dir / "local_manifest.json"

    def _resolve_doi_to_zenodo_id(self, doi):
        if "zenodo." not in doi:
            raise ValueError("Invalid: Provided URL is not a Zenodo DOI.")
        try:
            record_id = doi.split('zenodo.')[-1]
            if not record_id.isdigit():
                raise ValueError("Could not parse Zenodo record ID from DOI.")
            return record_id
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid Zenodo DOI format: {doi}") from e

    def _get_zenodo_manifest(self, record_id):
        api_url = f"https://zenodo.org/api/records/{record_id}"
        try:
           
            response = requests.get(api_url, headers=headers, timeout=120, allow_redirects=True)
            

            response.raise_for_status()
            data = response.json()

            manifest = {"version": data.get("version"), "files": {}}
            for file_info in data.get("files", []):
                filename = file_info.get("key")
                download_url = f"https://zenodo.org/records/{record_id}/files/{filename}"
                if not filename or not download_url:
                    continue
                is_zip = filename.endswith(".zip")
                is_json_manifest = filename == "dataset_manifest.json"
                manifest["files"][filename] = {
                    "url": download_url,
                    "compressed": is_zip,
                    "is_manifest": is_json_manifest
                }

            if not manifest["files"]:
                raise Exception(f"Zenodo record {record_id} does not contain any downloadable files.")
            return manifest
        except requests.RequestException as e:
            raise Exception(f"Failed to download Zenodo API manifest for record {record_id}: {e}")

    def update_manifest_parameters(self, sbert_model, tech_cutoff):
        manifest_path = self.local_manifest_path
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        else:
            manifest = {"version": "0'0", "files": {}}
        manifest["parameters"] = {"sbert_model": sbert_model, "tech_cutoff": tech_cutoff}
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _load_local_manifest(self):
        if self.local_manifest_path.exists():
            try:
                with open(self.local_manifest_path, 'r') as f:
                    return json.load(f)
            except Exception:
                print("Warning: Could not read local manifest.")
        return None

    def _save_local_manifest(self, manifest):
        with open(self.local_manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    def _determine_years_to_download(self, remote_manifest, start_date, end_date):
        years_in_range = set(range(start_date.year, end_date.year + 1))
        manifest_to_download = {"version": remote_manifest.get("version"), "files": {}}
        index_pattern = re.compile(r'(\d+)\.zip$')
        for filename, file_info in remote_manifest['files'].items():
            match = index_pattern.search(filename)
            if match:
                try:
                    file_year = int(match.group(1))
                    if file_year in years_in_range:
                        manifest_to_download['files'][filename] = file_info
                except (ValueError, IndexError):
                    continue
            else:
                manifest_to_download['files'][filename] = file_info
        return manifest_to_download

    def _determine_files_to_download(self, remote_manifest, local_manifest, force_update=False):
        files_to_download = {}
        remote_version = remote_manifest.get("version")
        local_version = local_manifest.get("version") if local_manifest else None
        if force_update or (local_version is None or remote_version != local_version):
            files_to_download = remote_manifest["files"]
        else:
            print(f"Checking for missing files...")
            for filename, file_info in remote_manifest["files"].items():
                local_file = self.local_output_dir / "tte" / filename
                if not local_file.exists():
                    print(f"Missing {filename} downloading.")
                    files_to_download[filename] = file_info
        return {"version": remote_version, "files": files_to_download}

    def _download_and_extract(self, manifest, base_path):
        tte_path = self.local_output_dir / 'tte'
        tte_path.mkdir(exist_ok=True)
        for filename, file_info in manifest['files'].items():
            local_file_path = tte_path / filename
            if local_file_path.exists():
                print(f"Skipping {filename}, already exists.")
                continue
            file_url = file_info['url']
            is_zip = file_info.get("compressed", False)
            is_json_manifest = file_info.get("is_manifest", False)
            print(f"Downloading {filename}")
            try:
                headers = {
            'User-Agent': custom_user_agent
            }
                response = requests.get(file_url, stream=True, allow_redirects=True)
                response.raise_for_status()
                with open(local_file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                if is_zip:
                    print(f"Extracting {filename}")
                    with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
                        zip_ref.extractall(tte_path)
                        os.remove(local_file_path)
                if is_json_manifest:
                    print(f"Loaded remote dataset_manifest.json")
                    with open(local_file_path, "r") as f:
                        remote_json_manifest = json.load(f)
                        manifest.update(remote_json_manifest)
            except requests.exceptions.HTTPError as e:
                print(f"An unexpected error occurred with {filename}: {e}")

    def _load_and_merge_data(self, temp_path, manifest_files=None):
        merged_patents = []
        merged_matches = []
        merged_embeddings = []
        for filename, file_info in (manifest_files or {}).items():
            if file_info.get("compressed"):
                year_match = re.search(r'(\d+)\.zip$', filename)
                if year_match:
                    year = year_match.group(1)
                    year_path = temp_path / f"tte_{year}"
                    if year_path.exists():
                        try:
                            patents_files = list(year_path.glob('**/patent_text_*'))
                            matches_files = list(year_path.glob('**/matches_*'))
                            embedding_files = list(year_path.glob('**/patent_embed_*'))
                            if patents_files:
                                print(f"Loading patents file: {patents_files[0]}")
                                patents = pd.read_csv(patents_files[0])
                                merged_patents.append(patents)
                            if matches_files:
                                print(f"Loading matches file: {matches_files[0]}")
                                matches = pd.read_csv(matches_files[0])
                                merged_matches.append(matches)
                            if embedding_files:
                                embeddings = np.load(embedding_files[0])
                                merged_embeddings.append(embeddings)
                        except Exception as e:
                            print(f"Warning: Could not load year data from {year_path}: {e}")
        if not merged_patents:
            print("No patent data loaded from downloaded files.")
            return None
        all_patents = pd.concat(merged_patents, ignore_index=True)
        all_matches = pd.concat(merged_matches, ignore_index=True)
        all_embeddings = np.vstack(merged_embeddings)
        return {'patents': all_patents, 'matches': all_matches, 'embeddings': all_embeddings}

    def _filter_and_deduplicate(self, merged_data, start_date, end_date):
        if not merged_data:
            return
        merged_data['patents']['date_published'] = pd.to_datetime(merged_data['patents']['date_published'])
        date_mask = (merged_data['patents']['date_published'] >= start_date) & (merged_data['patents']['date_published'] <= end_date)
        filtered_patents = merged_data['patents'][date_mask].copy()
        if len(filtered_patents) == 0:
            raise Exception(f"No patents found in date range {start_date.date()} to {end_date.date()}")
        filtered_patents = filtered_patents.drop_duplicates(subset=['patent_id'], keep='first')
        patent_ids = set(filtered_patents['patent_id'])
        filtered_matches = merged_data['matches'][merged_data['matches']['patent_id'].isin(patent_ids)].copy()
        return {'patents': filtered_patents.reset_index(drop=True), 'matches': filtered_matches.reset_index(drop=True)}

    def _save_merged_dataset(self, filtered_data):
        if not filtered_data:
            print("No data to save.")
            return
        (self.local_output_dir / 'patents').mkdir(exist_ok=True)
        filtered_data['patents'].to_csv(self.local_output_dir / 'patents' / 'patent_text.csv', index=False)
        filtered_data['matches'].to_csv(self.local_output_dir / 'matches.csv', index=False)

    def download_tte_data(self, from_date, to_date, remote_url, force_update=False):
        print(f"Downloading TTE data from {from_date} to {to_date}")
        start_date = pd.to_datetime(from_date)
        end_date = pd.to_datetime(to_date)
        record_id = self._resolve_doi_to_zenodo_id(remote_url)
        remote_manifest = self._get_zenodo_manifest(record_id)
        local_manifest = self._load_local_manifest()
        year_filtered_manifest = self._determine_years_to_download(remote_manifest, start_date, end_date)
        manifest_to_download = self._determine_files_to_download(year_filtered_manifest, local_manifest, force_update)
        if not manifest_to_download['files']:
            print("All files are already present and up-to-date.")
        else:
            self._download_and_extract(manifest_to_download, self.local_output_dir)
        merged_data = self._load_and_merge_data(self.local_output_dir / 'tte', manifest_to_download['files'])
        if merged_data:
            filtered_data = self._filter_and_deduplicate(merged_data, start_date, end_date)
            self._save_merged_dataset(filtered_data)
        self._save_local_manifest(remote_manifest)
        print("Download and merge complete")

def download_tte_dataset(from_date, to_date, remote_url, local_output_dir, force_update=False):

    downloader = TTEDownloader(local_output_dir)
    downloader.download_tte_data(from_date, to_date, remote_url, force_update)

if __name__ == "__main__":
    download_tte_dataset(
        from_date="2001-01-01",
        to_date="2025-12-31",
        remote_url="https://doi.org/10.5281/zenodo.18235818",
        local_output_dir="./tte_dataset",
        force_update=False
    )
