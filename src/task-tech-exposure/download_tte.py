import os
import pandas as pd
import numpy as np
import requests
import zipfile
import json
import time
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, Set, Optional, List

class TTEDownloader:
    """
    Improved downloader for TTE dataset from Zenodo with proper API usage,
    rate limiting, and error handling.
    """
    
    def __init__(self, local_output_dir: str):
        self.local_output_dir = Path(local_output_dir)
        self.local_output_dir.mkdir(parents=True, exist_ok=True)
        self.local_manifest_path = self.local_output_dir / "tte/zenodo_manifest.json"
        
        # Proper headers for Zenodo
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds between requests
    
    def _rate_limit(self):
        """Ensure we don't make requests too quickly."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _resolve_doi_to_zenodo_id(self, doi: str) -> str:
        """Extract Zenodo record ID from DOI."""
        if "zenodo." not in doi:
            raise ValueError("Invalid: Provided URL is not a Zenodo DOI.")
        try:
            record_id = doi.split('zenodo.')[-1].rstrip('/')
            if not record_id.isdigit():
                raise ValueError("Could not parse Zenodo record ID from DOI.")
            return record_id
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid Zenodo DOI format: {doi}") from e
    
    def _get_zenodo_manifest(self, record_id: str) -> Dict:
        """
        Fetch the manifest from Zenodo API.
        Uses the proper API URLs from the response.
        """
        api_url = f"https://zenodo.org/api/records/{record_id}"
        
        print(f"Fetching manifest from Zenodo API...")
        self._rate_limit()
        
        try:
            response = requests.get(
                api_url,
                headers=self.headers,
                timeout=30,
                allow_redirects=True
            )
            response.raise_for_status()
            data = response.json()
            
            # Build manifest using the URLs provided by Zenodo
            manifest = {
                "version": data.get("metadata", {}).get("version", "unknown"),
                "files": {}
            }
            
            for file_info in data.get("files", []):
                filename = file_info.get("key")
                # CRITICAL: Use the URL provided by Zenodo's API
                download_url = file_info.get("links", {}).get("self")
                
                if not filename or not download_url:
                    print(f"Warning: Skipping file with missing info: {file_info}")
                    continue
                
                is_zip = filename.endswith(".zip")
                is_json_manifest = filename == "dataset_manifest.json"
                
                manifest["files"][filename] = {
                    "url": download_url,
                    "size": file_info.get("size", 0),
                    "checksum": file_info.get("checksum", ""),
                    "compressed": is_zip,
                    "is_manifest": is_json_manifest
                }
            
            if not manifest["files"]:
                raise Exception(f"Zenodo record {record_id} does not contain any downloadable files.")
            
            print(f"Found {len(manifest['files'])} files in Zenodo record")
            return manifest
            
        except requests.RequestException as e:
            raise Exception(f"Failed to download Zenodo API manifest for record {record_id}: {e}")
    
    def _load_local_manifest(self) -> Optional[Dict]:
        """Load local manifest if it exists."""
        if self.local_manifest_path.exists():
            try:
                with open(self.local_manifest_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not read local manifest: {e}")
        return None
    
    def _save_local_manifest(self, manifest: Dict):
        """Save manifest to local disk."""
        with open(self.local_manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def _determine_years_to_download(
        self, 
        remote_manifest: Dict, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict:
        """
        Filter manifest to only include files for years in the date range.
        Also includes non-year files (models, onet, manifest, etc.)
        """
        years_in_range = set(range(start_date.year, end_date.year + 1))
        manifest_to_download = {
            "version": remote_manifest.get("version"),
            "files": {}
        }
        
        year_pattern = re.compile(r'tte_(\d{4})\.zip$')
        
        for filename, file_info in remote_manifest['files'].items():
            match = year_pattern.match(filename)
            
            if match:
                # This is a year file
                file_year = int(match.group(1))
                if file_year in years_in_range:
                    manifest_to_download['files'][filename] = file_info
                    print(f"  Including year file: {filename}")
            else:
                # Non-year files (models, onet, manifest) - always include
                manifest_to_download['files'][filename] = file_info
                print(f"  Including support file: {filename}")
        
        return manifest_to_download
    
    def _determine_files_to_download(
        self,
        remote_manifest: Dict,
        local_manifest: Optional[Dict],
        force_update: bool = False
    ) -> Dict:
        """
        Determine which files need to be downloaded.
        Skips files that already exist locally unless force_update is True.
        """
        files_to_download = {}
        remote_version = remote_manifest.get("version")
        local_version = local_manifest.get("version") if local_manifest else None
        
        if force_update:
            print("Force update enabled - will download all files in range")
            files_to_download = remote_manifest["files"]
        elif local_version is None or remote_version != local_version:
            print(f"Version mismatch (local: {local_version}, remote: {remote_version})")
            print("Will download all files in range")
            files_to_download = remote_manifest["files"]
        else:
            print(f"Version match ({remote_version}) - checking for missing files...")
            for filename, file_info in remote_manifest["files"].items():
                if filename.endswith('.zip'):
                    # For zip files, check if extracted directory exists
                    dir_name = filename.replace('.zip', '')
                    local_dir = self.local_output_dir / "tte" / dir_name
                    if not local_dir.exists():
                        print(f"  Missing extracted data for {filename}")
                        files_to_download[filename] = file_info
                else:
                    # For non-zip files, check if file exists
                    local_file = self.local_output_dir / "tte" / filename
                    if not local_file.exists():
                        print(f"  Missing file: {filename}")
                        files_to_download[filename] = file_info
        
        return {"version": remote_version, "files": files_to_download}
    
    def _download_and_extract(self, manifest: Dict):
        """
        Download and extract files from Zenodo.
        Uses proper error handling and rate limiting.
        """
        tte_path = self.local_output_dir / 'tte'
        tte_path.mkdir(exist_ok=True)
        
        total_files = len(manifest['files'])
        
        for idx, (filename, file_info) in enumerate(manifest['files'].items(), 1):
            print(f"\n[{idx}/{total_files}] Processing {filename}")
            
            # Check if we should skip
            if filename.endswith('.zip'):
                dir_name = filename.replace('.zip', '')
                local_dir = tte_path / dir_name
                if local_dir.exists():
                    print(f"  ✓ Already extracted, skipping")
                    continue
            else:
                local_file = tte_path / filename
                if local_file.exists():
                    print(f"  ✓ Already exists, skipping")
                    continue
            
            # Download the file
            file_url = file_info['url']
            file_size = file_info.get('size', 0)
            is_zip = file_info.get("compressed", False)
            is_json_manifest = file_info.get("is_manifest", False)
            
            print(f"  Downloading ({file_size / 1024 / 1024:.1f} MB)...")
            
            # Temporary download path
            temp_file = tte_path / f"{filename}.download"
            
            try:
                self._rate_limit()
                
                response = requests.get(
                    file_url,
                    headers=self.headers,
                    stream=True,
                    timeout=300,
                    allow_redirects=True
                )
                response.raise_for_status()
                
                # Download with progress
                downloaded = 0
                with open(temp_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if downloaded % (10 * 1024 * 1024) == 0:  # Every 10MB
                                print(f"  Downloaded {downloaded / 1024 / 1024:.1f} MB...")
                
                # Move to final location
                final_file = tte_path / filename
                temp_file.rename(final_file)
                
                print(f"  ✓ Download complete")
                
                # Extract if zip
                if is_zip:
                    print(f"  Extracting...")
                    try:
                        with zipfile.ZipFile(final_file, 'r') as zip_ref:
                            zip_ref.extractall(tte_path)
                        # Remove zip file after extraction
                        final_file.unlink()
                        print(f"  ✓ Extracted and removed zip")
                    except zipfile.BadZipFile as e:
                        print(f"  ✗ Bad zip file: {e}")
                        # Keep the file for inspection
                        continue
                
                # Process manifest file if needed
                if is_json_manifest:
                    print(f"  Reading dataset manifest...")
                    try:
                        with open(final_file, "r") as f:
                            remote_json_manifest = json.load(f)
                            # You can store this if needed
                            print(f"  ✓ Manifest contains {len(remote_json_manifest)} entries")
                    except Exception as e:
                        print(f"  Warning: Could not parse manifest: {e}")
                
            except requests.exceptions.HTTPError as e:
                print(f"  ✗ HTTP Error: {e}")
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                print(f"  ✗ Unexpected error: {e}")
                if temp_file.exists():
                    temp_file.unlink()
    
    def download_tte_data(
        self,
        from_date: str,
        to_date: str,
        remote_url: str,
        force_update: bool = False
    ):
        """
        Main method to download and process TTE data.
        
        Args:
            from_date: Start date in format 'YYYY-MM-DD'
            to_date: End date in format 'YYYY-MM-DD'
            remote_url: Zenodo DOI URL
            force_update: If True, re-download all files even if they exist
        """
        print(f"\n{'='*60}")
        print(f"TTE Dataset Downloader")
        print(f"{'='*60}")
        print(f"Date range: {from_date} to {to_date}")
        print(f"Source: {remote_url}")
        print(f"Output directory: {self.local_output_dir}")
        print(f"{'='*60}\n")
        
        # Parse dates
        start_date = pd.to_datetime(from_date)
        end_date = pd.to_datetime(to_date)
        
        # Get Zenodo record ID
        record_id = self._resolve_doi_to_zenodo_id(remote_url)
        
        # Fetch remote manifest
        remote_manifest = self._get_zenodo_manifest(record_id)
        
        # Load local manifest
        local_manifest = self._load_local_manifest()
        
        # Filter to years in range
        year_filtered_manifest = self._determine_years_to_download(
            remote_manifest, start_date, end_date
        )
        
        # Determine what needs downloading
        manifest_to_download = self._determine_files_to_download(
            year_filtered_manifest, local_manifest, force_update
        )
        
        # Download files
        if not manifest_to_download['files']:
            print("\n✓ All files are already present and up-to-date.")
        else:
            print(f"\nDownloading {len(manifest_to_download['files'])} files...")
            self._download_and_extract(manifest_to_download)

        # Save manifest
        self._save_local_manifest(remote_manifest)
        
        print(f"\n{'='*60}")
        print("✓ Download and extraction complete")
        print(f"{'='*60}\n")


def download_tte(
    local_output_dir: str,
    from_date: str = None,
    to_date: str = None,
    remote_url: str = "https://doi.org/10.5281/zenodo.17643646",
    force_update: bool = False
):
    """
    Convenience function to download TTE dataset.
    
    Args:
        from_date: Start date in format 'YYYY-MM-DD'
        to_date: End date in format 'YYYY-MM-DD'
        remote_url: Zenodo DOI URL
        local_output_dir: Directory to save data
        force_update: If True, re-download all files
    """
    downloader = TTEDownloader(local_output_dir)
    downloader.download_tte_data(from_date, to_date, remote_url, force_update)
