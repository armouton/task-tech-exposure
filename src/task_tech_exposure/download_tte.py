import os
import requests
import zipfile
import json
import time
from pathlib import Path
import re
from typing import Dict, Optional

# DOWNLOADER FOR TTE DATASET FROM ZENODO
class TTEDownloader:
    """
    Improved downloader for TTE dataset from Zenodo with proper API usage,
    rate limiting, and error handling.
    """
    
    def __init__(self, path_to_data: str):
        self.path_to_data = Path(path_to_data)
        self.path_to_data.mkdir(parents=True, exist_ok=True)
        self.local_manifest_path = self.path_to_data / "tte/zenodo_manifest.json"
        
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

    def _load_local_dataset_manifest(self) -> Optional[Dict]:
        """Load the dataset_manifest.json shipped with the dataset, if present."""
        path = self.path_to_data / "tte" / "dataset_manifest.json"
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not read local dataset manifest: {e}")
        return None

    def _fetch_remote_dataset_manifest(self, zenodo_api_manifest: Dict) -> Optional[Dict]:
        """Download and parse the dataset_manifest.json from the Zenodo record."""
        file_info = zenodo_api_manifest['files'].get('dataset_manifest.json')
        if not file_info:
            print("Warning: dataset_manifest.json not found in Zenodo record")
            return None
        self._rate_limit()
        try:
            resp = requests.get(
                file_info['url'], headers=self.headers,
                timeout=30, allow_redirects=True)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"Warning: Could not fetch remote dataset manifest: {e}")
            return None

    def _check_manifest_compatibility(
        self, local: Dict, remote: Dict, force_update: bool
    ) -> bool:
        """
        Check package-version and embedding-model consistency between the local
        dataset_manifest.json and the one about to be downloaded.
        Returns True if the download can proceed, False if it should be aborted.
        """
        # If the local manifest predates the required_package_version field, it is
        # from an early dataset version and is not compatible with the current package
        if not local.get('required_package_version'):
            if force_update:
                print("Note: Local dataset predates version tracking; "
                      "all files will be replaced.\n")
            else:
                print(f"\n{'!'*60}")
                print("WARNING: Your local dataset is from an older version that is")
                print("not compatible with the current package.")
                print("\nTo upgrade, re-run download_data() with force_update=True:")
                print("  tte.download_data(path_to_data=..., force_update=True)")
                print(f"{'!'*60}\n")
                return False

        # Package version check — always enforced, even with force_update
        req_str = remote.get('required_package_version')
        if req_str:
            try:
                from importlib.metadata import version as pkg_version
                installed_str = pkg_version('task-tech-exposure')
                installed = tuple(int(x) for x in installed_str.split('.'))
                required = tuple(int(x) for x in req_str.split('.'))
                if installed < required:
                    print(f"\n{'!'*60}")
                    print(f"WARNING: This dataset requires package version >= {req_str}")
                    print(f"  Installed version: {installed_str}")
                    print("\nTo resolve, either:")
                    print("  1. Upgrade the package:")
                    print("       pip install --upgrade "
                          "git+https://github.com/armouton/task-tech-exposure.git")
                    print("  2. Download a dataset version compatible with your package:")
                    print("       https://doi.org/10.5281/zenodo.17643647")
                    print(f"{'!'*60}\n")
                    return False
            except Exception:
                pass  # If version metadata is unavailable, skip this check

        # Embedding model check
        model_keys = ['match_model', 'tech_model', 'task_model']
        mismatched = [k for k in model_keys
                      if local.get(k) and local.get(k) != remote.get(k)]
        if mismatched:
            if force_update:
                print("Note: Embedding models have changed in the updated dataset.")
                for k in mismatched:
                    print(f"  {k}: '{local.get(k)}' -> '{remote.get(k)}'")
                print("force_update=True: all local files will be replaced.\n")
            else:
                print(f"\n{'!'*60}")
                print("WARNING: The remote dataset uses different embedding models.")
                for k in mismatched:
                    print(f"  {k}:  local='{local.get(k)}'")
                    print(f"  {' ' * len(k)}  remote='{remote.get(k)}'")
                print("\nExisting embeddings and match files are not compatible with")
                print("the updated models. Re-downloading individual year files is")
                print("not sufficient — the full dataset must be replaced.")
                print("\nTo upgrade, re-run download_data() with force_update=True:")
                print("  tte.download_data(path_to_data=..., force_update=True)")
                print(f"{'!'*60}\n")
                return False

        return True

    def _determine_years_to_download(
        self, 
        remote_manifest: Dict, 
        start_year: int, 
        end_year: int
    ) -> Dict:
        """
        Filter manifest to only include files for years in the year range.
        Also includes non-year files (models, onet, manifest, etc.)
        """
        years_in_range = set(range(start_year, end_year + 1))
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
                    local_dir = self.path_to_data / "tte" / dir_name
                    if not local_dir.exists():
                        print(f"  Missing extracted data for {filename}")
                        files_to_download[filename] = file_info
                else:
                    # For non-zip files, check if file exists
                    local_file = self.path_to_data / "tte" / filename
                    if not local_file.exists():
                        print(f"  Missing file: {filename}")
                        files_to_download[filename] = file_info
        
        return {"version": remote_version, "files": files_to_download}
    
    def _download_and_extract(self, manifest: Dict, force_update: bool = False):
        """
        Download and extract files from Zenodo.
        Uses proper error handling and rate limiting.
        """
        tte_path = self.path_to_data / 'tte'
        tte_path.mkdir(exist_ok=True)

        total_files = len(manifest['files'])

        for idx, (filename, file_info) in enumerate(manifest['files'].items(), 1):
            print(f"\n[{idx}/{total_files}] Processing {filename}")

            # Skip files that are already present, unless force_update is set
            if not force_update:
                if filename.endswith('.zip'):
                    local_dir = tte_path / filename.replace('.zip', '')
                    if local_dir.exists():
                        print(f"  OK Already extracted, skipping")
                        continue
                else:
                    if (tte_path / filename).exists():
                        print(f"  OK Already exists, skipping")
                        continue

            # Download the file
            file_url = file_info['url']
            file_size = file_info.get('size', 0)
            is_zip = file_info.get("compressed", False)

            print(f"  Downloading ({file_size / 1024 / 1024:.1f} MB)...")

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

                downloaded = 0
                with open(temp_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if downloaded % (10 * 1024 * 1024) == 0:
                                print(f"  Downloaded {downloaded / 1024 / 1024:.1f} MB...")

                final_file = tte_path / filename
                temp_file.rename(final_file)
                print(f"  OK Download complete")

                if is_zip:
                    print(f"  Extracting...")
                    try:
                        with zipfile.ZipFile(final_file, 'r') as zip_ref:
                            zip_ref.extractall(tte_path)
                        final_file.unlink()
                        print(f"  OK Extracted and removed zip")
                    except zipfile.BadZipFile as e:
                        print(f"  ERROR: Bad zip file: {e}")
                        continue

            except requests.exceptions.HTTPError as e:
                print(f"  ERROR: HTTP Error: {e}")
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                print(f"  ERROR: Unexpected error: {e}")
                if temp_file.exists():
                    temp_file.unlink()
    
    def download_tte_data(
        self,
        from_year: int,
        to_year: int,
        doi_url: str,
        force_update: bool = False
    ):
        """
        Main method to download and process TTE data.
        
        Args:
            from_year: Start year (e.g., 2007)
            to_year: End year (e.g., 2023)
            doi_url: Zenodo DOI URL
            force_update: If True, re-download all files even if they exist
        """
        print(f"\n{'='*60}")
        print(f"TTE Dataset Downloader")
        print(f"{'='*60}")
        print(f"Year range: {from_year} to {to_year}")
        print(f"Source: {doi_url}")
        print(f"Output directory: {self.path_to_data}")
        print(f"{'='*60}\n")
        
        # Validate years
        if not isinstance(from_year, int) or not isinstance(to_year, int):
            raise ValueError("from_year and to_year must be integers")
        if from_year > to_year:
            raise ValueError("from_year must be less than or equal to to_year")
        if from_year < 2001:
            raise ValueError("Data is only available from 2001 onwards")
        
        # Get Zenodo record ID
        record_id = self._resolve_doi_to_zenodo_id(doi_url)
        
        # Fetch remote manifest
        remote_manifest = self._get_zenodo_manifest(record_id)

        # Verify compatibility with any existing local dataset before touching files
        local_dataset_manifest = self._load_local_dataset_manifest()
        if local_dataset_manifest is not None:
            remote_dataset_manifest = self._fetch_remote_dataset_manifest(remote_manifest)
            if remote_dataset_manifest is not None:
                if not self._check_manifest_compatibility(
                        local_dataset_manifest, remote_dataset_manifest, force_update):
                    return

        # Load local manifest
        local_manifest = self._load_local_manifest()

        # Filter to years in range
        year_filtered_manifest = self._determine_years_to_download(
            remote_manifest, from_year, to_year
        )
        
        # Determine what needs downloading
        manifest_to_download = self._determine_files_to_download(
            year_filtered_manifest, local_manifest, force_update
        )
        
        # Download files
        if not manifest_to_download['files']:
            print("\nOK All files are already present and up-to-date.")
        else:
            print(f"\nDownloading {len(manifest_to_download['files'])} files...")
            self._download_and_extract(manifest_to_download, force_update)

        # Save manifest
        self._save_local_manifest(remote_manifest)
        
        print(f"\n{'='*60}")
        print("OK Download and extraction complete")
        print(f"{'='*60}\n")

# CALL FUNCTION FOR DOWNLOADER
def download_data(
    path_to_data: str,
    from_year: int = None,
    to_year: int = None,
    doi_url: str = "https://doi.org/10.5281/zenodo.17643646",
    force_update: bool = False
):
    """
    Convenience function to download TTE dataset.

    Args:
        path_to_data: Directory to save data
        from_year: Start year (e.g., 2007). Defaults to 2001 (earliest available).
        to_year: End year (e.g., 2023). Defaults to current year.
        doi_url: Zenodo DOI URL
        force_update: If True, re-download all files
    """
    if not os.path.exists(path_to_data):
        print(f"Directory {path_to_data} does not exist, attempting to create...")
        try:
            os.makedirs(path_to_data, exist_ok=True)
            print(f"Created directory {path_to_data}")
        except Exception as e:
            raise Exception(f"Failed to create directory {path_to_data}: {e}")

    if from_year is None:
        from_year = 2001
    if to_year is None:
        to_year = time.localtime().tm_year

    downloader = TTEDownloader(path_to_data)
    downloader.download_tte_data(from_year, to_year, doi_url, force_update)