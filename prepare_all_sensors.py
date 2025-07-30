"""
Comprehensive data preparation script for the satellite harmonization pipeline.

This script performs two main tasks:
1.  For Sentinel-2 data: It finds `.SAFE` directories, extracts the individual
    band `.jp2` files, and converts them to GeoTIFF (`.tif`) format using GDAL.
    This conversion is parallelized for efficiency. The output files are
    automatically named in the 'YYYYMMDD_Sensor_Band.tif' format required
    by the pipeline.

2.  For all other sensors: It scans for files with dates in 'DDMONYYYY' format
    and renames them to the 'YYYY-MM-DD_Sensor_OriginalName.tif' format.

The script is idempotent, meaning it is safe to run multiple times on the same
directories.
"""
import argparse
import re
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
# This dictionary defines which Sentinel-2 bands to extract and from which
# resolution subdirectories they should be read.
SENTINEL2_BANDS_TO_EXTRACT = {
    'B02': 'R10m', 'B03': 'R10m', 'B04': 'R10m', 'B08': 'R10m',
    'B11': 'R20m', 'B12': 'R20m', 'SCL': 'R20m'  # Scene Classification Layer
}

def check_gdal_dependency():
    """Checks if gdal_translate is available in the system's PATH."""
    if not shutil.which("gdal_translate"):
        print("❌ FATAL ERROR: `gdal_translate` command not found.")
        print("Please ensure GDAL is installed and that its command-line tools are in your system's PATH.")
        print("If using Conda, this can be achieved by installing 'gdal' from the 'conda-forge' channel.")
        return False
    return True

def convert_s2_band(jp2_file: Path, out_fpath: Path):
    """Worker function to convert a single Sentinel-2 JP2 file to GeoTIFF."""
    try:
        subprocess.run(
            ['gdal_translate', str(jp2_file), str(out_fpath)],
            check=True, capture_output=True, text=True
        )
        return f"Converted {jp2_file.name} -> {out_fpath.name}"
    except subprocess.CalledProcessError as e:
        return f"❌ Error converting {jp2_file.name}: {e.stderr.strip()}"
    except Exception as e:
        return f"❌ An unexpected error occurred for {jp2_file.name}: {e}"

def sentinel2_jp2_to_tif(s2_root_dir: Path):
    """Finds Sentinel-2 .SAFE directories and converts required bands to GeoTIFF in parallel."""
    print("========== Sentinel-2 SAFE JP2 -> GeoTIFF Conversion ==========")
    safe_paths = list(s2_root_dir.rglob('*.SAFE'))
    if not safe_paths:
        print(f"No .SAFE directories found in {s2_root_dir}. Skipping.")
        return

    tasks = []
    for safe_path in safe_paths:
        date = extract_date_from_s2safe(safe_path.name)
        if not date:
            print(f"⚠️ Could not parse date from {safe_path.name}, skipping.")
            continue

        granule_dirs = list((safe_path / 'GRANULE').glob('*'))
        if not granule_dirs: continue

        for granule_dir in granule_dirs:
            img_data_dir = granule_dir / 'IMG_DATA'
            for band, res_folder in SENTINEL2_BANDS_TO_EXTRACT.items():
                search_dir = img_data_dir / res_folder
                if not search_dir.exists(): continue
                
                jp2_files = list(search_dir.glob(f"*_{band}_*.jp2"))
                if not jp2_files: continue

                jp2_file = jp2_files[0]
                out_fname = f"{date}_Sentinel2_{band}.tif"
                out_fpath = s2_root_dir / out_fname

                if not out_fpath.exists():
                    tasks.append((jp2_file, out_fpath))

    if not tasks:
        print("All required Sentinel-2 GeoTIFF files already exist. No conversion needed.")
        return

    # Parallelize the conversion process
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(convert_s2_band, jp2, out_path): out_path for jp2, out_path in tasks}
        for future in as_completed(futures):
            print(future.result())

def extract_date_from_s2safe(safe_name: str) -> str | None:
    """Extracts date in YYYYMMDD format from a Sentinel-2 .SAFE directory name."""
    match = re.search(r'_(\d{8})T', safe_name)
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y%m%d').strftime('%Y%m%d')
        except ValueError:
            return None
    return None

def extract_date_ddmonyyyy(text: str) -> str | None:
    """Extracts 'DDMONYYYY' date from a string and returns it as 'YYYY-MM-DD'."""
    pattern = r'(\d{1,2})([A-Z]{3})(\d{4})'
    match = re.search(pattern, text.upper())
    if match:
        day, month, year = match.groups()
        date_str = f"{day}{month}{year}"
        try:
            return datetime.strptime(date_str, "%d%b%Y").strftime("%Y-%m-%d")
        except ValueError:
            return None
    return None

def rename_other_sensors(root_dir: Path, sensor_name: str) -> int:
    """Recursively renames files by prepending the extracted date and sensor name."""
    renamed_count = 0
    # Iterate from deepest to shallowest to handle directory renames correctly
    for item_path in sorted(list(root_dir.rglob('*')), key=lambda p: len(p.parts), reverse=True):
        if re.match(r'(\d{4}-\d{2}-\d{2}_|\d{8}_)', item_path.name):
            continue

        date_str = extract_date_ddmonyyyy(item_path.name)
        if date_str:
            new_name = f"{date_str}_{sensor_name}_{item_path.name}"
            new_path = item_path.with_name(new_name)
            if not new_path.exists():
                try:
                    item_path.rename(new_path)
                    renamed_count += 1
                    print(f"  -> Renamed: {item_path.name:<40} TO    {new_name}")
                except OSError as e:
                    print(f"  -> ERROR renaming {item_path.name}: {e}")
    return renamed_count

def main():
    """Main function to parse arguments and run the preparation process."""
    parser = argparse.ArgumentParser(description="Prepare satellite data for the harmonization pipeline.")
    parser.add_argument('base_path', type=Path, help="The base path to the raw sensor data folders.")
    args = parser.parse_args()

    if not args.base_path.is_dir():
        print(f"Error: Base path not found or not a directory: {args.base_path}")
        return

    if not check_gdal_dependency():
        return

    all_sensors = ["AWiFS", "LISS3", "LISS4", "SAR", "Landsat8", "Sentinel2"]

    # --- Step 1: Handle Sentinel-2 Conversion ---
    s2_path = args.base_path / "Sentinel2"
    if s2_path.is_dir():
        sentinel2_jp2_to_tif(s2_path)
    else:
        print(f"\n⚠️ Sentinel2 directory not found, skipping: {s2_path}")

    # --- Step 2: Handle Renaming for Other Sensors ---
    for sensor in all_sensors:
        if sensor == "Sentinel2":
            continue

        sensor_path = args.base_path / sensor
        if sensor_path.is_dir():
            print(f"\n📂 Scanning and renaming in: {sensor_path}")
            renamed = rename_other_sensors(sensor_path, sensor)
            print(f"✅ Finished. {renamed} total items were renamed in '{sensor}'.")
        else:
            print(f"\n⚠️ Directory not found, skipping: {sensor_path}")
    
    print("\nData preparation complete. Files are ready for the main pipeline.")

if __name__ == "__main__":
    main()
