import os
import re
from datetime import datetime

def extract_date(text):
    """
    Extracts a date in 'DDMONYYYY' format from anywhere within a string.
    This version correctly handles names like 'RA308JUN2025' and 'sar_01JAN2025'.
    Returns the date in 'YYYY-MM-DD' format or None if not found.
    """
    # This regex correctly finds the date inside longer strings.
    pattern = r'(\d{1,2})([A-Z]{3})(\d{4})'
    
    match = re.search(pattern, text.upper())
    
    if match:
        day, month, year = match.groups()
        date_str = f"{day}{month}{year}"
        
        try:
            date_obj = datetime.strptime(date_str, "%d%b%Y")
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            return None
            
    return None

def rename_items_in_dir(root_dir):
    """
    Recursively renames files and directories by prepending a found date.
    Processes from the deepest level up (topdown=False) to avoid path issues.
    """
    renamed_count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # Process both files and directories in the current path
        items_to_process = filenames + dirnames
        
        for name in items_to_process:
            full_path = os.path.join(dirpath, name)
            
            if not os.path.exists(full_path):
                continue

            date_str = extract_date(name)
            
            if date_str:
                new_name = f"{date_str}_{name}"
                new_path = os.path.join(dirpath, new_name)
                
                if not os.path.exists(new_path):
                    try:
                        os.rename(full_path, new_path)
                        renamed_count += 1
                        # This new print statement will show you what is being changed
                        print(f"  -> Renamed: {name} TO {new_name}")
                    except OSError as e:
                        print(f"  -> ERROR renaming {name}: {e}")

    return renamed_count

if __name__ == "__main__":
    # Ensure this path is correct for your system
    base_path = r"D:\Satellite_Data\raw_images"
    sensors = ["AWiFS", "LISS3", "LISS4", "SAR", "Landsat8", "Sentinel2"]

    for sensor in sensors:
        sensor_path = os.path.join(base_path, sensor)
        if os.path.isdir(sensor_path):
            print(f"\n📂 Scanning: {sensor_path}")
            renamed = rename_items_in_dir(sensor_path)
            print(f"✅ Finished. {renamed} total items were renamed in '{sensor}'.")
        else:
            print(f"\n⚠️ Directory not found, skipping: {sensor_path}")