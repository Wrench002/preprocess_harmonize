# 🛰️ Satellite Data Harmonization Pipeline

A modular Python pipeline for harmonizing Earth Observation (EO) data from multiple sensors (e.g., AWiFS, LISS-3, LISS-4, Sentinel-2, SAR, Landsat-8). This system automates the transformation from raw satellite data to harmonized, cloud-optimized products—enabling scalable geospatial analysis for agriculture, urban planning, environment, and AI/ML model training.

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![Python Version](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey.svg)

---

## ⚙️ Core Workflow

- **Sensor-Specific Stacking**: Merges bands by date into multi-band images per sensor.
- **Resampling & Reprojection**: Aligns all images to a common resolution and CRS.
- **Temporal Smoothing**: Applies pixel-wise time-series smoothing for cleaner analysis.
- **Advanced QA Masking**: Flags clouds, haze, water, SAR speckle, shadows, and no-data zones using sensor-specific thresholds.
- **Compositing & Tiling**: Generates cloud-optimized tiles and pyramid composites for ML/GIS.
- **Cross-Sensor Harmonization**: Radiometric and spatial harmonization across sensors.
- **STAC Metadata Generation**: Catalogs outputs for AI-ready and GIS-searchable data ingestion.

---

## 📋 Features

- ✅ **Supports**: AWiFS, LISS-3, LISS-4, Landsat-8, Sentinel-2, SAR
- ⚡ **Parallelized**: Uses `joblib` for multi-core processing
- 📦 **Automated SAFE Conversion**: Converts Sentinel-2 `.SAFE` to standardized TIFFs
- 🧠 **AI/ML Ready**: Outputs are harmonized and optimized for model pipelines
- 🧩 **Modular & Configurable**: YAML-based configs per sensor with customizable parameters

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Wrench002/preprocess_harmonize.git
cd preprocess_harmonize
2. Install Miniconda/Anaconda (if not already installed)
3. Create and Activate Conda Environment
bash
Copy
Edit
conda create -n satpipe python=3.10 -y
conda activate satpipe
4. Install Required Dependencies
bash
Copy
Edit
conda install -c conda-forge gdal rasterio numpy scipy joblib pyyaml tqdm psutil libgdal-jp2openjpeg
Note: libgdal-jp2openjpeg enables JPEG2000 (.jp2) decoding used in Sentinel-2 .SAFE files.

🖥️ Environment & Requirements
Tested on: Windows 11, Ubuntu 20.04+

RAM: Minimum 6 GB recommended for parallel execution

Command-Line Tools: Ensure gdal_translate and other GDAL tools are in your system PATH

📂 Data Preparation
1. Directory Structure
Organize input data under raw_images/:

ruby
Copy
Edit
raw_images/
├── AWiFS/
├── LISS3/
├── LISS4/
├── SAR/
├── Landsat8/
└── Sentinel2/
    └── S2??MSIL2A_*.SAFE/
2. Prepare and Rename Files
Use the utility script to standardize band names and convert Sentinel-2 .SAFE archives to GeoTIFF:

bash
Copy
Edit
python prepare_all_sensors.py D:/Satellite_Data/raw_images
Output format: YYYYMMDD_SENSOR_BAND.tif (e.g., 20250108_LISS3_B1.tif)

If your filenames don’t match, refer to /utils/ for helpers or edit the script as needed.

🚀 Running the Pipeline
Step 1: Edit the Configuration
Modify config.json and the sensor-specific YAML files in /configs/.

Choose sensors

Define harmonization levels

Set CRS, resampling, and tiling preferences

Adjust parallelization settings

Step 2: Launch Pipeline
bash
Copy
Edit
python main_pipeline.py \
  --input-path D:/Satellite_Data/raw_images \
  --output-path D:/Satellite_Data/final_images/harmonized \
  --config config.json \
  --parallel \
  --max-workers 4
On Windows, either use a single line or ^ for line continuation.

📁 Project Structure
pgsql
Copy
Edit
preprocess_harmonize/
├── configs/              # YAML configs: band mappings, QA rules, etc.
├── eo_qamask/            # QA masking logic per sensor
├── phase_1/              # Preprocessing modules: stacking, smoothing, etc.
├── phase_2/              # Harmonization: tiling, STAC, mosaics
├── data-prep-scripts/    # File renaming and conversion utilities
├── main_pipeline.py      # Entry point
├── config.json           # Main configuration
├── LICENSE               # Apache License 2.0
└── README.md             # This file
📖 Citation & License
This project is licensed under the Apache License 2.0.

If you use this pipeline in your research or production systems, please cite:

bibtex
Copy
Edit
@software{preprocess_harmonize,
  author = {Paranjai Gusaria},
  title = {Satellite Data Harmonization Pipeline},
  year = {2025},
  url = {https://github.com/Wrench002/preprocess_harmonize},
  license = {Apache-2.0}
}
🤝 Contributing
Contributions are welcome! Submit PRs, raise issues, or suggest improvements.

🙏 Acknowledgments
Inspired by data pipelines from ESA, ISRO, and open-source geospatial communities

Built on top of GDAL, Rasterio, SciPy, and Anaconda ecosystems

💬 Questions?
Open an issue or reach out:
📧 pgr.0002@gmail.com

Happy harmonizing! 🌍


