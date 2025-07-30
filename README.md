# 🛰️ Satellite Data Harmonization Pipeline

![Python](https://img.shields.io/badge/python-3.10+-blue?style=flat-square)
![License](https://img.shields.io/github/license/Wrench002/preprocess_harmonize?style=flat-square)
![Platform](https://img.shields.io/badge/platform-windows%20%7C%20linux-brightgreen?style=flat-square)

A modular Python pipeline for harmonizing Earth Observation (EO) data from multiple sensors (e.g., **AWiFS, LISS-3, LISS-4, Sentinel-2, SAR, Landsat-8**).  
This system automates the transformation from raw satellite data to harmonized, cloud-optimized products—enabling scalable geospatial analysis for agriculture, urban planning, environment, and AI/ML model training.

---

## ⚙️ Core Workflow

- **Sensor-Specific Stacking**  
  Merges bands by date into multi-band images per sensor.

- **Resampling & Reprojection**  
  Aligns all images to a common resolution and CRS.

- **Temporal Smoothing**  
  Applies pixel-wise time-series smoothing for cleaner analysis.

- **Advanced QA Masking**  
  Flags clouds, haze, water, SAR speckle, shadows, and no-data zones using sensor-specific thresholds.

- **Compositing & Tiling**  
  Generates cloud-optimized tiles and pyramid composites for ML/GIS.

- **Cross-Sensor Harmonization**  
  Radiometric and spatial harmonization across sensors.

- **STAC Metadata Generation**  
  Catalogs outputs for AI-ready and GIS-searchable data ingestion.

---

## 📋 Features

- ✅ Supports: AWiFS, LISS-3, LISS-4, Landsat-8, Sentinel-2, SAR  
- ⚡ Parallelized using `joblib` for multi-core execution  
- 📦 Auto-converts `.SAFE` (Sentinel-2) to standardized GeoTIFFs  
- 🧠 AI/ML Ready: Outputs are harmonized and optimized for model pipelines  
- 🧩 Modular & Configurable: YAML-based configs per sensor with customizable parameters  

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Wrench002/preprocess_harmonize.git
cd preprocess_harmonize

2. Set Up Conda Environment
bash-
conda create -n satpipe python=3.10 -y (or any name you prefer)

conda activate satpipe

3. Install Dependencies
bash-
conda install -c conda-forge gdal rasterio numpy scipy joblib pyyaml tqdm psutil libgdal-jp2openjpeg
```
📝 libgdal-jp2openjpeg is required to decode JPEG2000 (.jp2) used in Sentinel-2.

🖥️ Environment & Requirements
OS: Windows 11, Ubuntu 20.04+

RAM: 6 GB+ recommended

Tools: Ensure gdal_translate and other GDAL tools are available in PATH

📂 Data Preparation

1. Organize Input Data
```
bash-
raw_images/
├── AWiFS/
├── LISS3/
├── LISS4/
├── SAR/
├── Landsat8/
└── Sentinel2/
    └── S2??MSIL2A_*.SAFE/
```
2. Convert and Rename Files
Run:
```
bash-

python prepare_all_sensors.py (input path)
```
This standardizes band names and converts .SAFE files to GeoTIFF format.
Output format: YYYYMMDD_SENSOR_BAND.tif (e.g., 20250108_LISS3_B1.tif)

⚠️ If filenames don’t match expected format, refer to /utils/ for helper scripts.

🚀 Running the Pipeline

Step 1: Edit Configs

config.json: general settings

/configs/: sensor-specific YAMLs

Define:

Sensors

CRS/resampling settings

Harmonization level

Parallelization

Step 2: Run
```
bash-

python main_pipeline.py \
  --input-path (your input path)
  --output-path (your output path)
  --config config.json \
  --parallel \
  --max-workers 4
🔧 On Windows, use ^ or combine into one line.
```
📁 Project Structure
```
preprocess_harmonize/
├── configs/              # YAML configs for bands & QA
├── eo_qamask/            # QA masking functions per sensor
├── phase_1/              # Stacking, smoothing, resampling, etc.
├── phase_2/              # Harmonization, tiling, STAC output
├── data-prep-scripts/    # Utility scripts (renaming, SAFE conversion)
├── config.json           # Global pipeline config
├── main_pipeline.py      # Entry point
├── LICENSE
└── README.md
```
📖 Citation & License
Licensed under Apache License 2.0.

If you use this pipeline in research/publication:
@software{preprocess_harmonize,
  author = {Paranjai Gusaria},
  title = {Satellite Data Harmonization Pipeline},
  year = {2025},
  url = {https://github.com/Wrench002/preprocess_harmonize},
  license = {Apache-2.0}
}
🤝 Contributing
Pull requests, issues, and suggestions are welcome!
Please follow modular structure and submit clean PRs with explanations.

🙏 Acknowledgments
Inspired by pipelines from ISRO and the open geospatial community.

Built on: GDAL, Rasterio, NumPy, SciPy, Anaconda

💬 Contact
📧 Email: pgr.0002@gmail.com
🔗 Repo: github.com/Wrench002/preprocess_harmonize

Happy Harmonizing! 🌍
