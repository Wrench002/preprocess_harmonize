Markdown

# Satellite Data Harmonization Pipeline 🛰️

A modular pipeline for preprocessing, QA-masking, and harmonizing multi-sensor satellite imagery (optical and SAR) into multi-resolution, cloud-optimized tiles with a STAC-compliant catalog.

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![Python Version](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey.svg)

---

## 📋 Features

* **Sensor-Specific Preprocessing**: Automated resampling, stacking, masking, temporal smoothing, compositing, and tiling.
* **Advanced QA Masking**: Detects and masks cloud, shadow, haze, pollution, water, and SAR speckle.
* **Robust Backend**: Utilizes GDAL/Rasterio with Python-only or CLI fallbacks.
* **Harmonization**: Creates multi-resolution pyramids as Cloud-Optimized GeoTIFFs (COG) with STAC metadata.
* **Automation**: Includes scripts for converting Sentinel-2 SAFE (`.jp2`) archives into pipeline-ready GeoTIFFs.
* **Production Ready**: Idempotent, parallel-processing enabled, and designed for scalable workflows.
* **Easy Integration**: Seamlessly connects with downstream analytics tools.

---

## 📡 Supported Sensors

* AWiFS
* LISS-3
* LISS-4
* Landsat 8
* Sentinel-2
* SAR (e.g., EOS-4, Sentinel-1)

---

## 🛠️ Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Wrench002/preprocess_harmonize.git](https://github.com/Wrench002/preprocess_harmonize.git)
    cd preprocess_harmonize
    ```
2.  **Install Miniconda/Anaconda** if you don't have it already.

3.  **Create and Activate Conda Environment** (Recommended)
    ```bash
    conda create -n satpipe python=3.10 -y
    conda activate satpipe
    ```
4.  **Install Required Packages**
    ```bash
    conda install -c conda-forge gdal rasterio numpy scipy joblib pyyaml tqdm psutil libgdal-jp2openjpeg
    ```
    > **Note**: The `libgdal-jp2openjpeg` package provides JPEG2000 (`.jp2`) support, which is required for Sentinel-2 SAFE conversion.

---

## 🖥️ Setup & Environment

* **Tested On**: Windows 11 and Linux (Ubuntu 20+) with Anaconda/Miniconda.
* **Dependencies**: Requires GDAL command-line tools (especially `gdal_translate`) to be available on your system's `PATH`.
* **Hardware**: A minimum of **6 GB RAM** is recommended for parallel processing. Adjust the `max_workers` setting in `config.json` based on your system's capacity.

---

## 📂 Data Preparation

### 1. Raw Directory Organization

For each sensor, organize your input TIFFs or original Sentinel-2 `.SAFE` folders as follows:

```text
raw_images/
├── AWiFS/
├── LISS3/
├── LISS4/
├── SAR/
├── Landsat8/
└── Sentinel2/
    └── S2?MSIL2A<...>.SAFE/
2. Prepare and Rename Files
Run the preparation script to convert Sentinel-2 .SAFE bands to GeoTIFFs and standardize file names across all sensors. The script ensures filenames follow a YYYYMMDD_Sensor_Band.tif format (e.g., 20250108_LISS3_B1.tif).

Bash

python prepare_all_sensors.py D:/Satellite_Data/raw_images
Tip: If the script doesn't automatically match your files due to custom naming conventions, see the helper scripts in the /utils directory or the comments within the code.

🚀 How to Run
Edit the Configuration: Modify config.json to specify the sensors to process, desired pyramid levels, target CRS, and parallelization settings. Sensor-specific parameters are located in the YAML files within the /configs directory.

Run the Main Pipeline:

Bash

python main_pipeline.py --input-path D:/Satellite_Data/raw_images --output-path D:/Satellite_Data/final_images/harmonized --config config.json --parallel --max-workers 4
For Windows Users: Either write the command on a single line or use the ^ character for line continuation.

Pipeline Outputs
Phase 1: Creates per-sensor, per-date preprocessed tiles complete with QA masks.

Phase 2: Generates the final harmonized, multi-resolution tiled pyramids and a STAC catalog in the specified output directory.

📁 Directory Structure
Plaintext

preprocess_harmonize/
├── configs/              # YAML configs: band maps, QA thresholds, etc.
├── eo_qamask/            # Sensor-agnostic QA-masking library
├── phase_1/              # Sensor preprocessor code
├── phase_2/              # Harmonizer code (tile pyramid, STAC, etc)
├── data-prep-scripts/    # File preparation/conversion utilities
├── main_pipeline.py      # Pipeline entry point
├── config.json           # Main pipeline configuration
├── README.md             # This file
└── LICENSE               # Apache 2.0
©️ Citation and License
This project is licensed under the Apache License 2.0.

If you use this repository or its code in your research or production environment, please cite it as follows:

Plaintext

@software{preprocess_harmonize,
  author = {Paranjai Gusaria},
  title = {Satellite Data Harmonization Pipeline},
  year = {2025},
  url = {[https://github.com/Wrench002/preprocess_harmonize](https://github.com/Wrench002/preprocess_harmonize)},
  license = {Apache-2.0}
}
🙌 Contributing
Pull requests, issue reports, and community contributions are welcome! Please open an issue on GitHub for any bugs, feature requests, or suggestions for sensor-specific enhancements.

Acknowledgments
Sentinel/SAR pre-processing logic is designed in accordance with ESA/ISRO data policies.

Special thanks to the developers and communities behind GDAL, Rasterio, and Anaconda.

Questions?
Open an issue on GitHub or email pgr.0002@gmail.com.

Happy harmonizing!
