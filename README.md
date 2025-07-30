Satellite Data Harmonization Pipeline
A modular pipeline for preprocessing, QA-masking, and harmonizing multi-sensor satellite imagery (optical and SAR) into multi-resolution cloud-optimized tiles, with STAC-compliant catalog generation.

Supports: AWiFS, LISS-3, LISS-4, Landsat 8, Sentinel-2, SAR (e.g. EOS-4, Sentinel-1)
License: Apache 2.0

Features:-
Sensor-specific preprocessing (resampling, stacking, masking, temporal smoothing, compositing, tiling).

Advanced QA masking (cloud, shadow, haze, pollution, water, SAR speckle).

Robustly handles GDAL/Rasterio, Python-only or CLI fallbacks.

Harmonization into multi-resolution pyramids (COG tiles) and STAC metadata.

Sentinel-2 SAFE conversion automation (JP2 → GeoTIFF).

Idempotent, parallel, and production-ready.

Easy integration with downstream analytics tools.
===============================================================================================================================================
INSTALLATION
Clone the Repository:

bash-
git clone https://github.com/Wrench002/preprocess_harmonize.git
cd preprocess_harmonize
Install Miniconda/Anaconda if not already installed.

Create and activate environment (recommended):

bash-
conda create -n satpipe python=3.10 -y
conda activate satpipe
Install all required packages via conda-forge:

bash-
conda install -c conda-forge gdal rasterio numpy scipy joblib pyyaml tqdm psutil libgdal-jp2openjpeg
Note: The libgdal-jp2openjpeg package provides JPEG2000 (.jp2) support for Sentinel-2 SAFE conversion.
================================================================================================================================================
SETUP & ENVIRONMENT
Tested on Windows 11 and Linux (Ubuntu 20+) with Anaconda/Miniconda

Requires GDAL command-line tools and gdal_translate on PATH

Minimum 6 GB RAM recommended for parallel processing; adjust max_workers accordingly
================================================================================================================================================
DATA PREPARATION
1. Raw Directory Organization-
For each sensor, organize input TIFFs or original Sentinel-2 SAFE folders as follows:

text-
raw_images/
  AWiFS/
  LISS3/
  LISS4/
  SAR/
  Landsat8/
  Sentinel2/
      S2?_MSIL2A_<...>.SAFE/
2. Prepare and Rename Files
Run the prep script to:

Convert Sentinel-2 SAFE JP2 bands to pipeline-ready GeoTIFFs (single .tif per date-band, named YYYYMMDD_Sentinel2_Bxx.tif)

Batch rename all bands from other sensors so filenames contain both acquisition date and sensor (e.g., 2025-01-08_LISS3_BAND1.tif)

bash-
# Run the main preparation script
python prepare_all_sensors.py D:/Satellite_Data/raw_images
If files are not automatically matched for your sensor or naming conventions, see code comments or run the helper scripts in /utils.

How to Run-
Edit config.json to specify sensors, pyramid levels, CRS, and parallelization (see /configs for sensor YAMLs).

Run the pipeline:

bash-
python main_pipeline.py \
  --input-path D:/Satellite_Data/raw_images \
  --output-path D:/Satellite_Data/final_images/harmonized \
  --config config.json \
  --parallel \
  --max-workers 4
For Windows: Use all parameters on one line, or use ^ for line continuation.

Outputs-
Phase 1: Per-sensor, per-date preprocessed tiles with QA masks.

Phase 2: Harmonized, multi-resolution tiled pyramids and STAC catalog in the output directory.
================================================================================================================================================
DIRECTORY STRUCTURE
text:
preprocess_harmonize/
├── configs/               # YAML configs: band maps, QA thresholds, etc.
├── eo_qamask/             # Sensor-agnostic QA-masking library
├── phase_1/               # Sensor preprocessor code
├── phase_2/               # Harmonizer code (tile pyramid, STAC, etc)
├── data-prep scripts/     # File prep/conversion utilities
├── main_pipeline.py       # Pipeline entry point
├── config.json            # Main pipeline configuration
├── README.md              # This file
└── LICENSE                # Apache 2.0

CITATION AND LICENSE;
This project is licensed under the Apache License 2.0.

If you use this repository or its code and logic in research or production, please cite as:

text:
@software{preprocess_harmonize,
  author = {Paranjai Gusaria},
  title = {Satellite Data Harmonization Pipeline},
  year = {2025},
  url = {https://github.com/Wrench002/preprocess_harmonize},
  license = {Apache-2.0}
}
Contributing
Pull requests, issue reports, and community contributions are welcome!
Please open an issue for bugs, feature requests, or sensor-specific enhancements.

Acknowledgments
Sentinel/SAR pre-processing courtesy ESA/ISRO data policies.

Thanks to the GDAL, Rasterio, and Anaconda communities.

Questions?
Open an Issue, or email pgr.0002@gmail.com

Happy harmonizing!
