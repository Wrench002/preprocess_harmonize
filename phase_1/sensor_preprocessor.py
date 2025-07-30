"""
Satellite sensor-specific Phase-1 preprocessing.

This module contains the SensorPreprocessor class, which handles all steps
for processing a single sensor's data, from raw bands to a tiled composite.

"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import multiprocessing
from pathlib import Path
from typing import List, Sequence

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin
import rasterio.enums
from rasterio.warp import reproject, calculate_default_transform

# Performance and scientific computing imports
from joblib import Parallel, delayed
from scipy import ndimage as ndi
import scipy.sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter

# Custom masking library imports
# Ensure the 'eo_qamask' library is in the python path
try:
    from eo_qamask.api import apply as qamask_apply
    from eo_qamask.bitdefs import QABits
except ImportError:
    # Provide a fallback if the library isn't found, to prevent crashing.
    print("Warning: 'eo_qamask' library not found. Masking step will be skipped.")
    qamask_apply = None
    QABits = None


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------
def _which(cmd: str) -> str | None:
    """Cross-platform shutil.which() shim."""
    try:
        from shutil import which
    except ImportError:
        from distutils.spawn import find_executable as which
    return which(cmd)

def _log(msg: str) -> None:
    """Helper for consistent logging output."""
    print(msg, flush=True)

def _smooth_pixel_series(series, lambd=10.0, savgol_window=5, savgol_poly=2):
    """Enhanced pixel time-series smoothing with error handling."""
    m = len(series)

    # Validate inputs
    if m < 3:
        return series

    # Ensure savgol window is odd and appropriately sized
    if savgol_window >= m:
        savgol_window = max(3, m // 2)
    if savgol_window % 2 == 0:
        savgol_window += 1

    # Check for valid data points
    valid_mask = ~np.isnan(series)
    if np.sum(valid_mask) < 3:
        return series

    try:
        # Whittaker smoother
        w_sparse = scipy.sparse.spdiags(valid_mask.astype(float), 0, m, m)
        d_matrix = scipy.sparse.eye(m, format='csc')
        for _ in range(2):  # Second-order differences
            d_matrix = d_matrix[1:] - d_matrix[:-1]

        a_matrix = w_sparse + lambd * (d_matrix.T @ d_matrix)
        y_no_nan = np.nan_to_num(series, nan=0.0)

        smoothed = spsolve(a_matrix, w_sparse @ y_no_nan)

        # Apply Savitzky-Golay filter if window size is valid
        if savgol_window > savgol_poly and len(smoothed) >= savgol_window:
            smoothed = savgol_filter(smoothed, window_length=savgol_window,
                                    polyorder=min(savgol_poly, savgol_window - 1))

        return smoothed
    except Exception as e:
        _log(f"    - Warning: Smoothing failed, returning original series: {e}")
        return series

# ---------------------------------------------------------------------
# Preprocessor class
# ---------------------------------------------------------------------
class SensorPreprocessor:
    """
    Per-sensor Phase-1 pipeline.
    The public entry point is ``run_phase1_pipeline()``.
    """
    def __init__(self, sensor_name: str, sensor_params: dict):
        self.sensor_name = sensor_name
        self.sensor_params = sensor_params
        self.sensor_type = sensor_params.get('type', 'optical')
        self.base_dir = Path(__file__).resolve().parent.parent
        self.num_cores = min(multiprocessing.cpu_count(),
                             sensor_params.get('max_cores', 8))  # Limit cores

        # Validate required parameters
        required_params = ['resolution', 'tile_size']
        for param in required_params:
            if param not in sensor_params:
                raise ValueError(f"Missing required parameter '{param}' for {sensor_name}")

    @staticmethod
    def _run_cmd(cmd: Sequence[str]) -> subprocess.CompletedProcess:
        """Run *cmd*; raise with full log on failure."""
        use_shell = os.name == "nt"
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=use_shell,
            check=False
        )
        if result.returncode != 0:
            _log(f"ERROR running command: {' '.join(cmd)}")
            _log(result.stderr)
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )
        return result

    # --------------------------------------
    # Step 1 – Group & Stack Daily Scenes
    # --------------------------------------
    def step1_group_and_stack(self, input_dir: Path, output_dir: Path) -> list[str]:
        """
        Identify single-band GeoTIFFs with a date token (YYYY-MM-DD or YYYYMMDD)
        anywhere in the filename, stack them into one file per date, and return the new file list.
        """
        _log(f"\nStep 1: Grouping and stacking bands for {self.sensor_name}...")
        date_regex = re.compile(r"(\d{4}-?\d{2}-?\d{2})")
        
        scene_files = {}
        for file_path in Path(input_dir).rglob("*.tif"):
            match = date_regex.search(file_path.stem)
            if match:
                date_str = match.group(1).replace("-", "")
                if date_str not in scene_files:
                    scene_files[date_str] = []
                scene_files[date_str].append(file_path)

        if not scene_files:
            _log("  - No dated .tif files found to process.")
            return []

        out_files: list[str] = []
        for date, bands in sorted(scene_files.items()):
            try:
                with rasterio.open(bands[0]) as first:
                    meta = first.profile
                
                meta.update(
                    count=len(bands),
                    tiled=True,
                    compress="lzw",
                    BIGTIFF="IF_NEEDED",
                    blockxsize=min(512, meta["width"]),
                    blockysize=min(512, meta["height"]),
                )

                
                stacked_path = output_dir / f"{self.sensor_name}_{date}_stacked.tif"
                with rasterio.open(stacked_path, "w", **meta) as dst:
                    for idx, band_path in enumerate(sorted(bands), start=1):
                        with rasterio.open(band_path) as src:
                            dst.write(src.read(1), idx)
                out_files.append(str(stacked_path))
                _log(f"  - Created stacked scene for {date} with {len(bands)} bands.")
            except Exception as e:
                _log(f"  - WARNING: Could not stack scene for {date}. Error: {e}")

        return out_files

    # --------------------------------------
    # Step 2 – Resampling (with GDAL fallback)
    # --------------------------------------
    def step2_resampling(self, input_files: list[str], output_path: Path) -> list[str]:
        _log(f"\nStep 2: Resampling to native resolution...")
        resampled_files = []
        target_res = self.sensor_params['resolution']

        if _which("gdalwarp"):
            for file_path in input_files:
                output_file = output_path / f"resampled_{Path(file_path).name}"
                cmd = [
                    'gdalwarp', '-overwrite', 
                    '-tr', str(target_res), str(target_res),
                    '-r', 'cubic', '-multi', 
                    '-wo', f'NUM_THREADS={self.num_cores}',
                    '-co', 'TILED=YES', 
                    '-co', 'COMPRESS=LZW',
                    '-co', 'BIGTIFF=IF_NEEDED',
                    file_path, str(output_file)
                ]
                try:
                    self._run_cmd(cmd)
                    resampled_files.append(str(output_file))
                except subprocess.CalledProcessError:
                    _log(f"  - GDAL warp failed for {file_path}, using rasterio fallback")
                    resampled_files.append(self._python_resample(file_path, output_file, target_res))
        else:
            _log("  - gdalwarp not available, using pure-Python resampling")
            for file_path in input_files:
                output_file = output_path / f"resampled_{Path(file_path).name}"
                resampled_files.append(self._python_resample(file_path, output_file, target_res))
        
        return resampled_files

    # --------------------------------------
    # Step 3 – Masking and Filtering
    # --------------------------------------
    def step3_mask_and_filter(self, input_files: list[str], output_path: Path) -> list[str]:
        _log(f"\nStep 3: Applying QA mask and noise filter...")
        processed_files = []
        configs_dir = self.base_dir / 'configs'

        if qamask_apply is None:
            _log("  - Skipping mask and filter step as 'eo_qamask' is not installed.")
            return input_files

        for file_path in input_files:
            try:
                qa_mask, profile = qamask_apply(image_path=file_path, configs_dir=str(configs_dir), sensor=self.sensor_name)
                with rasterio.open(file_path) as src:
                    data = src.read().astype(np.float32)
                
                bad_pixel_filter = ((qa_mask & (1 << QABits.CLOUD.value)) | (qa_mask & (1 << QABits.SHADOW.value))).astype(bool)
                if bad_pixel_filter.shape == data.shape[1:]:
                    data[:, bad_pixel_filter] = np.nan
                
                if self.sensor_name == 'SAR':
                    processed_data = self._apply_lee_filter(data)
                elif self.sensor_name in ['LISS3', 'LISS4']:
                    processed_data = self._apply_destriping(data)
                else:
                    processed_data = data
                
                output_file = output_path / f"processed_{Path(file_path).name}"
                profile.update(dtype=rasterio.float32, count=data.shape[0], nodata=np.nan)
                self._save_raster(processed_data, profile, str(output_file))
                processed_files.append(str(output_file))
                _log(f"  - Masked and filtered {Path(file_path).name}")

            except Exception as e:
                _log(f"  - WARNING: Mask/filter failed for {Path(file_path).name}. Error: {e}")
                processed_files.append(file_path)
        return processed_files

    # --------------------------------------
    # Step 4 – Temporal Stacking (Memory-Safe)
    # --------------------------------------
    def step4_temporal_stacking(self, files: list[str], out_path: Path) -> list[str]:
        _log(f"\nStep 4: Stacking images by time period (memory-safe)...")
        if not files:
            return []
        
        try:
            import psutil
            initial_memory = psutil.virtual_memory().available / 1024**3
            _log(f"  - Available memory: {initial_memory:.1f} GB")
        except ImportError:
            pass
            
        stack_path = out_path / f"{self.sensor_name}_stack.tif"

        with rasterio.open(files[0]) as ref:
            profile = ref.profile
            total_bands = sum(rasterio.open(f).count for f in files)
            
            estimated_size = (ref.width * ref.height * total_bands *
                              np.dtype(profile['dtype']).itemsize) / 1024**3
            _log(f"  - Estimated output size: {estimated_size:.2f} GB")
            
            profile.update(
                count=total_bands,
                compress="lzw",
                tiled=True,
                BIGTIFF="IF_NEEDED" if estimated_size > 3.5 else "NO",
                predictor=2 if profile.get('dtype') in ['uint16', 'int16'] else 1
            )
        
        _log(f"  - Creating output stack file: {stack_path.name}")
        with rasterio.open(stack_path, "w", **profile) as dst:
            output_band_index = 1
            for f_idx, src_fp in enumerate(files):
                _log(f"  - Processing file {f_idx + 1}/{len(files)}: {Path(src_fp).name}")
                with rasterio.open(src_fp) as src:
                    window = Window(0, 0, src.width, src.height).intersection(
                        Window(0, 0, dst.width, dst.height)
                    )
                    if window.width > 0 and window.height > 0:
                        data_block = self._harmonize_array_shapes(src.read(window=window), (dst.height, dst.width))
                        dst.write(
                            data_block,
                            window=window,
                            indexes=range(
                                output_band_index,
                                output_band_index + src.count,
                            ),
                        )
                    output_band_index += src.count
        return [str(stack_path)]

    # --------------------------------------
    # Step 5 – Gap-filling and Smoothing
    # --------------------------------------
    def step5_gap_filling_smoothing(self, input_files: list[str], output_path: Path) -> list[str]:
        _log(f"\nStep 5: Applying temporal smoothing (in parallel)...")
        if not input_files: return []
        
        stack_file = input_files[0]
        with rasterio.open(stack_file) as src:
            data, profile = src.read(), src.profile
        
        n_bands, height, width = data.shape
        if n_bands < 5:
            _log("  - Skipping smoothing, not enough time-steps.")
            return input_files

        _log(f"  - Smoothing a stack with {n_bands} time-steps across {self.num_cores} cores...")
        pixels_as_cols = data.reshape(n_bands, height * width)
        
        smoothed_results = Parallel(n_jobs=self.num_cores, backend='loky', verbose=0)(
            delayed(_smooth_pixel_series)(pixels_as_cols[:, i]) for i in range(pixels_as_cols.shape[1])
        )
        
        smoothed_pixels = np.array(smoothed_results).T
        smoothed_data = smoothed_pixels.reshape(n_bands, height, width)
        
        output_file = output_path / f"smoothed_{Path(stack_file).name}"
        # FIX: Ensure nodata value is set correctly for the smoothed output.
        profile["nodata"] = np.nan
        self._save_raster(smoothed_data, profile, str(output_file))
        return [str(output_file)]

    # --------------------------------------
    # Step 6 – Compositing
    # --------------------------------------
    def step6_compositing(self, input_files: list[str], output_path: Path) -> list[str]:
        _log(f"\nStep 6: Creating final composite image...")
        if not input_files: return []
        stack_file = input_files[0]
        
        with rasterio.open(stack_file) as src:
            data, profile = src.read(), src.profile
        
        with np.errstate(all='ignore'):
            composite = np.nanmean(data, axis=0)
        
        output_file = output_path / f"{self.sensor_name}_composite.tif"
        profile.update(count=1, dtype='float32', nodata=np.nan)
        self._save_raster(composite[np.newaxis, :, :], profile, str(output_file))
        return [str(output_file)]

    # --------------------------------------
    # Step 7 – Tiling (with Python fallback)
    # --------------------------------------
    def step7_tiling(self, files: list[str], out_dir: Path) -> list[str]:
        _log("\nStep 7: Tiling final composite...")
        if not files:
            return []
        
        composite = files[0]
        tile_size = int(self.sensor_params.get("tile_size", 512))
        
        gdal_retile_path = _which("gdal_retile.py")
        if gdal_retile_path:
            _log("  - Using gdal_retile.py for tiling.")
            cmd = [
                gdal_retile_path,
                "-ps", str(tile_size), str(tile_size),
                "-co", "TILED=YES", "-co", "COMPRESS=LZW", "-co", "BIGTIFF=IF_NEEDED",
                "-targetDir", str(out_dir), str(composite),
            ]
            try:
                self._run_cmd(cmd)
            except Exception:
                _log("  - GDAL retile failed; falling back to pure-Python tiler.")
                self._python_retile(composite, out_dir, tile_size)
        else:
            _log("  - gdal_retile.py not on PATH — using pure-Python tiler.")
            self._python_retile(composite, out_dir, tile_size)
        
        return [str(p) for p in out_dir.glob("*.tif")]

    # ------------------------------------------------------------------
    # --- Internal Helpers
    # ------------------------------------------------------------------
    def _python_resample(self, input_path: str, output_path: Path, target_res: float) -> str:
        """Pure Python resampling fallback."""
        with rasterio.open(input_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, src.crs, src.width, src.height, *src.bounds,
                resolution=target_res
            )
            # FIX: Cast width and height to int to satisfy rasterio profile requirements.
            width = int(width)
            height = int(height)
            
            profile = src.profile.copy()
            profile.update({
                'transform': transform,
                'width': width,
                'height': height,
                'compress': 'lzw',
                'tiled': True,
                'BIGTIFF': 'IF_NEEDED'
            })
            
            with rasterio.open(str(output_path), 'w', **profile) as dst:
                for band_idx in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band_idx),
                        destination=rasterio.band(dst, band_idx),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=src.crs,
                        resampling=rasterio.enums.Resampling.cubic
                    )
        return str(output_path)

    def _harmonize_array_shapes(self, arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        """Resample array to target shape using proper rasterio reproject."""
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        
        _, h, w = arr.shape
        target_h, target_w = target_shape
        
        if h == target_h and w == target_w:
            return arr

        _log(f"    - Harmonizing shape from ({h}, {w}) to ({target_h}, {target_w})")
        destination = np.empty((arr.shape[0], target_h, target_w), dtype=arr.dtype)
        
        reproject(
            source=arr,
            destination=destination,
            src_transform=from_origin(0, 0, 1, 1),
            src_crs={'init': 'EPSG:4326'},
            dst_transform=from_origin(0, 0, w / target_w, h / target_h),
            dst_crs={'init': 'EPSG:4326'},
            resampling=rasterio.enums.Resampling.bilinear
        )
        return destination

    def _python_retile(self, src_path: str | Path, out_dir: Path, tile_size: int) -> None:
        """Pure-Python fallback for gdal_retile.py."""
        with rasterio.open(src_path) as src:
            profile = src.profile
            profile.update(
                driver='GTiff',
                height=tile_size,
                width=tile_size,
                tiled=True,
                blockxsize=min(tile_size, 256),
                blockysize=min(tile_size, 256),
                compress="lzw",
            )
            for j in range(0, src.height, tile_size):
                for i in range(0, src.width, tile_size):
                    window = Window(i, j, min(tile_size, src.width - i), min(tile_size, src.height - j))
                    data = src.read(window=window)
                    
                    if np.all(np.isnan(data)):
                        continue

                    tile_profile = profile.copy()
                    tile_profile['height'] = data.shape[1]
                    tile_profile['width'] = data.shape[2]
                    tile_profile['transform'] = src.window_transform(window)
                    
                    tile_fp = out_dir / f"{Path(src_path).stem}_{j}_{i}.tif"
                    with rasterio.open(tile_fp, "w", **tile_profile) as dst:
                        dst.write(data)

    def _apply_lee_filter(self, data, window_size=7):
        """Apply enhanced Lee filter for SAR speckle reduction."""
        filtered_bands = []
        for band_data in data:
            nan_mask = np.isnan(band_data)
            img = np.nan_to_num(band_data, nan=0.0)
            
            img_mean = np.mean(img[img > 0])
            img_var = np.var(img[img > 0])
            enl = (img_mean ** 2) / img_var if img_var > 0 else 1.0
            
            local_mean = ndi.uniform_filter(img.astype(np.float64), size=window_size)
            local_sq_mean = ndi.uniform_filter(img.astype(np.float64) ** 2, size=window_size)
            local_var = local_sq_mean - local_mean ** 2
            
            ci = np.sqrt(local_var) / (local_mean + 1e-10)
            cu = 1.0 / np.sqrt(enl)
            cmax = np.sqrt(1.0 + 2.0 / enl)
            
            w = np.where(ci <= cu, 0,
                np.where(ci >= cmax, 1,
                         np.exp(-0.5 * ((ci - cu) / (cmax - cu)) ** 2)))
            
            filtered_band = local_mean + w * (img - local_mean)
            filtered_band[nan_mask] = np.nan
            filtered_bands.append(filtered_band.astype(band_data.dtype))
        return np.array(filtered_bands)

    def _apply_destriping(self, data):
        """Apply a simple column-mean destriping filter."""
        destriped = data.copy().astype(np.float32)
        for b in range(data.shape[0]):
            col_mean = np.nanmean(data[b], axis=0)
            row_mean = np.nanmean(data[b])
            destriped[b] = data[b] - col_mean + row_mean
        return destriped

    def _save_raster(self, data, profile, output_path):
        """Save a numpy array to a GeoTIFF file."""
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)

    # ------------------------------------------------------------------
    # --- Public Orchestrator
    # ------------------------------------------------------------------
    def run_phase1_pipeline(self, sensor_input_path: str, output_base_path: str) -> list[str]:
        """Execute Phase 1 pipeline with progress monitoring."""
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        sensor_output_dir = Path(output_base_path) / self.sensor_name
        steps_output = {s: sensor_output_dir / f"step{s}" for s in range(1, 8)}
        for step_dir in steps_output.values():
            os.makedirs(step_dir, exist_ok=True)
        
        # FIX: Corrected argument passing for all pipeline steps.
        # The orchestrator now explicitly passes the output of one step
        # as the input to the next, making the data flow clear and correct.
        
        pbar = tqdm(total=7, desc=f"Processing {self.sensor_name}") if tqdm else None
        
        def update_pbar(name):
            if pbar:
                pbar.set_description(f"Processing {self.sensor_name}: {name}")
                pbar.update(1)

        update_pbar("Group & Stack")
        files = self.step1_group_and_stack(Path(sensor_input_path), steps_output[1])
        if not files: 
            if pbar: pbar.close()
            return []

        update_pbar("Resample")
        files = self.step2_resampling(files, steps_output[2])
        if not files: 
            if pbar: pbar.close()
            return []

        update_pbar("Mask & Filter")
        files = self.step3_mask_and_filter(files, steps_output[3])
        if not files: 
            if pbar: pbar.close()
            return []

        update_pbar("Temporal Stack")
        files = self.step4_temporal_stacking(files, steps_output[4])
        if not files: 
            if pbar: pbar.close()
            return []

        update_pbar("Smooth")
        files = self.step5_gap_filling_smoothing(files, steps_output[5])
        if not files: 
            if pbar: pbar.close()
            return []

        update_pbar("Composite")
        files = self.step6_compositing(files, steps_output[6])
        if not files: 
            if pbar: pbar.close()
            return []

        update_pbar("Tile")
        final_tiles = self.step7_tiling(files, steps_output[7])
        
        if pbar:
            pbar.close()
            
        return final_tiles
