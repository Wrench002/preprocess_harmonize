# phase_1/sensor_preprocessor.py
import os
import numpy as np
import rasterio
import subprocess
from pathlib import Path
from scipy import ndimage as ndi

import scipy.sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter

# --- CORRECTED IMPORT ---
# We only need the direct import of the 'apply' function and 'QABits'
from eo_qamask.api import apply as qamask_apply
from eo_qamask.bitdefs import QABits

class SensorPreprocessor:
    """
    Handles all Phase 1 (per-sensor) preprocessing. This class is the final,
    integrated version combining the eo_qamask library with all advanced
    processing steps like resampling, noise filtering, and temporal smoothing.
    """
    def __init__(self, sensor_type, config):
        self.sensor_type = sensor_type
        self.main_config = config
        self.config_dir = os.path.abspath('configs')
        self.sensor_params = self._get_sensor_params()

    def _run_cmd(self, cmd):
        """Utility to run a command and check for errors."""
        use_shell = os.name == 'nt'
        result = subprocess.run(cmd, capture_output=True, text=True, shell=use_shell)
        if result.returncode != 0:
            print(f"ERROR running command: {' '.join(cmd)}")
            print(result.stderr)
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
        return result

    @staticmethod
    def whittaker_smoother(y, lambd=10.0, d=2):
        """
        An implementation of the Whittaker-Eilers smoother and interpolator.
        """
        m = len(y)
        w = ~np.isnan(y)
        w_sparse = scipy.sparse.spdiags(w.astype(float), 0, m, m)
        d_matrix = scipy.sparse.eye(m, format='csc')
        for i in range(d):
            d_matrix = d_matrix[1:] - d_matrix[:-1]
        a_matrix = w_sparse + lambd * (d_matrix.T @ d_matrix)
        y_no_nan = np.nan_to_num(y)
        z = spsolve(a_matrix, w_sparse @ y_no_nan)
        return z

    def _apply_lee_filter(self, data, window_size=7):
        """
        Applies a full Lee filter to reduce speckle noise in SAR data.
        """
        filtered_bands = []
        for band_data in data:
            nan_mask = np.isnan(band_data)
            img = np.nan_to_num(band_data)
            sigma_v_sq = np.var(img)
            local_mean = ndi.uniform_filter(img, size=window_size)
            local_sq_mean = ndi.uniform_filter(img**2, size=window_size)
            local_var = local_sq_mean - local_mean**2
            k = 1 - (sigma_v_sq / np.where(local_var == 0, 1e-6, local_var))
            k = np.clip(k, 0, 1)
            filtered_band = local_mean + k * (img - local_mean)
            filtered_band[nan_mask] = np.nan
            filtered_bands.append(filtered_band)
        return np.array(filtered_bands)

    def _apply_destriping(self, data):
        """Removes column-wise striping artifacts."""
        destriped = data.copy().astype(np.float32)
        for b in range(data.shape[0]):
            col_mean = np.nanmean(data[b], axis=0)
            row_mean = np.nanmean(data[b])
            destriped[b] = data[b] - col_mean + row_mean
        return destriped

    def _save_raster(self, data, profile, output_path):
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)

    # --- PIPELINE STEPS ---

    def step1_ingest_data(self, input_path, output_path):
        print(f"\nStep 1: Ingesting and Scaling {self.sensor_type} data...")
        ingested_files = []
        for file_path in Path(input_path).rglob("*.tif"):
            with rasterio.open(file_path) as src:
                data = src.read()
                profile = src.profile
                tags = src.tags()
                scale_factor_str = tags.get('ScaleFactor')
                bits_per_pixel_str = tags.get('BitsPerPixel')
                if scale_factor_str:
                    try:
                        scale_factor = float(scale_factor_str)
                        print(f"  - Found ScaleFactor: {scale_factor}. Applying scaling.")
                        data = data.astype(np.float32) * scale_factor
                        profile['dtype'] = 'float32'
                    except (ValueError, TypeError):
                        print(f"  - WARNING: Could not parse ScaleFactor '{scale_factor_str}'.")
                elif bits_per_pixel_str:
                    try:
                        bits = int(bits_per_pixel_str)
                        max_val = (2**bits) - 1
                        print(f"  - No ScaleFactor. Found BitsPerPixel: {bits}. Normalizing DN by dividing by {max_val}.")
                        data = data.astype(np.float32) / max_val
                        profile['dtype'] = 'float32'
                    except (ValueError, TypeError):
                        print(f"  - WARNING: Could not parse BitsPerPixel '{bits_per_pixel_str}'.")
                else:
                    print("  - No scaling information found. Assuming data is already scaled (0-1).")
            output_file = os.path.join(output_path, f"ingested_{file_path.name}")
            self._save_raster(data, profile, output_file)
            ingested_files.append(output_file)
        return ingested_files

    def step2_qa_masking(self, input_files, output_path):
        print(f"\nStep 2: Applying QA masks via eo_qamask...")
        masked_files = []
        for file_path in input_files:
            config_path = os.path.join(self.config_dir, f"{self.sensor_type.lower()}_config.yaml")
            if not os.path.exists(config_path):
                masked_files.append(file_path)
                continue
            
            qa_mask, profile = qamask_apply(file_path, config_path)
            
            with rasterio.open(file_path) as src: data = src.read().astype(np.float32)
            
            bad_pixel_filter = ((qa_mask & (1 << QABits.CLOUD.value)) | (qa_mask & (1 << QABits.SHADOW.value))).astype(bool)
            data[:, bad_pixel_filter] = np.nan
            
            masked_path = os.path.join(output_path, f"masked_{os.path.basename(file_path)}")
            profile.update(dtype=rasterio.float32, count=data.shape[0], nodata=np.nan)
            self._save_raster(data, profile, masked_path)
            masked_files.append(masked_path)
        return masked_files

    def step3_resampling(self, input_files, output_path):
        print(f"\nStep 3: Resampling to native resolution...")
        resampled_files = []
        target_res = self.sensor_params['resolution']
        for file_path in input_files:
            output_file = os.path.join(output_path, f"resampled_{os.path.basename(file_path)}")
            cmd = ['gdalwarp', '-tr', str(target_res), str(target_res), '-r', 'cubic', file_path, output_file]
            self._run_cmd(cmd)
            resampled_files.append(output_file)
        return resampled_files

    def step4_noise_filtering(self, input_files, output_path):
        print(f"\nStep 4: Applying noise filtering...")
        filtered_files = []
        for file_path in input_files:
            with rasterio.open(file_path) as src: data, profile = src.read(), src.profile
            if self.sensor_type == 'SAR':
                print(f"  - Applying Lee speckle filter to {os.path.basename(file_path)}...")
                processed_data = self._apply_lee_filter(data)
            elif self.sensor_type in ['LISS3', 'LISS4']:
                print(f"  - Applying destriping to {os.path.basename(file_path)}...")
                processed_data = self._apply_destriping(data)
            else:
                processed_data = data
            output_file = os.path.join(output_path, f"filtered_{os.path.basename(file_path)}")
            self._save_raster(processed_data, profile, output_file)
            filtered_files.append(output_file)
        return filtered_files

    def step5_temporal_stacking(self, input_files, output_path):
        print(f"\nStep 5: Stacking images by time period...")
        if not input_files: return []
        input_files.sort()
        with rasterio.open(input_files[0]) as ref: profile = ref.profile
        all_data = []
        for f in input_files:
            with rasterio.open(f) as src:
                all_data.append(src.read())
        stack_data = np.vstack(all_data)
        output_file = os.path.join(output_path, f"{self.sensor_type}_stack.tif")
        profile.update(count=stack_data.shape[0])
        self._save_raster(stack_data, profile, output_file)
        return [output_file]

    def step6_gap_filling_smoothing(self, input_files, output_path):
        print(f"\nStep 6: Applying temporal smoothing and gap-filling...")
        if not input_files: return []
        stack_file = input_files[0]
        with rasterio.open(stack_file) as src:
            data, profile = src.read(), src.profile
        n_bands, height, width = data.shape
        print(f"  - Smoothing a stack with {n_bands} time-steps...")
        pixels_as_cols = data.reshape(n_bands, height * width)
        smoothed_pixels = np.zeros_like(pixels_as_cols)
        whittaker_lambda = 10.0
        savgol_window = 5
        savgol_poly = 2
        if savgol_window >= n_bands:
            print(f"  - WARNING: Sav-Gol window ({savgol_window}) is too large. Skipping Sav-Gol filter.")
            savgol_window = -1
        total_pixels = pixels_as_cols.shape[1]
        for i in range(total_pixels):
            pixel_series = pixels_as_cols[:, i]
            smoothed_series = self.whittaker_smoother(pixel_series, lambd=whittaker_lambda)
            if savgol_window > 0:
                smoothed_series = savgol_filter(smoothed_series, window_length=savgol_window, polyorder=savgol_poly)
            smoothed_pixels[:, i] = smoothed_series
            if (i + 1) % (total_pixels // 10) == 0:
                print(f"  - Smoothed {int(((i + 1) / total_pixels) * 100)}% of pixels...")
        smoothed_data = smoothed_pixels.reshape(n_bands, height, width)
        output_file = os.path.join(output_path, f"smoothed_{os.path.basename(stack_file)}")
        self._save_raster(smoothed_data, profile, output_file)
        return [output_file]

    def step7_compositing(self, input_files, output_path):
        print(f"\nStep 7: Creating final composite image...")
        if not input_files: return []
        stack_file = input_files[0]
        with rasterio.open(stack_file) as src: data, profile = src.read(), src.profile
        composite = np.nanmean(data, axis=0)
        output_file = os.path.join(output_path, f"{self.sensor_type}_composite.tif")
        profile.update(count=1, dtype='float32')
        self._save_raster(composite[np.newaxis, :, :], profile, output_file)
        return [output_file]

    def step8_tiling(self, input_files, output_path):
        print(f"\nStep 8: Tiling final composite...")
        if not input_files: return []
        composite_file = input_files[0]
        tile_size = self.sensor_params.get('tile_size', 512)
        
        retile_cmd = ['gdal_retile.py', '-ps', str(tile_size), str(tile_size), '-co', 'TILED=YES', '-targetDir', output_path, composite_file]
        
        # --- CORRECTED: Use the correct variable name 'retile_cmd' ---
        self._run_cmd(retile_cmd)
        
        return [str(p) for p in Path(output_path).glob("*.tif")]

    def run_phase1_pipeline(self, input_path, output_base_path):
        """The main orchestrator for the complete Phase 1 workflow."""
        print(f"\n{'='*20} Starting Phase 1 for {self.sensor_type} {'='*20}")
        steps = range(1, 9)
        steps_output = {s: os.path.join(output_base_path, self.sensor_type, f"step{s}") for s in steps}
        for step_dir in steps_output.values(): os.makedirs(step_dir, exist_ok=True)
        try:
            files = self.step1_ingest_data(input_path, steps_output[1])
            files = self.step2_qa_masking(files, steps_output[2])
            files = self.step3_resampling(files, steps_output[3])
            files = self.step4_noise_filtering(files, steps_output[4])
            files = self.step5_temporal_stacking(files, steps_output[5])
            files = self.step6_gap_filling_smoothing(files, steps_output[6])
            files = self.step7_compositing(files, steps_output[7])
            final_tiles = self.step8_tiling(files, steps_output[8])
            print(f"\n✓ Phase 1 for {self.sensor_type} produced {len(final_tiles)} final tiles.")
            return final_tiles
        except Exception as e:
            print(f"\n✗ Phase 1 FAILED for {self.sensor_type}.")
            raise e

    def _get_sensor_params(self):
        """Loads detailed sensor parameters."""
        return {
            'AWiFS': {'resolution': 56, 'tile_size': 1024, 'bands': ['Green', 'Red', 'NIR', 'SWIR']},
            'LISS3': {'resolution': 23.5, 'tile_size': 512, 'bands': ['Green', 'Red', 'NIR', 'SWIR']},
            'LISS4': {'resolution': 5.8, 'tile_size': 512, 'bands': ['Green', 'Red', 'NIR']},
            'SAR': {'resolution': 10, 'tile_size': 512, 'bands': ['VV', 'VH']},
            'Landsat8': {'resolution': 30, 'tile_size': 512, 'bands': ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']},
            'Sentinel2': {'resolution': 10, 'tile_size': 512, 'bands': ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']}
        }.get(self.sensor_type, {})