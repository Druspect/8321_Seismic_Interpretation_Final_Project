# preprocess_data.py

import os
import numpy as np
import segyio
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from tqdm.auto import tqdm
import struct
import argparse

# --- Constants and Configuration ---
SEED = 42
np.random.seed(SEED)

# Default paths (can be overridden by arguments if argparse is used)
DEFAULT_BASE_DATA_DIR = "F3_Demo_2020"
DEFAULT_SEGY_FILENAME = "Rawdata/Seismic_data.sgy"
DEFAULT_HORIZON_SUBDIR = "Rawdata/Surface_data"
DEFAULT_HORIZON_FILENAMES = [
    "F3-Horizon-FS4.xyt.bz2",
    "F3-Horizon-MFS4.xyt",
    "F3-Horizon-FS6.xyt",
    "F3-Horizon-FS7.xyt",
    "F3-Horizon-FS8.xyt",
    "F3-Horizon-Shallow.xyt",
    "F3-Horizon-Top-Foresets.xyt"
]
DEFAULT_OUTPUT_DIR = "preprocessed_data"
DEFAULT_OUTPUT_FILENAME = "preprocessed_seismic_data.npz"

PATCH_SIZE = 32
STRIDE = 16
MAX_PATCHES = 50000

EPSILON = 1e-8
DEFAULT_LOWCUT = 5
DEFAULT_HIGHCUT = 60
DEFAULT_FILTER_ORDER = 4
HORIZON_COORD_SCALE_FACTOR = 10.0

# --- Helper Functions ---
def bandpass_filter(trace, lowcut=DEFAULT_LOWCUT, highcut=DEFAULT_HIGHCUT, fs=250, order=DEFAULT_FILTER_ORDER):
    """Applies a bandpass filter to a single trace."""
    nyq = 0.5 * fs
    if np.all(trace == trace[0]): # Constant trace
        return trace
    if np.isnan(trace).any() or np.isinf(trace).any():
        print("Warning: NaN or Inf found in trace, returning zeros.")
        return np.zeros_like(trace)
    # Check for valid frequency cuts relative to Nyquist frequency
    if lowcut <= 0 or highcut <= 0 or lowcut >= nyq or highcut >= nyq or lowcut >= highcut:
        print(f"Warning: Invalid frequency cuts ({lowcut}, {highcut}) for Nyquist {nyq}. Returning original trace or zeros if problematic.")
        # Depending on strictness, either return trace or np.zeros_like(trace)
        return trace # Or np.zeros_like(trace) if strict filtering is essential
    try:
        b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
        return filtfilt(b, a, trace)
    except ValueError as e:
        print(f"Warning: Filtering failed for a trace - {e}. Returning zeros.")
        return np.zeros_like(trace)

def load_and_preprocess_data(segy_path, horizon_files_full_paths, patch_size_val, stride_val, max_patches_val):
    """Loads SEG-Y data and horizon picks, processes amplitudes,
    extracts 3D patches, and returns X (N,1,D,H,W), y (N,), num_classes."""
    
    if not os.path.exists(segy_path):
        raise FileNotFoundError(f"SEG-Y file not found: {segy_path}")
    print("Checking horizon file paths…")
    valid_horizons = []
    for hf in horizon_files_full_paths:
        if os.path.exists(hf):
            print(f"  {hf} → FOUND")
            valid_horizons.append(hf)
        else:
            print(f"  {hf} → MISSING")
    if not valid_horizons:
        raise ValueError(f"No valid horizon files found; checked: {horizon_files_full_paths}")

    print("Loading SEG-Y data...")
    try:
        with segyio.open(segy_path, "r", ignore_geometry=True) as f:
            f.mmap() # Enable memory mapping for efficient access
            inlines = f.attributes(segyio.TraceField.INLINE_3D)[:] 
            xlines = f.attributes(segyio.TraceField.CROSSLINE_3D)[:] 
            raw_cdpX = f.attributes(segyio.TraceField.CDP_X)[:].astype(float)
            raw_cdpY = f.attributes(segyio.TraceField.CDP_Y)[:].astype(float)
            samples = np.array(f.samples) # Time samples for each trace
            sample_rate = segyio.tools.dt(f) / 1000.0 # Sample interval in ms
            fs = 1000.0 / sample_rate # Sampling frequency in Hz

            print(f"  Sample rate: {sample_rate} ms, Freq: {fs:.1f} Hz")
            print(f"  Raw CDP X range: {raw_cdpX.min():.3f}–{raw_cdpX.max():.3f}, " \
                  f"Raw CDP Y range: {raw_cdpY.min():.3f}–{raw_cdpY.max():.3f}")

            uni_il, il_counts = np.unique(inlines, return_counts=True)
            uni_xl, xl_counts = np.unique(xlines, return_counts=True)
            
            # Check for regularity (optional, but good for understanding structure)
            if not (il_counts.size > 0 and xl_counts.size > 0 and \
                    np.all(il_counts == il_counts[0]) and np.all(xl_counts == xl_counts[0])):
                print("  Warning: Irregular inline/xline distribution detected.")
            else:
                 print(f"  Grid appears regular.")

            n_ilines, n_xlines, n_samples = len(uni_il), len(uni_xl), len(samples)
            print(f"  Volume dims (Unique IL, Unique XL, Samples): {n_ilines}, {n_xlines}, {n_samples}")

            # Attempt to read coordinate scalar from binary header
            scalar = 1
            try:
                with open(segy_path, "rb") as raw_file:
                    raw_file.seek(3216) # Position for coordinate scalar in EBCDIC header
                    scalar_bytes = raw_file.read(2)
                    if len(scalar_bytes) == 2:
                        scalar_val = struct.unpack(">h", scalar_bytes)[0]
                        if scalar_val != 0: # Scalar of 0 is unlikely/problematic
                           scalar = scalar_val
                        print(f"  Coordinate scalar found: {scalar}")
                    else:
                        print("  Warning: Could not read enough bytes for coordinate scalar, assuming 1.")
            except Exception as e_scalar:
                print(f"  Warning: Could not read coordinate scalar due to: {e_scalar}. Assuming 1.")

            if scalar != 1:
                scaled_cdpX_display = raw_cdpX.copy()
                scaled_cdpY_display = raw_cdpY.copy()
                if scalar > 0:
                    scaled_cdpX_display *= scalar
                    scaled_cdpY_display *= scalar
                elif scalar < 0: # Negative scalar means division
                    scaled_cdpX_display /= abs(scalar)
                    scaled_cdpY_display /= abs(scalar)
                print(f"  Scaled CDP X range (for info): {scaled_cdpX_display.min():.3f}–{scaled_cdpX_display.max():.3f}, " \
                      f"Scaled CDP Y range (for info): {scaled_cdpY_display.min():.3f}–{scaled_cdpY_display.max():.3f}")

            print("Processing amplitudes…")
            # Initialize volume based on unique inline/crossline counts
            volume = np.zeros((n_ilines, n_xlines, n_samples), dtype=np.float32)
            # Create a mapping from (inline_val, xline_val) to trace_index in the original file
            trace_header_map = {(inlines[i], xlines[i]): i for i in range(f.tracecount)}
            # Create a mapping from unique inline/xline values to their 0-based indices in the volume
            il_map_val_to_idx = {val: idx for idx, val in enumerate(uni_il)}
            xl_map_val_to_idx = {val: idx for idx, val in enumerate(uni_xl)}

            for il_val in tqdm(uni_il, desc="    Inlines"):
                for xl_val in uni_xl:
                    original_trace_idx = trace_header_map.get((il_val, xl_val))
                    if original_trace_idx is not None:
                        trace = f.trace.raw[original_trace_idx].astype(np.float32)
                        filtered_trace = bandpass_filter(trace, fs=fs)
                        env = np.abs(hilbert(filtered_trace))
                        # Get the 0-based indices for the volume
                        vol_il_idx = il_map_val_to_idx[il_val]
                        vol_xl_idx = xl_map_val_to_idx[xl_val]
                        volume[vol_il_idx, vol_xl_idx, :] = env
            
            # Normalization (as in notebook)
            mask = volume != 0 # Avoid normalizing regions with no data / all zeros
            mean_val, std_val = 0.0, 1.0
            if mask.any():
                p1, p99 = np.percentile(volume[mask], [1, 99])
                volume = np.clip(volume, p1, p99)
                mean_val = volume[mask].mean()
                std_val = volume[mask].std()
                if std_val > EPSILON: # Avoid division by zero or very small number
                    volume[mask] = (volume[mask] - mean_val) / std_val
                else:
                    volume[mask] = 0 # Or handle as appropriate if std_dev is too small
            print(f"  Normalized stats (mean/std): {mean_val:.3f}/{std_val:.3f}")

    except Exception as e:
        print(f"Error loading or processing SEG-Y file: {e}")
        raise

    print("Building KD-Tree on raw SEG-Y trace coordinates...")
    # Use raw CDP_X, CDP_Y for KDTree as horizon coordinates will be scaled to match these
    coords_for_kdtree = np.column_stack((raw_cdpX, raw_cdpY))
    try:
        tree = cKDTree(coords_for_kdtree)
    except Exception as e:
        print(f"Error building KDTree: {e}")
        raise

    # Time to sample index mapping
    time_to_idx_interpolator = interp1d(samples, np.arange(n_samples), 
                                      kind="nearest", bounds_error=False, fill_value=-1)

    print("Loading and mapping horizons…")
    horizon_stack = np.full((len(valid_horizons), n_ilines, n_xlines), 
                            np.nan, dtype=float) # Store sample indices

    for h_idx, hf_path in enumerate(valid_horizons):
        try:
            df = pd.read_csv(hf_path, sep=r'\s+', header=None, 
                             names=["X", "Y", "time_ms"], engine='python', compression='infer')
            if df.isnull().values.any():
                 print(f"  Warning: NaNs detected in {os.path.basename(hf_path)}, dropping rows.")
                 df = df.dropna()
            if df.empty:
                print(f"  Warning: No valid data after NaNs dropped in {os.path.basename(hf_path)}. Skipping.")
                continue

            print(f"  Applying scale factor ({HORIZON_COORD_SCALE_FACTOR}) to coordinates from {os.path.basename(hf_path)}")
            df["X"] *= HORIZON_COORD_SCALE_FACTOR
            df["Y"] *= HORIZON_COORD_SCALE_FACTOR
            
            current_horizon_surface = np.full((n_ilines, n_xlines), np.nan, dtype=float)
            mapped_points = 0
            for Xval, Yval, t_ms in df.itertuples(index=False):
                try:
                    # Query KDTree with scaled horizon coordinates to find nearest trace in SEG-Y
                    dist, nearest_trace_original_idx = tree.query([Xval, Yval])
                    if nearest_trace_original_idx < 0 or nearest_trace_original_idx >= len(raw_cdpX):
                        continue # Invalid index from tree query

                    # Get the inline and xline values of this nearest trace
                    trace_il_val = inlines[nearest_trace_original_idx]
                    trace_xl_val = xlines[nearest_trace_original_idx]

                    # Convert these inline/xline values to 0-based indices for our volume
                    vol_il_idx = il_map_val_to_idx.get(trace_il_val)
                    vol_xl_idx = xl_map_val_to_idx.get(trace_xl_val)

                    if vol_il_idx is None or vol_xl_idx is None:
                        continue # Should not happen if maps are built correctly

                    sample_idx_float = time_to_idx_interpolator(t_ms)
                    if sample_idx_float != -1: # Check for fill_value
                        # FIX: Handle both scalar and array cases for sample_idx_float
                        if isinstance(sample_idx_float, np.ndarray):
                            s_idx = int(np.round(sample_idx_float.item()))  # Convert to scalar first
                        else:
                            s_idx = int(round(sample_idx_float))
                            
                        if 0 <= s_idx < n_samples and np.isnan(current_horizon_surface[vol_il_idx, vol_xl_idx]):
                            current_horizon_surface[vol_il_idx, vol_xl_idx] = s_idx
                            mapped_points += 1
                except Exception as e_point:
                    print(f"  Warning: Error processing point ({Xval}, {Yval}, {t_ms}) from {os.path.basename(hf_path)}: {e_point}")
                    continue
            
            horizon_stack[h_idx] = current_horizon_surface
            coverage = (~np.isnan(current_horizon_surface)).sum() / current_horizon_surface.size * 100 if current_horizon_surface.size > 0 else 0
            print(f"  {os.path.basename(hf_path)}: {coverage:.2f}% coverage ({mapped_points} points mapped to volume grid)")

        except Exception as e_file:
            print(f"  Error reading or processing horizon file {hf_path}: {e_file}. Skipping.")
            continue

    if np.all(np.isnan(horizon_stack)):
         raise ValueError("Horizon mapping resulted in zero coverage for all files. Check coordinate systems, scaling, and file contents.")

    print("Extracting patches and labels…")
    patches, labels = [], []
    half_patch_depth = patch_size_val // 2
    half_patch_spatial = patch_size_val // 2 # Assuming square spatial patch for now

    # Iterate over the grid of the processed volume
    for il_start_idx in range(0, n_ilines - patch_size_val + 1, stride_val):
        for xl_start_idx in range(0, n_xlines - patch_size_val + 1, stride_val):
            # Center of the current spatial window for label determination
            center_vol_il_idx = il_start_idx + half_patch_spatial
            center_vol_xl_idx = xl_start_idx + half_patch_spatial
            
            # Ensure center indices are within bounds (can happen with small volumes/large patches)
            if not (0 <= center_vol_il_idx < n_ilines and 0 <= center_vol_xl_idx < n_xlines):
                continue

            depths_at_center = horizon_stack[:, center_vol_il_idx, center_vol_xl_idx]
            valid_depths = np.sort(depths_at_center[~np.isnan(depths_at_center)])
            
            if valid_depths.size == 0: # No horizons at this center location
                continue

            for sm_start_idx in range(0, n_samples - patch_size_val + 1, stride_val):
                patch = volume[il_start_idx : il_start_idx + patch_size_val,
                               xl_start_idx : xl_start_idx + patch_size_val,
                               sm_start_idx : sm_start_idx + patch_size_val]
                
                if np.isfinite(patch).all(): # Check for NaNs or Infs in the patch itself
                    center_depth_sample = sm_start_idx + half_patch_depth
                    # Determine label based on where the center_depth_sample falls among sorted valid_depths
                    label = np.searchsorted(valid_depths, center_depth_sample, side='right')
                    
                    patches.append(patch[np.newaxis, ...]) # Add channel dimension (1)
                    labels.append(label)
                    
                    if len(patches) >= max_patches_val:
                        break 
            if len(patches) >= max_patches_val:
                break
        if len(patches) >= max_patches_val:
            break

    if not patches:
        raise ValueError("No valid patches extracted. Check patch_size, stride, horizon coverage, volume normalization, or max_patches limit.")
    print(f"  Extracted {len(patches)} patches.")

    X_data = np.stack(patches).astype(np.float32)
    y_data = np.array(labels, dtype=np.int64)
    
    # Remap labels to be 0-indexed and contiguous
    unique_labels_raw = np.unique(y_data)
    num_classes_val = len(unique_labels_raw)
    label_map = {raw_lab: i for i, raw_lab in enumerate(unique_labels_raw)}
    y_mapped_data = np.vectorize(label_map.get)(y_data)

    print(f"Final shapes → X: {X_data.shape}, y: {y_mapped_data.shape}, classes: {num_classes_val} (mapped from {unique_labels_raw})")
    return X_data, y_mapped_data, num_classes_val

def preprocess_seismic_data(base_data_dir=DEFAULT_BASE_DATA_DIR,
                           segy_filename=DEFAULT_SEGY_FILENAME,
                           horizon_subdir=DEFAULT_HORIZON_SUBDIR,
                           horizon_filenames=DEFAULT_HORIZON_FILENAMES,
                           output_dir=DEFAULT_OUTPUT_DIR,
                           output_filename=DEFAULT_OUTPUT_FILENAME,
                           patch_size=PATCH_SIZE,
                           stride=STRIDE,
                           max_patches=MAX_PATCHES):
    """
    Main function to preprocess seismic data and save the results.
    
    Args:
        base_data_dir (str): Base directory for the F3 dataset
        segy_filename (str): Relative path to SEG-Y file from base_data_dir
        horizon_subdir (str): Relative path to horizon files directory from base_data_dir
        horizon_filenames (list): List of horizon filenames
        output_dir (str): Directory to save preprocessed data
        output_filename (str): Filename for preprocessed data
        patch_size (int): Size of patches to extract
        stride (int): Stride for patch extraction
        max_patches (int): Maximum number of patches to extract
        
    Returns:
        tuple: (X_patches, y_labels, num_classes) - The preprocessed data
    """
    # Construct full paths
    segy_path = os.path.join(base_data_dir, segy_filename)
    horizon_dir = os.path.join(base_data_dir, horizon_subdir)
    horizon_paths = [os.path.join(horizon_dir, hf) for hf in horizon_filenames]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Processing seismic data from {segy_path}")
    print(f"Using horizon files from {horizon_dir}")
    print(f"Output will be saved to {output_path}")
    
    # Load and preprocess data
    X_patches, y_labels, num_classes = load_and_preprocess_data(
        segy_path, horizon_paths, patch_size, stride, max_patches
    )
    
    # Save preprocessed data
    print(f"Saving preprocessed data to {output_path}")
    np.savez_compressed(
        output_path,
        X_patches=X_patches,
        y_labels=y_labels,
        num_classes=num_classes
    )
    print("Preprocessing complete!")
    
    return X_patches, y_labels, num_classes

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess seismic data for deep learning.")
    parser.add_argument("--base_data_dir", type=str, default=DEFAULT_BASE_DATA_DIR, help="Base directory for the F3 dataset.")
    parser.add_argument("--segy_filename", type=str, default=DEFAULT_SEGY_FILENAME, help="Relative path to SEG-Y file from base_data_dir.")
    parser.add_argument("--horizon_subdir", type=str, default=DEFAULT_HORIZON_SUBDIR, help="Relative path to horizon files directory from base_data_dir.")
    parser.add_argument("--horizon_filenames", type=str, nargs="+", default=DEFAULT_HORIZON_FILENAMES, help="List of horizon filenames.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save preprocessed data.")
    parser.add_argument("--output_filename", type=str, default=DEFAULT_OUTPUT_FILENAME, help="Filename for preprocessed data.")
    parser.add_argument("--patch_size", type=int, default=PATCH_SIZE, help="Size of patches to extract.")
    parser.add_argument("--stride", type=int, default=STRIDE, help="Stride for patch extraction.")
    parser.add_argument("--max_patches", type=int, default=MAX_PATCHES, help="Maximum number of patches to extract.")
    
    args = parser.parse_args()
    
    preprocess_seismic_data(
        base_data_dir=args.base_data_dir,
        segy_filename=args.segy_filename,
        horizon_subdir=args.horizon_subdir,
        horizon_filenames=args.horizon_filenames,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        patch_size=args.patch_size,
        stride=args.stride,
        max_patches=args.max_patches
    )
