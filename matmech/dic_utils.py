"""
DIC (Digital Image Correlation) Data Extraction Utility

This module provides functions to extract strain data from LaVision DaVis .vc7 files
and convert them to usable formats (CSV, NumPy arrays).

The main outputs are:
1. Time-series CSV with spatially-averaged strain data per frame
2. Full 3D arrays (time x grid_y x grid_x) for each strain component
3. Individual frame data as 2D grids

Author: Auto-generated for matmech library
"""

import os
import glob
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import lvpyio
    HAS_LVPYIO = True
except ImportError:
    HAS_LVPYIO = False
    logging.warning("lvpyio not installed. Install with: pip install lvpyio")


# Constants for strain component names in DaVis VC7 files
STRAIN_COMPONENTS = {
    "exx": "TS:Exx",           # Normal strain in X direction  
    "eyy": "TS:Eyy",           # Normal strain in Y direction
    "exy": "TS:Exy",           # Shear strain component
    "eyx": "TS:Eyx",           # Shear strain component (should equal exy)
    "exx_rot_free": "TS:Exx (rotation free)",  # Rotation-free normal strain X
    "eyy_rot_free": "TS:Eyy (rotation free)",  # Rotation-free normal strain Y
    "exy_rot_free": "TS:Exy (rotation free)",  # Rotation-free shear strain
    "correlation": "TS:Correlation value",      # Correlation quality metric
    "confidence": "TS:Confidence region",       # Confidence region
}

DISPLACEMENT_COMPONENTS = {
    "u": "U0",  # Displacement in X
    "v": "V0",  # Displacement in Y
}


@dataclass
class DICFrameData:
    """Container for a single DIC frame's data."""
    frame_number: int
    acquisition_time_us: float  # Time in microseconds
    grid_x: np.ndarray          # X coordinates of grid points (mm)
    grid_y: np.ndarray          # Y coordinates of grid points (mm)
    strain_exx: np.ndarray      # 2D array of Exx strain
    strain_eyy: np.ndarray      # 2D array of Eyy strain
    strain_exy: np.ndarray      # 2D array of Exy strain
    displacement_u: np.ndarray  # 2D array of U displacement (mm)
    displacement_v: np.ndarray  # 2D array of V displacement (mm)
    mask: np.ndarray            # Valid data mask (True = valid)


def check_lvpyio():
    """Check if lvpyio is installed."""
    if not HAS_LVPYIO:
        raise ImportError(
            "lvpyio is required to read VC7 files. "
            "Install it with: pip install lvpyio"
        )


def read_vc7_file(filepath: str) -> DICFrameData:
    """
    Read a single VC7 file and extract strain/displacement data.
    
    Args:
        filepath: Path to the .vc7 file
        
    Returns:
        DICFrameData containing all extracted fields
    """
    check_lvpyio()
    
    buffer = lvpyio.read_buffer(filepath)
    
    if len(buffer.frames) == 0:
        raise ValueError(f"No frames found in {filepath}")
    
    frame = buffer.frames[0]
    
    # Extract frame number from filename (e.g., B00002.vc7 -> 2)
    basename = os.path.basename(filepath)
    frame_num = int(basename[1:6])  # Extract 00002 from B00002.vc7
    
    # Get acquisition time
    acq_time_us = 0.0
    if 'AcqTimeSeries' in frame.attributes:
        time_str = frame.attributes['AcqTimeSeries']
        # Parse "73401 µs" format
        try:
            acq_time_us = float(time_str.split()[0])
        except (ValueError, IndexError):
            pass
    
    # Build coordinate grids from scales
    scales = frame.scales
    ny, nx = frame.shape
    
    # Create coordinate arrays
    grid_x = np.arange(nx) * scales.x.slope + scales.x.offset
    grid_y = np.arange(ny) * scales.y.slope + scales.y.offset
    
    # Access all component data via frame[0] which returns a dict
    # Keys are component names like 'TS:Exx', 'U0', 'MASK', etc.
    data_dict = frame[0]
    
    def get_component(name: str) -> np.ndarray:
        """Safely extract a component, returning NaN array if not found."""
        if name in data_dict:
            return np.array(data_dict[name], dtype=np.float64)
        return np.full((ny, nx), np.nan)
    
    # Extract main strain/displacement data
    strain_exx = get_component("TS:Exx")
    strain_eyy = get_component("TS:Eyy")
    strain_exy = get_component("TS:Exy")
    displacement_u = get_component("U0")
    displacement_v = get_component("V0")
    
    # Get mask (valid data points)
    if "MASK" in data_dict:
        mask = np.array(data_dict["MASK"]) > 0
    else:
        mask = np.ones((ny, nx), dtype=bool)
    
    return DICFrameData(
        frame_number=frame_num,
        acquisition_time_us=acq_time_us,
        grid_x=grid_x,
        grid_y=grid_y,
        strain_exx=strain_exx,
        strain_eyy=strain_eyy,
        strain_exy=strain_exy,
        displacement_u=displacement_u,
        displacement_v=displacement_v,
        mask=mask,
    )



def extract_dic_timeseries(
    strain_dir: str,
    output_csv: Optional[str] = None,
    frame_range: Optional[Tuple[int, int]] = None,
) -> pd.DataFrame:
    """
    Extract time-series of strain from all VC7 files in a directory.
    
    This creates a CSV with one row per frame, containing:
    - Spatially-averaged strain values (mean_exx, mean_eyy, etc.)
    - Center-line strain values (DIC_normal_strain, DIC_transverse_strain)
      which are averages along the center vertical and horizontal lines
      of the sample, providing more meaningful strain measurements.
    
    Args:
        strain_dir: Directory containing .vc7 files (e.g., "Strain2")
        output_csv: Optional path to save the output CSV
        frame_range: Optional (start, end) tuple to limit frames processed
        
    Returns:
        DataFrame with strain time-series data
    """
    check_lvpyio()
    
    # Find all VC7 files
    vc7_files = sorted(glob.glob(os.path.join(strain_dir, "B*.vc7")))
    
    if not vc7_files:
        raise FileNotFoundError(f"No VC7 files found in {strain_dir}")
    
    logging.info(f"Found {len(vc7_files)} VC7 files in {strain_dir}")
    
    # Apply frame range filter
    if frame_range:
        start, end = frame_range
        vc7_files = [f for f in vc7_files 
                     if start <= int(os.path.basename(f)[1:6]) <= end]
        logging.info(f"Processing frames {start} to {end} ({len(vc7_files)} files)")
    
    records = []
    
    for i, filepath in enumerate(vc7_files):
        try:
            frame_data = read_vc7_file(filepath)
            
            # Apply mask for statistics
            mask = frame_data.mask
            
            def masked_stats(arr):
                valid = arr[mask] if mask.any() else arr.flatten()
                valid = valid[np.isfinite(valid)]
                if len(valid) == 0:
                    return np.nan, np.nan
                return np.nanmean(valid), np.nanstd(valid)
            
            mean_exx, std_exx = masked_stats(frame_data.strain_exx)
            mean_eyy, std_eyy = masked_stats(frame_data.strain_eyy)
            mean_exy, std_exy = masked_stats(frame_data.strain_exy)
            mean_u, _ = masked_stats(frame_data.displacement_u)
            mean_v, _ = masked_stats(frame_data.displacement_v)
            
            valid_count = np.sum(mask & np.isfinite(frame_data.strain_exx))
            
            # --- Center-line strain extraction ---
            # Get grid dimensions
            ny, nx = frame_data.strain_eyy.shape

            # Use centroid of valid data to define "center"
            # This handles cases where the sample is not perfectly centered in the ROI

            valid_y_idxs, valid_x_idxs = np.where(mask)
            
            if len(valid_y_idxs) > 0:
                # Find "center" of valid region
                center_y = int(np.median(valid_y_idxs))
                center_x = int(np.median(valid_x_idxs))
                
                # Normal strain (along Y-axis) - take median column of valid area
                # We use the column that passes continuously through the most valid points
                col_counts = np.sum(mask, axis=0)
                best_col_idx = np.argmax(col_counts)
                
                # If median X is significantly different from best_col, prefer best_col 
                # (column with most data)
                center_col_idx = best_col_idx if col_counts[best_col_idx] > col_counts[center_x] else center_x
                
                center_col_eyy = frame_data.strain_eyy[:, center_col_idx]
                center_col_mask = mask[:, center_col_idx]
                valid_eyy = center_col_eyy[center_col_mask & np.isfinite(center_col_eyy)]
                dic_normal_strain = np.nanmean(valid_eyy) if len(valid_eyy) > 0 else np.nan
                
                # Transverse strain (along X-axis) - take median row of valid area
                # We check a few rows around center_y to find one with good data coverage
                best_row_idx = center_y
                max_valid_pts = -1
                
                # Search +/- 5 rows around center to find best transverse line
                search_range = range(max(0, center_y-5), min(ny, center_y+6))
                for r in search_range:
                    pts = np.sum(mask[r, :])
                    if pts > max_valid_pts:
                        max_valid_pts = pts
                        best_row_idx = r
                
                center_row_exx = frame_data.strain_exx[best_row_idx, :]
                center_row_mask = mask[best_row_idx, :]
                valid_exx = center_row_exx[center_row_mask & np.isfinite(center_row_exx)]
                dic_transverse_strain = np.nanmean(valid_exx) if len(valid_exx) > 0 else np.nan
            else:
                dic_normal_strain = np.nan
                dic_transverse_strain = np.nan

            # Calculate instantaneous Poisson's ratio
            # ν = -ε_transverse / ε_axial
            if dic_normal_strain != 0 and np.isfinite(dic_normal_strain) and np.isfinite(dic_transverse_strain):
                poisson_ratio = -dic_transverse_strain / dic_normal_strain
            else:
                poisson_ratio = np.nan
            
            records.append({
                "frame": frame_data.frame_number,

                "time_us": frame_data.acquisition_time_us,
                "time_s": frame_data.acquisition_time_us / 1e6,
                "DIC_normal_strain": dic_normal_strain,
                "DIC_transverse_strain": dic_transverse_strain,
                "DIC_poisson_ratio": poisson_ratio,
                "mean_exx": mean_exx,
                "mean_eyy": mean_eyy,
                "mean_exy": mean_exy,
                "std_exx": std_exx,
                "std_eyy": std_eyy,
                "std_exy": std_exy,
                "mean_u_mm": mean_u,
                "mean_v_mm": mean_v,
                "valid_points": valid_count,
            })
            
            if (i + 1) % 500 == 0:
                logging.info(f"Processed {i + 1}/{len(vc7_files)} frames...")
                
        except Exception as e:
            logging.warning(f"Error processing {filepath}: {e}")
            continue
    
    df = pd.DataFrame(records)
    
    if output_csv:
        df.to_csv(output_csv, index=False)
        logging.info(f"Saved time-series to {output_csv}")
    
    return df




def extract_full_strain_cube(
    strain_dir: str,
    component: str = "exx",
    frame_range: Optional[Tuple[int, int]] = None,
    output_npz: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract a full 3D array (time x y x x) of strain data.
    
    This is useful for detailed spatial-temporal analysis of strain evolution.
    
    Args:
        strain_dir: Directory containing .vc7 files
        component: Strain component to extract ("exx", "eyy", "exy", "u", "v")
        frame_range: Optional (start, end) tuple to limit frames
        output_npz: Optional path to save as compressed NumPy archive
        
    Returns:
        Dict with keys: 'data' (3D array), 'time_us', 'grid_x', 'grid_y', 'frames'
    """
    check_lvpyio()
    
    vc7_files = sorted(glob.glob(os.path.join(strain_dir, "B*.vc7")))
    
    if not vc7_files:
        raise FileNotFoundError(f"No VC7 files found in {strain_dir}")
    
    if frame_range:
        start, end = frame_range
        vc7_files = [f for f in vc7_files 
                     if start <= int(os.path.basename(f)[1:6]) <= end]
    
    logging.info(f"Extracting '{component}' from {len(vc7_files)} frames...")
    
    # Read first file to get dimensions
    first_frame = read_vc7_file(vc7_files[0])
    ny, nx = first_frame.strain_exx.shape
    n_frames = len(vc7_files)
    
    # Pre-allocate arrays
    data_cube = np.full((n_frames, ny, nx), np.nan, dtype=np.float64)
    time_array = np.zeros(n_frames)
    frame_numbers = np.zeros(n_frames, dtype=np.int32)
    
    # Map component name to attribute
    component_map = {
        "exx": "strain_exx",
        "eyy": "strain_eyy", 
        "exy": "strain_exy",
        "u": "displacement_u",
        "v": "displacement_v",
    }
    
    if component not in component_map:
        raise ValueError(f"Unknown component: {component}. Choose from {list(component_map.keys())}")
    
    attr_name = component_map[component]
    
    for i, filepath in enumerate(vc7_files):
        try:
            frame_data = read_vc7_file(filepath)
            data_cube[i] = getattr(frame_data, attr_name)
            time_array[i] = frame_data.acquisition_time_us
            frame_numbers[i] = frame_data.frame_number
            
            if (i + 1) % 500 == 0:
                logging.info(f"Extracted {i + 1}/{n_frames} frames...")
                
        except Exception as e:
            logging.warning(f"Error processing {filepath}: {e}")
    
    result = {
        "data": data_cube,
        "time_us": time_array,
        "grid_x": first_frame.grid_x,
        "grid_y": first_frame.grid_y,
        "frames": frame_numbers,
        "component": component,
    }
    
    if output_npz:
        np.savez_compressed(output_npz, **result)
        logging.info(f"Saved strain cube to {output_npz}")
    
    return result


def export_single_frame_csv(
    vc7_file: str,
    output_csv: str,
) -> pd.DataFrame:
    """
    Export a single VC7 frame as a CSV with x, y coordinates and all strain values.
    
    Args:
        vc7_file: Path to the .vc7 file
        output_csv: Path for output CSV
        
    Returns:
        DataFrame with columns: x, y, exx, eyy, exy, u, v, valid
    """
    check_lvpyio()
    
    frame_data = read_vc7_file(vc7_file)
    
    # Create meshgrid
    xx, yy = np.meshgrid(frame_data.grid_x, frame_data.grid_y)
    
    # Flatten all arrays
    df = pd.DataFrame({
        "x_mm": xx.flatten(),
        "y_mm": yy.flatten(),
        "exx": frame_data.strain_exx.flatten(),
        "eyy": frame_data.strain_eyy.flatten(),
        "exy": frame_data.strain_exy.flatten(),
        "u_mm": frame_data.displacement_u.flatten(),
        "v_mm": frame_data.displacement_v.flatten(),
        "valid": frame_data.mask.flatten(),
    })
    
    df.to_csv(output_csv, index=False)
    logging.info(f"Exported frame {frame_data.frame_number} to {output_csv}")
    
    return df


# ============================================================================
# DATA COMBINATION FUNCTIONS
# ============================================================================

def merge_dic_with_instron(
    dic_csv: str,
    instron_csv: str,
    output_csv: Optional[str] = None,
    dic_time_col: str = "time_s",
    instron_time_col: str = "Time (s)",
    tolerance_s: float = 0.05,
) -> pd.DataFrame:
    """
    Merge DIC strain time-series with Instron force/stress data.
    
    Uses nearest-neighbor time matching to align DIC frames with Instron data.
    
    Args:
        dic_csv: Path to DIC strain time-series CSV
        instron_csv: Path to Instron data CSV (standardized format)
        output_csv: Optional path to save merged data
        dic_time_col: Time column name in DIC data
        instron_time_col: Time column name in Instron data
        tolerance_s: Max time difference (seconds) for matching
        
    Returns:
        Merged DataFrame with both DIC and Instron columns
    """
    # Load data
    dic_df = pd.read_csv(dic_csv)
    instron_df = pd.read_csv(instron_csv)
    
    # Ensure time columns exist
    if dic_time_col not in dic_df.columns:
        raise ValueError(f"DIC data missing time column: {dic_time_col}")
    if instron_time_col not in instron_df.columns:
        # Try to find a time column
        time_cols = [c for c in instron_df.columns if "time" in c.lower()]
        if time_cols:
            instron_time_col = time_cols[0]
            logging.info(f"Using Instron time column: {instron_time_col}")
        else:
            raise ValueError(f"Instron data missing time column")
    
    # Perform asof merge (nearest time match)
    dic_df = dic_df.sort_values(dic_time_col)
    instron_df = instron_df.sort_values(instron_time_col)
    
    # Rename Instron time column temporarily for merge
    instron_df = instron_df.rename(columns={instron_time_col: "_inst_time"})
    
    # Use merge_asof for nearest-time matching
    merged = pd.merge_asof(
        dic_df,
        instron_df,
        left_on=dic_time_col,
        right_on="_inst_time",
        direction="nearest",
        tolerance=tolerance_s,
    )
    
    # Calculate time difference for quality check
    if "_inst_time" in merged.columns:
        merged["time_diff_s"] = merged[dic_time_col] - merged["_inst_time"]
        merged = merged.drop(columns=["_inst_time"])
    
    logging.info(f"Merged {len(merged)} DIC frames with Instron data")
    
    # Report match quality
    if "time_diff_s" in merged.columns:
        max_diff = merged["time_diff_s"].abs().max()
        mean_diff = merged["time_diff_s"].abs().mean()
        logging.info(f"Time match quality: mean={mean_diff*1000:.1f}ms, max={max_diff*1000:.1f}ms")
    
    if output_csv:
        merged.to_csv(output_csv, index=False)
        logging.info(f"Saved merged data to {output_csv}")
    
    return merged


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_strain_field(
    vc7_file: str,
    component: str = "eyy",
    output_path: Optional[str] = None,
    cmap: str = "RdYlBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
) -> None:
    """
    Create a 2D heatmap of strain field from a single VC7 frame.
    
    Args:
        vc7_file: Path to the .vc7 file
        component: Strain component to plot ("exx", "eyy", "exy")
        output_path: Path to save the figure (optional)
        cmap: Matplotlib colormap name
        vmin, vmax: Color scale limits (auto if None)
        title: Plot title (auto-generated if None)
        figsize: Figure size in inches
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        raise ImportError("matplotlib required for visualization")
    
    frame_data = read_vc7_file(vc7_file)
    
    # Select component
    component_map = {
        "exx": (frame_data.strain_exx, "εxx (Lateral Strain)"),
        "eyy": (frame_data.strain_eyy, "εyy (Axial Strain)"),
        "exy": (frame_data.strain_exy, "εxy (Shear Strain)"),
        "u": (frame_data.displacement_u, "U Displacement (mm)"),
        "v": (frame_data.displacement_v, "V Displacement (mm)"),
    }
    
    if component not in component_map:
        raise ValueError(f"Unknown component: {component}")
    
    data, label = component_map[component]
    
    # Apply mask
    masked_data = np.where(frame_data.mask, data, np.nan)
    
    # Create meshgrid for plotting
    xx, yy = np.meshgrid(frame_data.grid_x, frame_data.grid_y)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine color scale
    if vmin is None:
        vmin = np.nanpercentile(masked_data, 2)
    if vmax is None:
        vmax = np.nanpercentile(masked_data, 98)
    
    # Use diverging norm if data spans zero
    if vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = None
    
    # Plot heatmap
    im = ax.pcolormesh(xx, yy, masked_data, cmap=cmap, norm=norm,
                       vmin=vmin if norm is None else None,
                       vmax=vmax if norm is None else None,
                       shading="auto")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label=label)
    
    # Labels
    ax.set_xlabel("X Position (mm)")
    ax.set_ylabel("Y Position (mm)")
    ax.set_aspect("equal")
    
    if title is None:
        title = f"Frame {frame_data.frame_number} - {label}"
    ax.set_title(title)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logging.info(f"Saved strain field plot to {output_path}")
        plt.close()
    else:
        plt.show()


def create_strain_animation(
    strain_dir: str,
    output_path: str,
    component: str = "eyy",
    frame_range: Optional[Tuple[int, int]] = None,
    frame_step: int = 10,
    fps: int = 30,
    cmap: str = "RdYlBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[float, float] = (10, 8),
) -> None:
    """
    Create an animation of strain field evolution over time.
    
    Args:
        strain_dir: Directory containing .vc7 files
        output_path: Path for output video (mp4, gif, etc.)
        component: Strain component to animate
        frame_range: Optional (start, end) frame numbers
        frame_step: Process every Nth frame (for speed)
        fps: Frames per second in output video
        cmap: Matplotlib colormap
        vmin, vmax: Fixed color scale limits (recommended for consistency)
        figsize: Figure size in inches
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        raise ImportError("matplotlib required for visualization")
    
    check_lvpyio()
    
    # Find and filter VC7 files
    vc7_files = sorted(glob.glob(os.path.join(strain_dir, "B*.vc7")))
    
    if frame_range:
        start, end = frame_range
        vc7_files = [f for f in vc7_files 
                     if start <= int(os.path.basename(f)[1:6]) <= end]
    
    # Apply step
    vc7_files = vc7_files[::frame_step]
    
    logging.info(f"Creating animation from {len(vc7_files)} frames...")
    
    # Read first frame for setup
    first_frame = read_vc7_file(vc7_files[0])
    xx, yy = np.meshgrid(first_frame.grid_x, first_frame.grid_y)
    
    component_map = {
        "exx": ("strain_exx", "εxx"),
        "eyy": ("strain_eyy", "εyy"),
        "exy": ("strain_exy", "εxy"),
    }
    
    if component not in component_map:
        raise ValueError(f"Unknown component: {component}")
    
    attr_name, label = component_map[component]
    
    # Auto-detect scale from sample frames if not provided
    if vmin is None or vmax is None:
        sample_indices = np.linspace(0, len(vc7_files)-1, min(20, len(vc7_files)), dtype=int)
        all_vals = []
        for idx in sample_indices:
            frame = read_vc7_file(vc7_files[idx])
            data = getattr(frame, attr_name)
            valid = data[frame.mask]
            all_vals.extend(valid[np.isfinite(valid)])
        if vmin is None:
            vmin = np.percentile(all_vals, 2)
        if vmax is None:
            vmax = np.percentile(all_vals, 98)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # helper to get deformed coordinates
    def get_deformed_grid(frame_data):
        # Base grid
        xx, yy = np.meshgrid(frame_data.grid_x, frame_data.grid_y)
        # Add displacement if available
        if hasattr(frame_data, 'displacement_u') and hasattr(frame_data, 'displacement_v'):
            xx = xx + frame_data.displacement_u
            yy = yy + frame_data.displacement_v
        return xx, yy

    # Initial plot
    xx, yy = get_deformed_grid(first_frame)
    data = getattr(first_frame, attr_name)
    masked_data = np.where(first_frame.mask, data, np.nan)
    
    # Calculate global bounds for fixed axes during animation
    # This prevents the camera from "chasing" the sample, making movement obvious
    x_min, x_max = np.nanmin(xx), np.nanmax(xx)
    y_min, y_max = np.nanmin(yy), np.nanmax(yy)
    # Add some padding
    w, h = x_max - x_min, y_max - y_min
    ax.set_xlim(x_min - w*0.1, x_max + w*0.1)
    ax.set_ylim(y_min - h*0.1, y_max + h*0.1)
    
    if vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        im = ax.pcolormesh(xx, yy, masked_data, cmap=cmap, norm=norm, shading="auto")
    else:
        im = ax.pcolormesh(xx, yy, masked_data, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    
    cbar = plt.colorbar(im, ax=ax, label=label)
    ax.set_xlabel("X Position (mm)")
    ax.set_ylabel("Y Position (mm)")
    ax.set_aspect("equal")
    title = ax.set_title(f"Frame 1 - t = 0.00s")
    
    plt.tight_layout()
    
    def update(frame_idx):
        filepath = vc7_files[frame_idx]
        frame_data = read_vc7_file(filepath)
        data = getattr(frame_data, attr_name)
        masked_data = np.where(frame_data.mask, data, np.nan)
        
        # update coordinates
        xx_new, yy_new = get_deformed_grid(frame_data)
        im.set_array(masked_data.ravel())
        
        # Update mesh coordinates (this makes it move)
        # pcolormesh returns a QuadMesh, we need to update 'set_array' (colors) and coordinates
        # Unfortunately QuadMesh coordinates are tricky to update efficiently in older matplotlib versions
        # safely set coordinates if possible, otherwise we might need disjoint update
        try:
             # Standard matplotlib way for QuadMesh
            im.set_offsets(np.c_[xx_new.ravel(), yy_new.ravel()])
            # Note: set_offsets expects (N, 2) array for scatter, but QuadMesh is different.
            # actually pcolormesh is hard to animate geometry changes efficiently without redrawing.
            # Let's try simpler set_array if coordinates don't change, but here they DO.
            
            # Re-drawing is expensive but reliable for changing geometry
            ax.collections.clear()
            if vmin < 0 < vmax:
                new_im = ax.pcolormesh(xx_new, yy_new, masked_data, cmap=cmap, norm=norm, shading="auto")
            else:
                new_im = ax.pcolormesh(xx_new, yy_new, masked_data, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
            return [new_im, title]
            
        except Exception:
             # Fallback if complex update fails
            pass

        time_s = frame_data.acquisition_time_us / 1e6
        title.set_text(f"Frame {frame_data.frame_number} - t = {time_s:.2f}s")
        
        if (frame_idx + 1) % 50 == 0:
            logging.info(f"Rendered {frame_idx + 1}/{len(vc7_files)} frames...")
        
        return [ax.collections[0], title]
    
    anim = FuncAnimation(fig, update, frames=len(vc7_files), blit=True)
    
    # Choose writer based on output format
    if output_path.endswith(".gif"):
        writer = PillowWriter(fps=fps)
    else:
        try:
            writer = FFMpegWriter(fps=fps, bitrate=2000)
        except Exception:
            logging.warning("FFMpeg not available, falling back to GIF")
            output_path = output_path.rsplit(".", 1)[0] + ".gif"
            writer = PillowWriter(fps=fps)
    
    anim.save(output_path, writer=writer)
    logging.info(f"Saved animation to {output_path}")
    plt.close()


def plot_dic_vs_instron(
    merged_df: pd.DataFrame,
    output_path: Optional[str] = None,
    dic_strain_col: str = "mean_eyy",
    instron_strain_col: Optional[str] = None,
    stress_col: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
) -> None:
    """
    Plot comparison between DIC strain and Instron strain/stress.
    
    Args:
        merged_df: Merged DataFrame from merge_dic_with_instron
        output_path: Path to save the figure
        dic_strain_col: Column name for DIC strain
        instron_strain_col: Column name for Instron strain (auto-detect if None)
        stress_col: Column name for stress (auto-detect if None)
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for visualization")
    
    # Auto-detect columns
    if instron_strain_col is None:
        strain_cols = [c for c in merged_df.columns if "strain" in c.lower() and "dic" not in c.lower() and "mean" not in c.lower()]
        if strain_cols:
            instron_strain_col = strain_cols[0]
    
    if stress_col is None:
        stress_cols = [c for c in merged_df.columns if "stress" in c.lower()]
        if stress_cols:
            stress_col = stress_cols[0]
    
    time_col = "time_s"
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Strain comparison over time
    ax1 = axes[0]
    ax1.plot(merged_df[time_col], merged_df[dic_strain_col], 
             label="DIC Strain (Eyy)", linewidth=1, alpha=0.8)
    if instron_strain_col and instron_strain_col in merged_df.columns:
        ax1.plot(merged_df[time_col], merged_df[instron_strain_col], 
                 label="Instron Strain", linewidth=1, alpha=0.8)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Strain")
    ax1.set_title("Strain Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Stress-strain using DIC strain
    ax2 = axes[1]
    if stress_col and stress_col in merged_df.columns:
        ax2.scatter(merged_df[dic_strain_col], merged_df[stress_col], 
                   s=1, alpha=0.5, label="DIC")
        if instron_strain_col and instron_strain_col in merged_df.columns:
            ax2.scatter(merged_df[instron_strain_col], merged_df[stress_col], 
                       s=1, alpha=0.5, label="Instron")
        ax2.set_xlabel("Strain")
        ax2.set_ylabel("Stress (MPa)")
        ax2.set_title("Stress-Strain (DIC vs Instron)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No stress data available", ha="center", va="center",
                transform=ax2.transAxes)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logging.info(f"Saved comparison plot to {output_path}")
        plt.close()
    else:
        plt.show()


def plot_poisson_analysis(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    time_col: str = "time_s",
    normal_strain_col: str = "DIC_normal_strain",
    transverse_strain_col: str = "DIC_transverse_strain",
    poisson_col: str = "DIC_poisson_ratio",
    figsize: Tuple[float, float] = (14, 10),
) -> None:
    """
    Create a comprehensive Poisson ratio analysis plot.
    
    Shows:
    1. Normal (axial) vs Transverse strain over time
    2. Strain-strain scatter plot with linear fit for Poisson ratio
    3. Poisson ratio evolution over time
    4. Summary statistics
    
    Args:
        df: DataFrame with DIC strain data (from extract_dic_timeseries)
        output_path: Path to save the figure
        time_col: Time column name
        normal_strain_col: Column for normal/axial strain (Eyy)
        transverse_strain_col: Column for transverse strain (Exx)
        poisson_col: Column for Poisson ratio
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
        from scipy import stats
    except ImportError:
        raise ImportError("matplotlib and scipy required for visualization")
    
    # Filter out invalid data
    valid_mask = (
        np.isfinite(df[normal_strain_col]) & 
        np.isfinite(df[transverse_strain_col]) &
        (df[normal_strain_col] != 0)
    )
    df_valid = df[valid_mask].copy()
    
    if len(df_valid) == 0:
        logging.warning("No valid data for Poisson analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Normal and Transverse strain over time
    ax1 = axes[0, 0]
    ax1.plot(df_valid[time_col], df_valid[normal_strain_col], 
             label="Normal (Eyy)", linewidth=0.8, alpha=0.8, color="tab:blue")
    ax1.plot(df_valid[time_col], df_valid[transverse_strain_col], 
             label="Transverse (Exx)", linewidth=0.8, alpha=0.8, color="tab:red")
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Strain")
    ax1.set_title("DIC Strain Components vs Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Strain-strain scatter with linear fit
    ax2 = axes[0, 1]
    ax2.scatter(df_valid[normal_strain_col], df_valid[transverse_strain_col], 
               s=3, alpha=0.5, c=df_valid[time_col], cmap="viridis")
    
    # Linear regression for overall Poisson ratio
    normal_vals = df_valid[normal_strain_col].values
    transverse_vals = df_valid[transverse_strain_col].values
    
    # Only use data where both strains are reasonable (avoid near-zero division)
    fit_mask = np.abs(normal_vals) > 0.001  # At least 0.1% strain
    if fit_mask.sum() > 10:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            normal_vals[fit_mask], transverse_vals[fit_mask]
        )
        overall_poisson = -slope
        
        # Plot fit line
        x_fit = np.linspace(normal_vals.min(), normal_vals.max(), 100)
        y_fit = slope * x_fit + intercept
        ax2.plot(x_fit, y_fit, 'r-', linewidth=2, 
                label=f"ν = {overall_poisson:.3f} (R² = {r_value**2:.3f})")
    else:
        overall_poisson = np.nanmedian(df_valid[poisson_col])
    
    ax2.set_xlabel("Normal Strain (Eyy)")
    ax2.set_ylabel("Transverse Strain (Exx)")
    ax2.set_title("Strain-Strain Relationship")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label("Time (s)")
    
    # Plot 3: Poisson ratio evolution
    ax3 = axes[1, 0]
    if poisson_col in df_valid.columns:
        # Filter extreme values for visualization
        poisson_vals = df_valid[poisson_col].values
        poisson_clipped = np.clip(poisson_vals, 0, 1)  # Physical range for elastomers
        
        ax3.plot(df_valid[time_col], poisson_vals, 
                linewidth=0.5, alpha=0.5, color="gray", label="Raw")
        
        # Rolling median for smoothed view
        window = max(10, len(df_valid) // 100)
        rolling_poisson = pd.Series(poisson_vals).rolling(window, center=True).median()
        ax3.plot(df_valid[time_col], rolling_poisson, 
                linewidth=2, color="tab:green", label=f"Smoothed (n={window})")
        
        ax3.axhline(y=0.5, color='k', linestyle='--', linewidth=1, alpha=0.5,
                   label="Incompressible (ν=0.5)")
        ax3.axhline(y=overall_poisson, color='r', linestyle='-', linewidth=1.5, alpha=0.8,
                   label=f"Overall ν = {overall_poisson:.3f}")
    
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Poisson's Ratio (ν)")
    ax3.set_title("Poisson's Ratio Evolution")
    ax3.set_ylim(0, 0.7)  # Reasonable range for elastomers
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics text box
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
    ╔══════════════════════════════════════╗
    ║    POISSON RATIO ANALYSIS SUMMARY    ║
    ╠══════════════════════════════════════╣
    ║                                      ║
    ║  Overall Poisson Ratio: {overall_poisson:>8.4f}     ║
    ║                                      ║
    ║  Normal Strain (Eyy):                ║
    ║    Range: {df_valid[normal_strain_col].min():>8.4f} to {df_valid[normal_strain_col].max():<8.4f} ║
    ║    Mean:  {df_valid[normal_strain_col].mean():>8.4f}               ║
    ║                                      ║
    ║  Transverse Strain (Exx):            ║
    ║    Range: {df_valid[transverse_strain_col].min():>8.4f} to {df_valid[transverse_strain_col].max():<8.4f} ║
    ║    Mean:  {df_valid[transverse_strain_col].mean():>8.4f}               ║
    ║                                      ║
    ║  Data Points: {len(df_valid):>6}                  ║
    ║  Time Range: {df_valid[time_col].min():>6.1f}s to {df_valid[time_col].max():<6.1f}s   ║
    ╚══════════════════════════════════════╝
    """
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, 
            fontsize=11, fontfamily='monospace', verticalalignment='center')
    
    plt.suptitle("DIC Poisson Ratio Analysis", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logging.info(f"Saved Poisson analysis plot to {output_path}")
        plt.close()
    else:
        plt.show()


# ============================================================================
# Example usage / command-line interface
# ============================================================================


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    parser = argparse.ArgumentParser(description="Extract DIC strain data from VC7 files")
    parser.add_argument("strain_dir", help="Directory containing VC7 files")
    parser.add_argument("--output", "-o", help="Output CSV file path")
    parser.add_argument("--cube", help="Output NPZ file for full 3D strain cube")
    parser.add_argument("--component", default="exx", 
                       choices=["exx", "eyy", "exy", "u", "v"],
                       help="Component to extract for cube output")
    parser.add_argument("--start", type=int, help="Start frame number")
    parser.add_argument("--end", type=int, help="End frame number")
    
    args = parser.parse_args()
    
    frame_range = None
    if args.start and args.end:
        frame_range = (args.start, args.end)
    
    # Default output path
    if not args.output:
        args.output = os.path.join(args.strain_dir, "dic_strain_timeseries.csv")
    
    # Extract time-series
    df = extract_dic_timeseries(
        args.strain_dir, 
        output_csv=args.output,
        frame_range=frame_range
    )
    
    print(f"\n=== Summary ===")
    print(f"Frames processed: {len(df)}")
    print(f"Mean Exx: {df['mean_exx'].mean():.6f} ± {df['mean_exx'].std():.6f}")
    print(f"Mean Eyy: {df['mean_eyy'].mean():.6f} ± {df['mean_eyy'].std():.6f}")
    print(f"Output saved to: {args.output}")
    
    # Optionally extract full cube
    if args.cube:
        result = extract_full_strain_cube(
            args.strain_dir,
            component=args.component,
            frame_range=frame_range,
            output_npz=args.cube
        )
        print(f"Strain cube shape: {result['data'].shape}")
        print(f"Cube saved to: {args.cube}")

