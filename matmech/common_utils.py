"""
This module provides common utility functions used across the analysis library,
such as loading data from CSV files and splitting DataFrames by time points.
"""

import logging
import os
from typing import Any, List

import pandas as pd


def load_csv_data(file_path: str, **kwargs: Any) -> pd.DataFrame:
    """
    Loads a generic CSV file into a pandas DataFrame.

    Args:
        file_path (str): The full path to the CSV file.
        **kwargs (Any): Additional keyword arguments to pass to pandas.read_csv.

    Returns:
        pd.DataFrame: The loaded data.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    logging.info(f"Loading data from: {os.path.basename(file_path)}")
    return pd.read_csv(file_path, **kwargs)


def split_data_by_time(
    df: pd.DataFrame, split_points: List[float], time_col: str
) -> List[pd.DataFrame]:
    """
    Splits a DataFrame into multiple segments based on a list of time points.

    Each segment includes data from the end of the previous segment (exclusive)
    up to the current split point (inclusive).

    Args:
        df (pd.DataFrame): The input DataFrame to split.
        split_points (List[float]): A list of time points (in seconds) at which
                                    to split the data.
        time_col (str): The name of the time column in the DataFrame.

    Returns:
        List[pd.DataFrame]: A list of DataFrames, each representing a segment
                            of the original data.
    """
    segments: List[pd.DataFrame] = []
    last_time = 0.0

    for end_time in split_points:
        # Ensure the mask correctly handles the start of the first segment
        # and subsequent segments without overlapping or missing data.
        mask = (df[time_col] > last_time) & (df[time_col] <= end_time)
        segment_df = df.loc[mask].copy()
        segments.append(segment_df)
        logging.info(
            f"Created segment from t={last_time:.2f}s to t={end_time:.2f}s "
            f"with {len(segment_df)} data points."
        )
        last_time = end_time

    return segments


def remove_turnaround_artifacts(
    df: pd.DataFrame,
    position_col: str,
    force_col: str,
    time_col: str,
    target_window_duration: float = 0.5,
    std_dev_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Removes artifacts (spikes) in the force data that occur during sharp direction changes
    (turnarounds) in the displacement/position using a dynamic smoothing filter.

    The algorithm:
    1. Calculate dynamic window size based on sampling rate and `target_window_duration`.
    2. Generate a robust reference curve using a rolling median filter on the force data.
    3. Detect outliers where the force deviates significantly from this reference near turnarounds.
    4. Remove these outlier rows.

    Args:
        df (pd.DataFrame): Input DataFrame.
        position_col (str): Name of the column representing position/displacement.
        force_col (str): Name of the column representing force.
        time_col (str): Name of the column representing time (for rate calculation).
        target_window_duration (float): Target duration (in seconds) for the smoothing window.
        std_dev_threshold (float): Threshold for outlier detection (sigmas of the residual).

    Returns:
        pd.DataFrame: A filtered DataFrame with artifacts removed.
    """
    if df.empty or position_col not in df.columns or force_col not in df.columns or time_col not in df.columns:
        return df

    df_clean = df.copy()

    # 1. Calculate Dynamic Window Size
    # Estimate sampling rate (dt)
    dt_series = df_clean[time_col].diff().dropna()
    if dt_series.empty:
        logging.warning("Cannot calculate sampling rate. Skipping artifact removal.")
        return df_clean
    
    avg_dt = dt_series.mean()
    if avg_dt <= 0:
        logging.warning("Invalid time steps detected. Skipping artifact removal.")
        return df_clean
        
    # Window size in points = duration / dt
    window_points = int(target_window_duration / avg_dt)
    # Ensure window is odd and at least 3
    if window_points % 2 == 0:
        window_points += 1
    window_points = max(3, window_points)
    
    logging.info(f"Dynamic artifact removal: dt={avg_dt:.4f}s, window={window_points} points.")

    # 2. Detect Turnarounds (Local Extrema in Position)
    # We look for sign changes in the gradient of position to find reversals
    pos_grad = df_clean[position_col].diff()
    # Fill NA to keep alignment
    pos_grad = pos_grad.fillna(0)
    
    # Identify reversals: where the product of consecutive gradients is negative
    # (indicating a change in direction)
    sign_changes = (pos_grad * pos_grad.shift(-1)) < 0
    turnaround_indices = df_clean.index[sign_changes].tolist()

    if not turnaround_indices:
        logging.info("No turnarounds detected. Skipping artifact removal.")
        return df_clean

    # 3. Robust Smoothing (Reference Curve)
    # Use rolling median to ignore spikes
    force_series = df_clean[force_col]
    smoothed_force = force_series.rolling(window=window_points, center=True, min_periods=1).median()
    
    # Calculate Residuals (Noise)
    residuals = (force_series - smoothed_force).abs()
    
    # Determine global noise level (median absolute deviation implies sigma)
    # MAD = median(|x - median|) which is our 'residuals' if smoothed is median
    # Sigma approx = 1.4826 * MAD
    mad = residuals.median()
    sigma_est = 1.4826 * mad
    
    # Safety check: if signal is very clean or quantized, MAD might be 0.
    # We fallback to the mean absolute deviation or a small epsilon relative to signal range.
    if sigma_est < 1e-9:
        mean_res = residuals.mean()
        if mean_res > 1e-9:
            sigma_est = 1.2533 * mean_res # approx for normal dist from mean dev
        else:
            # Signal is practically flat/perfect. Use a tiny epsilon to avoid removing everything.
            sigma_est = 1e-9
    
    cutoff = std_dev_threshold * sigma_est
    
    # 4. Filter Turnaround Regions
    indices_to_remove = set()
    
    # We define a "scan region" around each turnaround
    # Let's say +/- half the window size
    scan_radius = window_points // 2
    
    for idx_center in turnaround_indices:
        # Define indices in the DataFrame index space
        # Assuming discrete integer index for simplicity (standard for loaded CSVs)
        # Proper way using iloc logic requires mapping, but let's assume index is reliable or use range check
        
        # We'll rely on the row *position* ideally, but DataFrame index usage is standard if unique.
        
        # Determine the range of INDICES to check
        # We strictly check standard integer range if index is Int64Index
        scan_start = idx_center - scan_radius
        scan_end = idx_center + scan_radius
        
        # Filter points in this region that exceed the cutoff
        bad_points = residuals.loc[scan_start:scan_end]
        outliers = bad_points[bad_points > cutoff].index.tolist()
        
        indices_to_remove.update(outliers)

    if indices_to_remove:
        logging.info(
            f"Removing {len(indices_to_remove)} artifact points. "
            f"Cutoff used: {cutoff:.6f} (Threshold: {std_dev_threshold} sigma, Est. Sigma: {sigma_est:.6f})"
        )
        # Only remove if they are valid indices (defensive)
        valid_indices = [i for i in indices_to_remove if i in df_clean.index]
        df_clean = df_clean.drop(valid_indices)
    
    return df_clean
