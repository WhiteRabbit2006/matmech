"""
The main workflow orchestration module for the mat-analyzer library.

This module defines the `run_analysis_workflow` function, which serves as the
primary entry point for processing mechanical test data. It handles:
1. Configuration setup and merging default profiles with user settings.
2. Path and directory management for input data and output graphs.
3. Data loading, standardization, and optional taring/inversion.
4. Segmentation of data into test phases based on a recipe.
5. Phase-by-phase analysis using a registry of analysis functions.
6. Generation of static and animated plots based on user or default configurations.
"""

import copy
import logging
import os
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

# noinspection PyPackages
from matmech import axial_analysis, common_utils, config_defaults, plotting_tools, torsional_analysis
from matmech.constants import TIME_COL

# The Analysis Registry: Maps a string from the config to an analysis function.
# This makes the workflow extensible without modification.
ANALYSIS_REGISTRY: Dict[str, Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame]] = {
    "AXIAL": axial_analysis.calculate_axial_properties,
    "TORSIONAL": torsional_analysis.calculate_torsional_properties_rect,
}


def _resolve_column_info(
    df: pd.DataFrame, user_key: str, user_units: str = "auto"
) -> Tuple[str, str]:
    """
    Resolves user-friendly keys (e.g., 'force') to specific DataFrame columns and plot labels.
    Handles unit conversions and auto-scaling for plotting.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        user_key (str): A user-friendly key (e.g., 'time', 'force', 'axial_stress').
        user_units (str): The desired units for plotting (e.g., 'MPa', 'kN', 'm').
                          If 'auto', units will be chosen based on data magnitude.

    Returns:
        Tuple[str, str]: A tuple containing:
                         - The actual DataFrame column name to plot.
                         - The formatted axis label including units.

    Raises:
        ValueError: If a user key is not provided or unit conversion is not defined.
        KeyError: If the user key is not in the registry or the required base column is missing.
    """
    if not user_key:
        raise ValueError("A user key (e.g., 'force', 'time') must be provided.")

    registry_key = user_key.lower()
    if registry_key not in config_defaults.DATA_COLUMN_REGISTRY:
        raise KeyError(f"Data key '{user_key}' not defined in the Data Column Registry.")

    col_info = config_defaults.DATA_COLUMN_REGISTRY[registry_key]
    standard_name = col_info["standard_name"]

    if standard_name not in df.columns:
        raise KeyError(
            f"Required base column '{standard_name}' for '{user_key}' "
            f"does not exist in the DataFrame. Available columns: {df.columns.tolist()}"
        )

    chosen_units = user_units
    if user_units == "auto":
        chosen_units = col_info["default_units"]
        if "auto_scale_options" in col_info and not df[standard_name].empty:
            # Determine best units for display based on max absolute value
            max_val = df[standard_name].abs().max() + 1e-12  # Add small epsilon to avoid log(0) issues
            for threshold, unit_str in col_info["auto_scale_options"]:
                if max_val >= threshold * 0.1:  # If max_val is at least 10% of threshold
                    chosen_units = unit_str
                    break

    if chosen_units and chosen_units != col_info["default_units"]:
        if chosen_units in col_info["conversions"]:
            converted_col_name = f"{standard_name}_to_{chosen_units}"
            conversion_func = col_info["conversions"][chosen_units]

            if converted_col_name not in df.columns:
                df[converted_col_name] = conversion_func(df[standard_name])

            column_to_plot = converted_col_name
            # Replace default units in label with chosen units
            axis_label = col_info["label"].replace(
                f"({col_info['default_units']})", f"({chosen_units})"
            )
        else:
            raise ValueError(f"Unit conversion '{chosen_units}' not defined for '{user_key}'.")
    else:
        column_to_plot = standard_name
        axis_label = col_info["label"]

    return column_to_plot, axis_label


def run_analysis_workflow(script_path: str, user_config: Dict[str, Any]) -> None:
    """
    The main entry point for running a complete data analysis workflow.

    This function orchestrates the entire process from configuration loading
    to data processing, analysis, and plot generation.

    Args:
        script_path (str): The absolute path to the directory where the calling
                           script (e.g., main.py) is located. This is used to
                           locate data and output directories.
        user_config (Dict[str, Any]): A dictionary containing user-defined
                                     configuration settings for the analysis.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Starting data analysis workflow...")

    # === 1. CONFIGURATION SETUP ===
    software_type = user_config.get("software_type", config_defaults.DEFAULT_SOFTWARE_TYPE)
    logging.info(f"Using software profile: '{software_type}'")

    if software_type not in config_defaults.SOFTWARE_PROFILES:
        raise ValueError(f"Software type '{software_type}' not defined in SOFTWARE_PROFILES.")

    base_profile = copy.deepcopy(config_defaults.SOFTWARE_PROFILES[software_type])
    final_config = {**base_profile, **user_config}

    # Deep merge dictionaries for nested configurations (e.g., 'geometry', 'plots')
    # User-provided empty dictionaries for nested configs should explicitly override defaults.
    for key, value in user_config.items():
        if isinstance(value, dict):
            if key in base_profile and isinstance(base_profile[key], dict) and value: # Only merge if user_config dict is not empty
                final_config[key] = {**base_profile[key], **value}
            else: # If user_config dict is empty, or key not in base_profile, or base_profile[key] is not a dict, replace entirely
                final_config[key] = value
        elif isinstance(value, list):
            # For lists, user_config's list completely replaces base_profile's list.
            final_config[key] = value
        # For other types, the shallow merge {**base_profile, **user_config} is sufficient.

    # === 2. PATH AND DIRECTORY SETUP ===
    output_dir = os.path.join(script_path, "graphs")
    input_file_path = os.path.join(script_path, "data", final_config["data_file_name"])
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory set to: '{output_dir}'")

    # === 3. DATA LOADING AND STANDARDIZATION ===
    full_raw_df = common_utils.load_csv_data(input_file_path)
    sources = final_config.get("column_sources", {})

    # Convert all relevant source columns to numeric, coercing errors, and clean the DataFrame.
    # This handles cases where CSVs from testing software include non-numeric header/footer rows.
    source_cols = []
    for s in sources.values():
        potential_sources = s if isinstance(s, list) else [s]
        for potential_source in potential_sources:
            raw_col = potential_source.get("raw_col")
            if raw_col and raw_col in full_raw_df.columns:
                source_cols.append(raw_col)
                if isinstance(s, list):
                    break
    for col in source_cols:
        full_raw_df[col] = pd.to_numeric(full_raw_df[col], errors="coerce")

    # Drop rows where any of the essential source columns are NaN. This removes metadata rows.
    if source_cols:
        full_raw_df = full_raw_df.dropna(subset=source_cols).reset_index(drop=True)

    clean_df = pd.DataFrame()
    inversion_flags = final_config.get("inversion_flags", {})
    tare_options = final_config.get("tare_options", {})

    for key, source_info in sources.items():
        # Handle list of fallback column definitions
        if isinstance(source_info, list):
            found_source = None
            for s in source_info:
                if isinstance(s, dict) and s.get("raw_col") in full_raw_df.columns:
                    found_source = s
                    break
            if not found_source:
                raw_cols = [s.get("raw_col", "N/A") for s in source_info]
                logging.warning(
                    f"None of the source columns {raw_cols} for '{key}' found in data file. Skipping '{key}'."
                )
                continue
            source_info = found_source

        registry_entry = config_defaults.DATA_COLUMN_REGISTRY[key]
        standard_name = registry_entry["standard_name"]
        raw_col, raw_units = source_info["raw_col"], source_info["raw_units"]

        if raw_col not in full_raw_df.columns:
            logging.warning(
                f"Source column '{raw_col}' for '{key}' not in data file. Skipping '{key}'."
            )
            continue

        series = full_raw_df[raw_col].copy()

        # Standardize units
        if raw_units != registry_entry["default_units"]:
            if raw_units not in registry_entry["standardize_from"]:
                raise ValueError(
                    f"Standardization from '{raw_units}' to "
                    f"'{registry_entry['default_units']}' not defined for '{key}'."
                )
            convert_func = registry_entry["standardize_from"][raw_units]
            series = convert_func(series)
            logging.debug(f"Standardized '{key}' from '{raw_units}' to '{registry_entry['default_units']}'.")

        # Apply inversion if flagged
        if inversion_flags.get(key, False):
            series *= -1
            logging.debug(f"Applied inversion to '{key}' channel.")

        # Apply taring if flagged
        if tare_options.get(key, False):
            if not series.empty:
                series -= series.iloc[0]
                logging.debug(f"Applied taring to '{key}' channel (normalized to start at zero).")
            else:
                logging.warning(f"Attempted to tare '{key}', but the series was empty.")

        clean_df[standard_name] = series

    logging.info("Data standardization complete.")

    # === 4. DATA SEGMENTATION ===
    recipe: List[Dict[str, Any]] = final_config["test_recipe"]

    # Get the standard name for the time column from the constants
    time_standard_name = TIME_COL

    # Defensive check: Ensure the time column exists in clean_df
    if time_standard_name not in clean_df.columns:
        raise KeyError(
            f"Required time column '{time_standard_name}' not found in processed data. "
            f"Available columns: {clean_df.columns.tolist()}"
        )

    # Auto-set end_time for the final phase if not specified
    if recipe and ("end_time" not in recipe[-1] or recipe[-1].get("end_time") is None):
        if not clean_df.empty:
            last_time = clean_df[time_standard_name].max()
            recipe[-1]["end_time"] = last_time
            logging.info(
                f"Final phase '{recipe[-1]['name']}' had no end_time. "
                f"Using end of data ({last_time:.2f} s) as phase end."
            )
        else:
            logging.warning("Dataframe is empty, cannot determine end time for final phase.")
            recipe[-1]["end_time"] = 0

    # Now, validate all phases have an end time, which is required for splitting.
    for phase in recipe:
        if "end_time" not in phase or phase.get("end_time") is None:
            raise ValueError(
                f"Phase '{phase['name']}' is missing 'end_time', "
                "which is required for all but the final phase."
            )

    split_points = [phase["end_time"] for phase in recipe]

    data_segments = common_utils.split_data_by_time(
        clean_df, split_points, time_col=time_standard_name
    )

    # === 5. PHASE-BY-PHASE ANALYSIS (USING REGISTRY) ===
    processed_data_store: Dict[str, pd.DataFrame] = {}
    for i, (phase, segment_df) in enumerate(zip(recipe, data_segments)):
        phase_name, analysis_type = phase["name"], phase["type"]
        logging.info(f"\n--- Analyzing Phase {i+1}: {phase_name} (Type: {analysis_type}) ---")

        if segment_df.empty:
            logging.warning(f"Segment for phase '{phase_name}' is empty. Skipping analysis.")
            processed_data_store[phase_name] = pd.DataFrame()
            continue

        if analysis_type in ANALYSIS_REGISTRY:
            analysis_func = ANALYSIS_REGISTRY[analysis_type]
            processed_df = analysis_func(segment_df, final_config["geometry"])
        else:
            logging.info(
                f"No analysis function registered for type '{analysis_type}'. "
                "Passing data through without further processing."
            )
            processed_df = segment_df
        processed_data_store[phase_name] = processed_df

    # === 6. PLOT GENERATION ===
    plot_configs_raw = final_config.get("plots", [])
    resolved_plot_configs: List[Dict[str, Any]] = []

    for plot_def in plot_configs_raw:
        if isinstance(plot_def, str):
            # If it's a string, it's a key for a default plot
            if plot_def in config_defaults.DEFAULT_PLOTS:
                resolved_plot_configs.append(copy.deepcopy(config_defaults.DEFAULT_PLOTS[plot_def]))
            else:
                logging.warning(
                    f"Plot key '{plot_def}' not found in DEFAULT_PLOTS registry. Skipping."
                )
        elif isinstance(plot_def, dict):
            # If it's a dictionary, it's a custom plot configuration
            resolved_plot_configs.append(plot_def)
        else:
            logging.warning(
                f"Invalid plot definition type: {type(plot_def)}. "
                "Expected string or dict. Skipping."
            )

    logging.info(f"\n--- Generating {len(resolved_plot_configs)} requested plot definition(s) ---")

    # Get all phase names for '*' handling in plot configurations
    all_phase_names = [phase["name"] for phase in recipe]

    for plot_config in resolved_plot_configs:
        # Ensure 'output_filename' is present for all plots, including custom ones
        if "output_filename" not in plot_config:
            logging.warning(
                f"Plot configuration missing 'output_filename'. Skipping plot: "
                f"{plot_config.get('title', 'Untitled Plot')}"
            )
            continue

        target_phases = plot_config.get("phases", [])

        # Handle '*' for all phases
        phases_to_iterate = all_phase_names if "*" in target_phases else target_phases

        for phase_name in phases_to_iterate:
            df_to_plot = processed_data_store.get(phase_name)
            if df_to_plot is None or df_to_plot.empty:
                logging.warning(
                    f"No data available for phase '{phase_name}' to generate plot "
                    f"'{plot_config.get('title', 'Untitled')}'."
                )
                continue
            try:
                # Resolve column names and labels for plotting, handling units
                x_col_to_plot, x_label = _resolve_column_info(
                    df_to_plot, plot_config["x_col"], plot_config.get("x_units", "auto")
                )
                y_col_to_plot, y_label = _resolve_column_info(
                    df_to_plot, plot_config["y_col"], plot_config.get("y_units", "auto")
                )

                # Determine plot types (static, animated, or both)
                plot_types_config = plot_config.get("type", "static")
                plot_types = (
                    [plot_types_config]
                    if isinstance(plot_types_config, str)
                    else plot_types_config
                )

                for plot_type in plot_types:
                    plot_type = plot_type.lower()
                    format_keys = {**plot_config, "phase_name": phase_name}
                    base_filename = plot_config["output_filename"].format(**format_keys)
                    suffix = ".mp4" if plot_type == "animated" else ".png"
                    filename = base_filename + suffix
                    title = plot_config["title"].format(**format_keys)
                    output_file_path = os.path.join(output_dir, filename)

                    if plot_type == "animated":
                        plotting_tools.animate_curve(
                            df=df_to_plot,
                            x_col=x_col_to_plot,
                            y_col=y_col_to_plot,
                            title=title,
                            x_label=x_label,
                            y_label=y_label,
                            output_path=output_file_path,
                            **plot_config.get("animation_options", {}),
                        )
                    elif plot_type == "static":
                        # Pass base column names and units for fit calculation
                        plotting_tools.plot_curve(
                            df=df_to_plot,
                            x_col_plot=x_col_to_plot,
                            y_col_plot=y_col_to_plot,
                            x_col_base=config_defaults.DATA_COLUMN_REGISTRY[
                                plot_config["x_col"].lower()
                            ]["standard_name"],
                            y_col_base=config_defaults.DATA_COLUMN_REGISTRY[
                                plot_config["y_col"].lower()
                            ]["standard_name"],
                            y_base_units=config_defaults.DATA_COLUMN_REGISTRY[
                                plot_config["y_col"].lower()
                            ]["default_units"],
                            title=title,
                            x_label=x_label,
                            y_label=y_label,
                            output_path=output_file_path,
                            fit_line=plot_config.get("fit_line", False),
                            fit_bounds=plot_config.get("fit_bounds"),
                        )
                    else:
                        logging.warning(f"Unknown plot type '{plot_type}'. Skipping.")
            except (KeyError, ValueError) as e:
                logging.warning(
                    f"Skipping plot '{plot_config.get('title', 'Untitled')}' "
                    f"for phase '{phase_name}'. Reason: {e}"
                )
    logging.info(f"\nMulti-phase analysis complete. Graphs saved in '{output_dir}'.")
