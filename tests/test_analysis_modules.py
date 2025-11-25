# tests/test_analysis_modules.py
"""
Unit tests for the analysis modules (axial_analysis, torsional_analysis).

These tests verify the correctness of property calculations based on
known inputs and expected outputs.
"""

import logging
import re  # Import the 're' module for regex escaping
import numpy as np
import pandas as pd
import pytest
import os
import shutil

from matmech import axial_analysis, torsional_analysis, common_utils, plotting_tools, workflow, config_defaults
from matmech.constants import (
    AXIAL_STRAIN_COL,
    AXIAL_STRESS_MPA_COL,
    FORCE_COL,
    POSITION_COL,
    ROTATION_COL,
    SHEAR_STRAIN_COL,
    SHEAR_STRESS_MPA_COL,
    TORQUE_COL,
    TIME_COL,
)


def test_calculate_axial_properties():
    """Verify that axial stress and strain are calculated correctly."""
    test_data = {
        POSITION_COL: [0, 0.5, 1.0],
        FORCE_COL: [0, 500, 1000],
    }
    df = pd.DataFrame(test_data)

    geometry = {
        "gauge_length_mm": 100.0,
        "axial_width_mm": 20.0,
        "axial_thickness_mm": 5.0,
    }

    # Expected area = 20mm * 5mm = 100 mm^2 = 0.0001 m^2
    # Expected stress @ 1000N = 1000N / 0.0001m^2 = 10,000,000 Pa = 10 MPa
    # Expected strain @ 1.0mm = 1.0mm / 100mm = 0.01

    result_df = axial_analysis.calculate_axial_properties(df, geometry)

    assert AXIAL_STRAIN_COL in result_df.columns
    assert AXIAL_STRESS_MPA_COL in result_df.columns

    assert np.isclose(result_df[AXIAL_STRAIN_COL].iloc[-1], 0.01)
    assert np.isclose(result_df[AXIAL_STRESS_MPA_COL].iloc[-1], 10.0)


def test_calculate_axial_properties_zero_area(caplog):
    """Test axial stress calculation with zero cross-sectional area."""
    test_data = {
        POSITION_COL: [0, 0.5, 1.0],
        FORCE_COL: [0, 500, 1000],
    }
    df = pd.DataFrame(test_data)
    geometry = {
        "gauge_length_mm": 100.0,
        "axial_width_mm": 0.0,  # Zero width
        "axial_thickness_mm": 5.0,
    }
    with caplog.at_level(logging.ERROR):
        result_df = axial_analysis.calculate_axial_properties(df, geometry)
        assert "Cross-sectional area is zero" in caplog.text
    assert AXIAL_STRESS_MPA_COL not in result_df.columns  # Should not add stress columns


def test_calculate_axial_properties_missing_geometry_keys(caplog):
    """Test axial property calculation with missing geometry keys."""
    test_data = {
        POSITION_COL: [0, 0.5, 1.0],
        FORCE_COL: [0, 500, 1000],
    }
    df = pd.DataFrame(test_data)
    geometry_missing_width = {
        "gauge_length_mm": 100.0,
        "axial_thickness_mm": 5.0,
    }
    with caplog.at_level(logging.WARNING):
        result_df = axial_analysis.calculate_axial_properties(df, geometry_missing_width)
        assert "Could not calculate axial stress" in caplog.text
    assert AXIAL_STRESS_MPA_COL not in result_df.columns

    geometry_missing_gauge_length = {
        "axial_width_mm": 20.0,
        "axial_thickness_mm": 5.0,
    }
    with caplog.at_level(logging.WARNING):
        result_df = axial_analysis.calculate_axial_properties(df, geometry_missing_gauge_length)
        assert "Could not calculate axial strain from displacement" in caplog.text
    assert AXIAL_STRAIN_COL not in result_df.columns


def test_calculate_axial_properties_pre_existing_strain(caplog):
    """Verify that pre-existing axial strain is not overwritten."""
    test_data = {
        POSITION_COL: [0, 0.5, 1.0],
        FORCE_COL: [0, 500, 1000],
        AXIAL_STRAIN_COL: [0.0, 0.001, 0.002],  # Pre-existing strain
    }
    df = pd.DataFrame(test_data)
    geometry = {
        "gauge_length_mm": 100.0,
        "axial_width_mm": 20.0,
        "axial_thickness_mm": 5.0,
    }
    with caplog.at_level(logging.INFO):
        result_df = axial_analysis.calculate_axial_properties(df, geometry)
        assert "Using pre-calculated axial strain found in data." in caplog.text
    assert np.isclose(result_df[AXIAL_STRAIN_COL].iloc[-1], 0.002)  # Should be original value


def test_calculate_axial_properties_zero_gauge_length(caplog):
    """Test axial strain calculation with zero gauge length."""
    test_data = {
        POSITION_COL: [0, 0.5, 1.0],
        FORCE_COL: [0, 500, 1000],
    }
    df = pd.DataFrame(test_data)
    geometry = {
        "gauge_length_mm": 0.0,  # Zero gauge length
        "axial_width_mm": 20.0,
        "axial_thickness_mm": 5.0,
    }
    with caplog.at_level(logging.ERROR):
        result_df = axial_analysis.calculate_axial_properties(df, geometry)
        assert "Gauge length is zero" in caplog.text
    assert AXIAL_STRAIN_COL not in result_df.columns


def test_calculate_axial_properties_empty_df():
    """Test axial property calculation with an empty DataFrame."""
    df = pd.DataFrame({POSITION_COL: [], FORCE_COL: []})
    geometry = {
        "gauge_length_mm": 100.0,
        "axial_width_mm": 20.0,
        "axial_thickness_mm": 5.0,
    }
    result_df = axial_analysis.calculate_axial_properties(df, geometry)
    assert result_df.empty
    assert AXIAL_STRESS_MPA_COL in result_df.columns # Columns are added, but will be empty
    assert AXIAL_STRAIN_COL in result_df.columns


def test_calculate_axial_properties_negative_values():
    """Test axial property calculation with negative force and displacement."""
    test_data = {
        POSITION_COL: [0, -0.5, -1.0],
        FORCE_COL: [0, -500, -1000],
    }
    df = pd.DataFrame(test_data)
    geometry = {
        "gauge_length_mm": 100.0,
        "axial_width_mm": 20.0,
        "axial_thickness_mm": 5.0,
    }
    result_df = axial_analysis.calculate_axial_properties(df, geometry)
    assert np.isclose(result_df[AXIAL_STRAIN_COL].iloc[-1], -0.01)
    assert np.isclose(result_df[AXIAL_STRESS_MPA_COL].iloc[-1], -10.0)


def test_load_csv_data(tmp_path):
    """Test successful loading of CSV data and FileNotFoundError."""
    # Test successful loading
    csv_content = "col1,col2\n1,A\n2,B"
    file_path = tmp_path / "test.csv"
    file_path.write_text(csv_content)
    df = common_utils.load_csv_data(str(file_path))
    pd.testing.assert_frame_equal(df, pd.DataFrame({"col1": [1, 2], "col2": ["A", "B"]}))

    # Test FileNotFoundError
    with pytest.raises(FileNotFoundError):
        common_utils.load_csv_data(str(tmp_path / "non_existent.csv"))


def test_split_data_by_time():
    """Test splitting a DataFrame into segments based on time points."""
    df = pd.DataFrame({
        TIME_COL: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "value": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    })
    split_points = [3.5, 7.5]
    segments = common_utils.split_data_by_time(df, split_points, TIME_COL)

    assert len(segments) == 2
    # Segment 1: (0, 3.5] -> indices 1, 2, 3
    pd.testing.assert_frame_equal(segments[0], df.iloc[1:4].copy())
    # Segment 2: (3.5, 7.5] -> indices 4, 5, 6, 7
    pd.testing.assert_frame_equal(segments[1], df.iloc[4:8].copy())

    # Test with no split points
    segments_no_split = common_utils.split_data_by_time(df, [], TIME_COL)
    assert len(segments_no_split) == 0

    # Test with empty DataFrame
    empty_df = pd.DataFrame({TIME_COL: [], "value": []})
    segments_empty_df = common_utils.split_data_by_time(empty_df, [1.0], TIME_COL)
    assert len(segments_empty_df) == 1
    assert segments_empty_df[0].empty

    # Test with split points outside data range
    segments_outside = common_utils.split_data_by_time(df, [100.0], TIME_COL)
    assert len(segments_outside) == 1
    pd.testing.assert_frame_equal(segments_outside[0], df.iloc[1:].copy())


def test_calculate_linear_fit():
    """Test linear fit calculation with various data, bounds, and unit auto-scaling."""
    df = pd.DataFrame({
        "x_data": [1, 2, 3, 4, 5],
        "y_data": [2, 4, 6, 8, 10],
        "y_data_offset": [3, 5, 7, 9, 11],
        "y_data_neg_slope": [10, 8, 6, 4, 2],
    })

    # Perfect linear fit
    fit_results = plotting_tools.calculate_linear_fit(df, "x_data", "y_data", "MPa")
    assert np.isclose(fit_results["modulus_val"], 2.0)
    assert fit_results["modulus_units"] == "MPa"
    assert np.isclose(fit_results["y_intercept"], 0.0)
    assert np.isclose(fit_results["x_intercept"], 0.0)

    # Linear fit with offset
    fit_results_offset = plotting_tools.calculate_linear_fit(df, "x_data", "y_data_offset", "MPa")
    assert np.isclose(fit_results_offset["modulus_val"], 2.0)
    assert np.isclose(fit_results_offset["y_intercept"], 1.0)
    assert np.isclose(fit_results_offset["x_intercept"], -0.5)

    # Linear fit with negative slope
    fit_results_neg = plotting_tools.calculate_linear_fit(df, "x_data", "y_data_neg_slope", "MPa")
    assert np.isclose(fit_results_neg["modulus_val"], -2.0)
    assert np.isclose(fit_results_neg["y_intercept"], 12.0)
    assert np.isclose(fit_results_neg["x_intercept"], 6.0)

    # Test with fit_bounds
    fit_results_bounds = plotting_tools.calculate_linear_fit(df, "x_data", "y_data", "MPa", fit_bounds=(2.5, 4.5))
    # Data points for fit: (3,6), (4,8) -> Slope = 2, Y-intercept = 0
    assert np.isclose(fit_results_bounds["modulus_val"], 2.0)
    assert np.isclose(fit_results_bounds["y_intercept"], 0.0)

    # Test auto-scaling of modulus units (MPa to GPa)
    df_gpa = pd.DataFrame({
        "x_data": [1, 2, 3],
        "y_data": [1000, 2000, 3000],  # 1000 MPa/unit = 1 GPa/unit
    })
    fit_results_gpa = plotting_tools.calculate_linear_fit(df_gpa, "x_data", "y_data", "MPa")
    assert np.isclose(fit_results_gpa["modulus_val"], 1.0)
    assert fit_results_gpa["modulus_units"] == "GPa"

    # Test auto-scaling of modulus units (MPa to kPa)
    df_kpa = pd.DataFrame({
        "x_data": [1, 2, 3],
        "y_data": [0.001, 0.002, 0.003],  # 0.001 MPa/unit = 1 kPa/unit
    })
    fit_results_kpa = plotting_tools.calculate_linear_fit(df_kpa, "x_data", "y_data", "MPa")
    assert np.isclose(fit_results_kpa["modulus_val"], 1.0)
    assert fit_results_kpa["modulus_units"] == "kPa"

    # Insufficient data
    df_single = pd.DataFrame({"x_data": [1], "y_data": [2]})
    fit_results_single = plotting_tools.calculate_linear_fit(df_single, "x_data", "y_data", "MPa")
    assert fit_results_single == {}

    df_empty = pd.DataFrame({"x_data": [], "y_data": []})
    fit_results_empty = plotting_tools.calculate_linear_fit(df_empty, "x_data", "y_data", "MPa")
    assert fit_results_empty == {}


def test_calculate_torsional_properties_rect():
    """Verify that torsional shear stress and strain are calculated correctly."""
    test_data = {
        ROTATION_COL: [0, 45, 90],
        TORQUE_COL: [0, 10, 20],
    }
    df = pd.DataFrame(test_data)

    geometry = {
        "gauge_length_mm": 100.0,
        "torsional_side1_mm": 20.0,  # long side
        "torsional_side2_mm": 10.0,  # short side
    }

    result_df = torsional_analysis.calculate_torsional_properties_rect(df, geometry)

    # Expected rotation @ 90deg = pi/2 radians
    # Expected shear strain @ 90deg = (short_side_m * rad) / L_m
    # = (0.010m * pi/2) / 0.100m = pi/20 = 0.157
    assert SHEAR_STRAIN_COL in result_df.columns
    assert np.isclose(result_df[SHEAR_STRAIN_COL].iloc[-1], np.pi / 20)

    # For aspect_ratio = 20/10 = 2
    # alpha = (1/3) * (1 - 0.630 * (1/2)) = (1/3) * (1 - 0.315) = (1/3) * 0.685 = 0.22833
    # denominator = alpha * long_side_m * (short_side_m**2)
    #             = 0.22833 * 0.020 * (0.010**2) = 0.22833 * 0.020 * 0.0001 = 4.5666e-7
    # Shear Stress (Pa) @ 20 N.m = 20 / 4.5666e-7 = 43,799,000 Pa = 43.799 MPa
    assert SHEAR_STRESS_MPA_COL in result_df.columns
    expected_alpha = (1 / 3) * (1 - 0.630 * (1 / 2))
    expected_denominator = expected_alpha * 0.020 * (0.010**2)
    expected_shear_stress_pa = 20 / expected_denominator
    expected_shear_stress_mpa = expected_shear_stress_pa / 1e6
    assert np.isclose(result_df[SHEAR_STRESS_MPA_COL].iloc[-1], expected_shear_stress_mpa)


def test_calculate_torsional_properties_rect_zero_short_side(caplog):
    """Test torsional property calculation with zero short side dimension."""
    test_data = {
        ROTATION_COL: [0, 45, 90],
        TORQUE_COL: [0, 10, 20],
    }
    df = pd.DataFrame(test_data)
    geometry = {
        "gauge_length_mm": 100.0,
        "torsional_side1_mm": 20.0,
        "torsional_side2_mm": 0.0,  # Zero short side
    }
    with caplog.at_level(logging.ERROR):
        result_df = torsional_analysis.calculate_torsional_properties_rect(df, geometry)
        assert "Short side dimension is zero" in caplog.text
    assert SHEAR_STRESS_MPA_COL not in result_df.columns  # Should not add stress columns
    assert SHEAR_STRAIN_COL not in result_df.columns # Shear strain also depends on short side


def test_calculate_torsional_properties_rect_missing_geometry_keys(caplog):
    """Test torsional property calculation with missing geometry keys."""
    test_data = {
        ROTATION_COL: [0, 45, 90],
        TORQUE_COL: [0, 10, 20],
    }
    df = pd.DataFrame(test_data)
    geometry_missing_side1 = {
        "gauge_length_mm": 100.0,
        "torsional_side2_mm": 10.0,
    }
    with caplog.at_level(logging.ERROR):
        result_df = torsional_analysis.calculate_torsional_properties_rect(df, geometry_missing_side1)
        assert "Missing key in geometry config: 'torsional_side1_mm'" in caplog.text
    assert SHEAR_STRESS_MPA_COL not in result_df.columns
    assert SHEAR_STRAIN_COL not in result_df.columns


def test_calculate_torsional_properties_rect_square_section():
    """Verify torsional properties for a square cross-section."""
    test_data = {
        ROTATION_COL: [0, 45, 90],
        TORQUE_COL: [0, 10, 20],
    }
    df = pd.DataFrame(test_data)
    geometry = {
        "gauge_length_mm": 100.0,
        "torsional_side1_mm": 10.0,
        "torsional_side2_mm": 10.0,
    }
    result_df = torsional_analysis.calculate_torsional_properties_rect(df, geometry)

    # For aspect_ratio = 10/10 = 1
    # alpha = (1/3) * (1 - 0.630 * (1/1)) = (1/3) * (1 - 0.630) = (1/3) * 0.37 = 0.12333
    # Note: The comment in torsional_analysis.py says alpha = 0.1406 for square.
    # The formula used is an approximation. Let's use the formula's result.
    expected_alpha = (1 / 3) * (1 - 0.630 * (1 / 1))
    expected_denominator = expected_alpha * 0.010 * (0.010**2)
    expected_shear_stress_pa = 20 / expected_denominator
    expected_shear_stress_mpa = expected_shear_stress_pa / 1e6

    assert SHEAR_STRESS_MPA_COL in result_df.columns
    assert np.isclose(result_df[SHEAR_STRESS_MPA_COL].iloc[-1], expected_shear_stress_mpa)
    assert np.isclose(result_df[SHEAR_STRAIN_COL].iloc[-1], np.pi / 20)


def test_calculate_torsional_properties_rect_empty_df():
    """Test torsional property calculation with an empty DataFrame."""
    df = pd.DataFrame({ROTATION_COL: [], TORQUE_COL: []})
    geometry = {
        "gauge_length_mm": 100.0,
        "torsional_side1_mm": 20.0,
        "torsional_side2_mm": 10.0,
    }
    result_df = torsional_analysis.calculate_torsional_properties_rect(df, geometry)
    assert result_df.empty
    assert SHEAR_STRESS_MPA_COL in result_df.columns # Columns are added, but will be empty
    assert SHEAR_STRAIN_COL in result_df.columns


def test_calculate_torsional_properties_rect_negative_values():
    """Test torsional property calculation with negative torque and rotation."""
    test_data = {
        ROTATION_COL: [0, -45, -90],
        TORQUE_COL: [0, -10, -20],
    }
    df = pd.DataFrame(test_data)
    geometry = {
        "gauge_length_mm": 100.0,
        "torsional_side1_mm": 20.0,
        "torsional_side2_mm": 10.0,
    }
    result_df = torsional_analysis.calculate_torsional_properties_rect(df, geometry)

    expected_alpha = (1 / 3) * (1 - 0.630 * (1 / 2))
    expected_denominator = expected_alpha * 0.020 * (0.010**2)
    expected_shear_stress_pa = -20 / expected_denominator
    expected_shear_stress_mpa = expected_shear_stress_pa / 1e6

    assert np.isclose(result_df[SHEAR_STRAIN_COL].iloc[-1], -np.pi / 20)
    assert np.isclose(result_df[SHEAR_STRESS_MPA_COL].iloc[-1], expected_shear_stress_mpa)
