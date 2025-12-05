"""
This script defines a test configuration for analyzing BlueHill data
from a silicone 30a axial pull experiment. It uses the 'matmech'
package to process data and generate plots.
"""

import os

# This script now relies on the 'matmech' package
# being installed in editable mode (pip install -e .).
from matmech.workflow import run_analysis_workflow

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

user_config = {
    "software_type": "bluehill",
    "data_file_name": "strain_data.csv",
    "test_recipe": [
        {"name": "axial_pull", "type": "AXIAL"},
    ],
    "geometry": {
        "axial_width_mm": 30.0,
        "axial_thickness_mm": 2.65,
        "gauge_length_mm": 110.0,
        "torsional_side1_mm": 1.0, # Placeholder, not used for axial only
        "torsional_side2_mm": 1.0  # Placeholder, not used for axial only
    },

    # --- Optional inversion and taring settings ---
    "inversion_flags": {
        "force": False,
        "torque": False
    },
    "tare_options": {
        "position": True,
        "force": True
    },

    "plots": [
        "time_position_static",
        "force_position_static",
        "stress_strain_static",
        "time_force_static",
        "time_stress_static",
        "time_strain_static",
        # You can also add custom plot configurations here, e.g.:
        # {
        #     "x_col": "time",
        #     "y_col": "force",
        #     "title": "{phase_name} - Custom Force vs. Time",
        #     "output_filename": "{phase_name}_custom_force_time",
        #     "phases": ["axial_pull"],
        #     "type": "static",
        # }
    ],
}


if __name__ == "__main__":
    run_analysis_workflow(SCRIPT_DIR, user_config)
