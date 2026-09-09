#!/bin/bash

# Batch screenshot generator for all *valid.txt files in *_dsl_patterns directories
# Output will be saved to <model>_img/ directories (e.g., claude_img/, blip_img/, etc.)
# This script activates the virtual environment and runs the Python batch processor

# Activate virtual environment
source env/bin/activate

# Run the batch screenshot script with all arguments passed through
python batch_screenshot_valid_patterns.py "$@"
