#!/bin/bash

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# Ensure DM Sans font is installed (for pareto_plots.py)
python scripts/install_dm_sans_font.py

# python scripts/plots/plot_inf_budgets.py
python scripts/plots/plot_masking_patterns.py
# python scripts/plots/plot_schedule_functions.py
