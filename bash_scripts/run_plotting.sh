#!/bin/bash

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

python scripts/plots/plot_inf_budgets.py