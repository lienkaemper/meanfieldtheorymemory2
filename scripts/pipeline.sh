#!/bin/sh

python scripts/simulate_for_h_range.py
python scripts/analyze_for_h_range.py
python scripts/predict_for_h_range.py
python scripts/plot_theory_vs_sim_by_h.py
python scripts/plot_weights.py
python scripts/raster_plot.py