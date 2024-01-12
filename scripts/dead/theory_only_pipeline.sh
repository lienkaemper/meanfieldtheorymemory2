#!/bin/sh

python scripts/predict_for_h_range.py
python scripts/noisy_tagging.py
python scripts/plot_theory.py show
