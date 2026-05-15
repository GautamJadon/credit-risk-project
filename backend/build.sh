#!/usr/bin/env bash
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Train ML model and save artifacts
cd ml
python train_model.py
cd ..