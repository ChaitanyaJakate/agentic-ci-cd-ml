#!/bin/bash

if [ -f model_type.txt ]; then
  TYPE=$(cat model_type.txt)
  echo "Running model: $TYPE"
  python scripts/run_${TYPE}.py
else
  echo "No model_type.txt found"
fi
