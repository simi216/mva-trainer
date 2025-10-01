#!/bin/bash

echo "Working dir: $PWD"
source ~/mva_trainer/venv/bin/activate

echo "Command: $@"
eval "$@"