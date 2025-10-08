#!/bin/bash

echo "Working dir: $PWD"
source ~/mva-trainer/venv/bin/activate

echo "Command: $@"
eval "$@"