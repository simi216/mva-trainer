#!/bin/bash

echo "Working dir: $PWD"
source ~/mva_trainer/env/bin/activate

echo "Command: $@"
eval "$@"