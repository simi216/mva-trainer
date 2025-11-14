#!/bin/bash
# generate_params.sh

> params.txt  # Clear the file

for arch in FeatureConcatTransformer; do
    for hidden in 128 256; do
        for central_layers in 6; do
            for regression_layers in 2 4 6; do
                echo "$hidden $central_layers $regression_layers $arch" >> params.txt
            done
        done
    done
done

echo "Generated $(wc -l < params.txt) combinations"