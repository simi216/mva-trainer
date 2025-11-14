#!/bin/bash
# generate_params.sh

> params.txt  # Clear the file

for arch in FeatureConcatTransformer; do
    for high_level_features in "--use_high_level_features" ""; do
        for hidden in 256; do
            for layers in 4 6 8 10; do
                for heads in 8; do
                    echo "$hidden $layers $heads $arch $high_level_features" >> params.txt
                done
            done
        done
    done
done

echo "Generated $(wc -l < params.txt) combinations"