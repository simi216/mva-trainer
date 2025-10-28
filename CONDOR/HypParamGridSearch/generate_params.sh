#!/bin/bash
# generate_params.sh

> params.txt  # Clear the file

for arch in FeatureConcatTransformer CrossAttentionTransformer; do
    for hidden in 32 64 128 256; do
        for layers in 4 6 8 10; do
            for heads in 4 8 16; do
                echo "$hidden $layers $heads $arch" >> params.txt
            done
        done
    done
done

echo "Generated $(wc -l < params.txt) combinations"