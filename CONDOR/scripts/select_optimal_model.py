import os
import sys
import numpy as np
import string

def main(argv):
    model_params = {}
    PLOT_DIR = argv[1] + "/"
    for directory in os.listdir(PLOT_DIR):
        if not os.path.isfile(PLOT_DIR + directory + "/accuracies.dat"):
            continue
        with open(PLOT_DIR + directory + "/accuracies.dat", "r") as f:
            line = f.readline()
            line = line.split(" ")
            line = [x for x in line if x != ""]
            line = [x for x in line if x != "\n"]
            line = [x for x in line if x != "\r"]
            line = [x for x in line if x != "\t"]
            line = [float(x) for x in line]
            if len(line) < 3:
                continue
            model_params[tuple(directory.split())] = line
    # Sort the dictionary by the maximum accuracy
    sorted_model_params = sorted(model_params, key=lambda x: (model_params[x][-2]), reverse=True)

    # Get the top 10 models
    top_10_models = sorted_model_params[:10]
    # Print the top 10 models
    print("Top 10 models:")
    for model in top_10_models:
        print(model, model_params[model][-2], model_params[model][-1])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python select_optimal_model.py <path_to_plot_directory>")
        sys.exit(1)
    main(sys.argv)