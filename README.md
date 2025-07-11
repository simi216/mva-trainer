# ttbar-mva-trainer


## Overview
This repository contains a standalone analysis framework for training and evaluating machine learning models on ttbar events.

## Feature Overview

- **Preprocessing**: Data preprocessing using C++.

- **Data Loading**: Loading and preprocessing data using Python. The data loading and preprocessing is defined in the `core` directory.

- **Model Training**: Training and evaluating machine learning models using TensorFlow. 

- **Evaluation**: Evaluation of machine learning models using K-Fold cross-validation.

- **Condor Integration**: Integration with the Condor job scheduler for distributed training and evaluation.


## Prepocessing
The preprocessing is done using C++ and is defined in the `preprocessing` directory. Header files for the preprocessing are located in the `preprocessing/include` directory, while the implementation files are in the `preprocessing/src` directory.

The preprocessing is done using the `Preprocessor` class, which is defined in the `preprocessing/src/preprocessor.cpp` file. The `Preprocessor` class is used to preprocess the data and save it to a file.
It uses the `reco` class, which is defined in the `preprocessing/src/reco_mc_20.cpp`. It is used to read the data from the input file and preprocess it. This class can be automatically generated using `ROOT` and the `MakeClass()` method defined on the `TTree` class. The names of the variables using in the `PreProcessor` class have to match the names of the variables in the `reco` class.

The directory `preprocessing/scripts` contains scripts for running the preprocessing. The script `run_preprocessing.sh` is used to run the preprocessing on a single file. The script `merge_root_files.cpp` is used to merge the preprocessed files into a single file. The script `run_merge.sh` is used to run the merging of the preprocessed files. This mostly done to merge preprocessed files that are run sequentially using CONDOR. The script `run_preprocessing_condor.sh` is used to run the preprocessing on multiple files using Condor.

The code is compiled using `Makefile` in the `preprocessing` directory. The Makefile is used to compile the code and create the executable files.

The executable files are located in the `preprocessing/bin` directory. The executable files are used to run the preprocessing and merging of the preprocessed files. The `Makefile` also contains rules for cleaning the build directory and removing the executable files.

## Data Loading
To load the data for training and evaluation, the `DataPreprocessor` class is used, which is defined in the `core/DataLoader.py` file. This class is used to load the data from the preprocessed files and arange it in a format that can be used for training and evaluation. The `DataPreprocessor` class is used to load the data from the preprocessed files and arrange it in a format that can be used for training and evaluation. It also provides methods for splitting the data into training and testing sets, as well as providing k-folds of the data for cross-validation.

## Model Training
The model training is for assignment and regression tasks is descripted in `core/AssignmentBaseModel.py` and `core/RegressionBaseModel.py` respectively. The files contain the `AssignmentBaseModel` and `RegressionBaseModel` classes, which are used to train and evaluate machine learning models using TensorFlow. The `AssignmentBaseModel` class is used for training and evaluating models for assignment tasks, while the `RegressionBaseModel` class is used for training and evaluating models for combined regression and assignment tasks.

To implement a model, you need to create a class that inherits from either `AssignmentBaseModel` or `RegressionBaseModel`. You can then implement the method `build_model(**kwargs)` to define the model architecture.

## Evaluation
The evaluation of the machine learning models is done using K-Fold cross-validation. The `AssignmentKFold` and `RegressionKFold` classes are defined in the `core/AssignmentKFold.py` and `core/RegressionKFold.py` files respectively and are used to evaluate the models using K-Fold cross-validation. The classes provides methods for training and evaluating arbitrary models derived from the base classes `AssignmentBaseModel` and `RegressionBaseModel`. So aslong as a custom-model is formatted to obey the in- and output-interfaces of the base classes, it can be used in the whole framework.

## Dependencies
The code is written in Python 3 and the dependencies are managed using `pip`. The required dependencies are listed in the `requirements.txt` file. To install the dependencies, you can run the following command:

```bash
pip install -r requirements.txt
```

or if you want to install the dependencies in a virtual environment, you can run the following commands:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
