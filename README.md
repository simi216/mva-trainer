# ttbar-mva-trainer
A standalone analysis framework for training and evaluating machine learning models for reconstruction of dileptonic ttbar events. The framework includes data preprocessing using C++, data loading and preprocessing using Python, model training and evaluation using TensorFlow, and integration with the Condor job scheduler for distributed training and evaluation.
The models are designed to perform both reconstruction of the neutrino momenta as well as assignment of jets to the corresponding b-quarks from the top quark decays.

To inject the trained machine learning models into the TopCPToolKit, the models can be exported to ONNX format. Currently, only models providing jet-to-quark assignments as output can be exported to ONNX format for use in the TopCPToolKit.


## Setup

The code can be run in a virtual environment. To set up the virtual environment, you can run the following commands:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Overview
This repository contains a standalone analysis framework for training and evaluating machine learning models on ttbar events.

## Feature Overview

- **Preprocessing**: The preprocessing of the data is done using C++ and is defined in the `preprocessing` directory. The preprocessing is done using the `Preprocessor` class, which reads the data from the input file and preprocesses it.
- **Data Loading**: The data loading and preprocessing is done using the `DataPreprocessor` class, which is defined in the `core/DataLoader.py` file. This class is used to load the data from the preprocessed files and arrange it in a format that can be used for training and evaluation.
- **Reconstruction Models**: The reconstruction models are defined in the `core/reconstruction` directory. The base class for all reconstruction models is the `BaseReconstructor` class, which is defined in the `core/reconstruction/Reconstruction.py` file. Machine learning-based reconstruction models are to be implemented by inheriting from the `MLReconstructorBase` class, while baseline reconstruction models are to be implemented by inheriting from the `BaselineReconstructor` class.
- **Model Training**: The training of the models is handled by the `MLReconstructorBase` class which provides methods for training and evaluating machine learning-based reconstruction models.
- **Evaluation**: The evaluation of the models is done using the `ReconstructionEvaluator` class, which is defined in the `core/reconstruction/Evaluation.py` file. This class provides methods for evaluating the performance of various reconstruction models using different metrics and visualizations.
- **Export Models for use in TopCPToolKit**: The trained machine learning models can be exported for use in the TopCPToolKit. The `export_to_onnx` method in the `MLWrapperBase` class is used to export the trained model to a format that can be used in the TopCPToolKit.

- **Condor Integration**: The framework includes integration with the Condor job scheduler for distributed training and evaluation. The Condor scripts are located in the `CONDOR` directory.

## Prepocessing
The preprocessing is done using C++ and is defined in the `preprocessing` directory. Header files for the preprocessing are located in the `preprocessing/include` directory, while the implementation files are in the `preprocessing/src` directory.

The preprocessing is done using the `Preprocessor` class, which is defined in the `preprocessing/src/preprocessor.cpp` file. The `Preprocessor` class is used to preprocess the data and save it to a file.
It uses the `reco` class, which is defined in the `preprocessing/src/reco_mc_20.cpp`. It is used to read the data from the input file and preprocess it. This class can be automatically generated using `ROOT` and the `MakeClass()` method defined on the `TTree` class. The names of the variables using in the `PreProcessor` class have to match the names of the variables in the `reco` class.

The directory `preprocessing/scripts` contains scripts for running the preprocessing. The script `run_preprocessing.sh` is used to run the preprocessing on a single file. The script `merge_root_files.cpp` is used to merge the preprocessed files into a single file. The script `run_merge.sh` is used to run the merging of the preprocessed files. This mostly done to merge preprocessed files that are run sequentially using CONDOR. The script `run_preprocessing_condor.sh` is used to run the preprocessing on multiple files using Condor.

The code is compiled using `Makefile` in the `preprocessing` directory. The Makefile is used to compile the code and create the executable files.

The executable files are located in the `preprocessing/bin` directory. The executable files are used to run the preprocessing and merging of the preprocessed files. The `Makefile` also contains rules for cleaning the build directory and removing the executable files.

## Data Loading
To load the data for training and evaluation, the `DataPreprocessor` class is used, which is defined in the `core/DataLoader.py` file. This class is used to load the data from the preprocessed files and arange it in a format that can be used for training and evaluation. The `DataPreprocessor` class is used to load the data from the preprocessed files and arrange it in a format that can be used for training and evaluation. It also provides methods for splitting the data into training and testing sets, as well as providing k-folds of the data for cross-validation.

## Reconstruction Models
The base class for all reconstruction models is the `BaseReconstructor` class, which is defined in the `core/reconstruction/Reconstruction.py` file. This class provides the basic functionality for all reconstruction models regardless of the underlying model type.

### Machine Learning Models
Machine learning-based reconstruction models are to be implemented by inheriting from the `MLReconstructorBase` class, which is defined in the `core/reconstruction/Reconstruction.py` file. This class provides additional functionality for machine learning-based reconstruction models, such as training and evaluation methods and handling the input and output data formats.

To implement a new machine learning-based reconstruction model, you need to create a new class that inherits from the `MLReconstructorBase` class and implement the following method:
- `build_model(self, **kwargs)`: This method is used to define the architecture of the machine learning model. You can use TensorFlow/Keras to define the model architecture.

### Baseline Models
Baseline reconstruction models are to be implemented by inheriting from the `BaselineReconstructor` class, which is defined in the `core/assignment_models/BaseLineAssingmentMethods.py` file. This class provides the functionality for baseline reconstruction models, such as simple heuristic-based assignment methods.


## Model Training
The training of the models is handled by the `MLReconstructorBase` class which provides methods for training and evaluating machine learning-based reconstruction models. The training process involves loading the data using the `DataPreprocessor` class, preprocessing the data, and then training the model using the specified architecture and hyperparameters.


## Evaluation
To evaluate the performance of the reconstruction models, the `ReconstructionEvaluator` class is used, which is defined in the `core/reconstruction/Evaluation.py` file. This class provides methods for evaluating the performance of various reconstruction models using different metrics and visualizations. The evaluation process involves loading the test data, making predictions using the trained models, and then calculating various performance metrics such as accuracy, precision, recall, and F1-score. The class also provides methods for visualizing the results using plots and histograms.

To evaluate metrics for machine learning-based reconstructors, the `MLEvaluator` method is used. This method provides functionality for evaluating machine learning-based reconstruction models using various metrics and visualizations.


## Export Models for use in TopCPToolKit
The trained machine learning models can be exported for use in the TopCPToolKit. The `export_to_onnx` method in the `MLWrapperBase` class is used to export the trained model to a format that can be used in the TopCPToolKit. The exported model can then be integrated into the TopCPToolKit for use in physics analyses. For further use, please refer to the TopCPToolKit documentation and the DiLepTagger module.
Note: So far, only models providing assignments as output can be exported to ONNX format for use in the TopCPToolKit.

## Condor Integration
### Hyperparameter Grid Search
The framework includes integration with the Condor job scheduler for distributed training and evaluation. The Condor scripts are located in the `CONDOR` directory. The `HypParamGridSearch` subdirectory contains scripts for performing hyperparameter grid search using Condor. The `run_training.sh` script is used to run the training of the models using Condor. The `train_hyperparameter.py` script is used to train the models with specified hyperparameters.

### Distributed PreProcessing
The `Preprocessing` subdirectory contains scripts for performing distributed preprocessing using Condor. The `run_preprocessing.sh` script is used to run the preprocessing of the data using Condor. The `submitPreProcessing.sh` script is used to submit the preprocessing jobs to Condor. Provide the `input_file` argument to the script to specify the input file for preprocessing.



## Dependencies
The code is written in Python 3.9 and the dependencies are managed using `pip`. The required dependencies are listed in the `requirements.txt` file. To install the dependencies, you can run the following command:

```bash
pip install -r requirements.txt
```

or if you want to install the dependencies in a virtual environment, you can run the following commands:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
