import sys

sys.path.append("..")
from core.DataLoader import (
    DataPreprocessor,
    get_load_config_from_yaml,
)
from importlib import reload
import yaml
import core
import keras
import core.keras_models.RegressionTransformer as RegressionTransformer
import numpy as np

ROOT_DIR = "/afs/desy.de/user/a/aulich/mva-trainer/"
PLOTS_DIR = ROOT_DIR + "plots/regression_transformer_old/"
MODEL_DIR = ROOT_DIR + "models/regression_transformer_old/"
CONFIG_PATH = ROOT_DIR + "config/workspace_config.yaml"

import os

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


load_config = get_load_config_from_yaml(CONFIG_PATH)

DataProcessor = DataPreprocessor(load_config)

with open(CONFIG_PATH, "r") as file:
    data_configs = yaml.safe_load(file)

data_config = DataProcessor.load_from_npz(
    load_config["nominal"], max_events=10_000_000, event_numbers="even"
)

X_train, y_train = DataProcessor.get_data()

del DataProcessor  # Free memory

Transformer = RegressionTransformer.FeatureConcatTransformer(data_config)


Transformer.build_model(
    hidden_dim=256,
    num_heads=8,
    num_layers=10,
    dropout_rate=0.1,
    compute_HLF=True,
    log_variables=True,
)

Transformer.adapt_normalization_layers(X_train)

Transformer.compile_model(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
    loss={
        "normalized_regression": core.utils.RegressionLoss(),
        "assignment": core.utils.AssignmentLoss(),
    },
    metrics={
        "normalized_regression": core.utils.RegressionDeviation(),
        "assignment": core.utils.AssignmentAccuracy(),
    },
    loss_weights={
        "normalized_regression": 3.0,
        "assignment": 1.0,
    },
)

Transformer.train_model(
    X_train,
    y_train,
    batch_size=1024,
    epochs=100,
    copy_data=False,
    callbacks=keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        mode="min",
    ),
)

Transformer.save_model(MODEL_DIR + "model.keras")