import sys
sys.path.append("..")
from core.DataLoader import DataPreprocessor, DataConfig, LoadConfig, get_load_config_from_yaml
import numpy as np
from importlib import reload
import matplotlib.pyplot as plt
import yaml
import core.regression_models as Models
import core
import keras
import parser


PLOTS_DIR = f"plots/regresion_transformer/"
MODEL_DIR  = f"models/regresion_transformer/"
import os

def parse_args():
    parser = parser.ArgumentParser(description="Train Regression Transformer Model")
    parser.add_argument(
        "--data_config",
        type=str,
        default="workspace_config.yaml",
        help="Path to the data configuration YAML file.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data_config_path = args.data_config
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)


    load_config = get_load_config_from_yaml(data_config_path)

    DataProcessor = DataPreprocessor(load_config)

    data_config_path = "workspace_config.yaml"
    with open(data_config_path, "r") as file:
        data_configs = yaml.safe_load(file)

    data_config = DataProcessor.load_data(
        data_configs["data_path"]["nominal"], "reco", max_events=4000000
    )
    X_train, y_train, X_val, y_val = DataProcessor.split_data(test_size=0.1)


    TransformerMatcher = Models.FeatureConcatTransformer(data_config, name="Transformer")

    TransformerMatcher.build_model(
        num_heads=8,
        hidden_dim=128,
        num_layers=8,
        dropout_rate=0.1,
        input_as_four_vector=True,
    )

    TransformerMatcher.adapt_normalization_layers(X_train)

    learn_rate = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3,
        first_decay_steps=1000,
        t_mul=2.0,
        m_mul=1.0,
        alpha=1e-6,
    )

    TransformerMatcher.compile_model(
        loss={
            "assignment": core.utils.AssignmentLoss(),
            "regression": core.utils.RegressionLoss(),
        },
        optimizer=keras.optimizers.AdamW(learning_rate=learn_rate, weight_decay=1e-4),
        metrics=
            {
                "assignment": [core.utils.AssignmentAccuracy(name="accuracy")],
                "regression": [core.utils.RegressionRelativeError(name="relative_error")],
            }
    )
    TransformerMatcher.load_model(MODEL_DIR + "model.keras")
    TransformerMatcher.model.summary()



    TransformerMatcher.train_model(
        epochs=50,
        X_train=X_train,
        y_train=y_train,
        sample_weights=core.utils.compute_sample_weights(X_train, y_train),
        batch_size=1028,
        callbacks=keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=50, restore_best_weights=True, mode="min"
        ),
    )
    TransformerMatcher.save_model(MODEL_DIR + "model.keras")



if __name__ == "__main__":
    main()