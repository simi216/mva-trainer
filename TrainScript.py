import sys
import argparse
import os
import numpy as np
import keras
import matplotlib.pyplot as plt
import yaml
import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


from core import keras_models
from core import utils
from core import DataConfig, LoadConfig


@dataclass
class TrainConfig:
    batch_size: int = 1024
    epochs: int = 50
    callbacks: Optional[Dict[str, Dict[str, any]]] = field(default_factory=dict)
    validation_split: float = 0.1
    shuffle: bool = True

    def to_dict(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "validation_split": self.validation_split,
            "shuffle": self.shuffle,
        }


@dataclass
class ModelOptions:
    compute_HLF: bool = False
    log_variables: bool = False
    use_global_event_features = (False,)
    dropout_rate: float = 0.1

    def to_dict(self) -> dict:
        return {
            "compute_HLF": self.compute_HLF,
            "log_variables": self.log_variables,
            "use_global_event_features": self.use_global_event_features,
            "dropout_rate": self.dropout_rate,
        }


@dataclass
class CompileOptions:
    loss: Dict[str, any] = field(
        default_factory=lambda: {"assignment_output": "AssignmentLoss"}
    )
    metrics: Dict[str, any] = field(
        default_factory=lambda: {"assignment_output": ["AssignmentAccuracy"]}
    )
    optimizer: Dict[str, any] = field(
        default_factory=lambda: {
            "class_name": "Adam",
            "config": {"learning_rate": 0.001},
        }
    )


@dataclass
class ModelConfig:
    model_type: str = "FeatureConcatTransformer"
    model_options: ModelOptions = field(default_factory=ModelOptions)
    model_params: Dict[str, any] = field(default_factory=dict)
    compile_options: CompileOptions = field(default_factory=CompileOptions)


from core.DataLoader import (
    DataPreprocessor,
    DataConfig,
    LoadConfig,
    get_load_config_from_yaml,
)


def parse_args():
    """Parse command line arguments for running the training script."""
    parser = argparse.ArgumentParser(
        description="Train Transformer model with specified hyperparameters"
    )

    # Configuration file arguments
    parser.add_argument(
        "--load_config",
        type=str,
        required=True,
        help="Path to the load configuration YAML file",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        required=True,
        help="Path to the training configuration YAML file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to the model configuration YAML file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save trained models and logs",
    )

    parser.add_argument(
        "--event_numbers",
        type=str,
        default="all",
        help="Comma-separated list of event numbers to use for training (optional)",
    )

    parser.add_argument(
        "--max_events",
        type=int,
        default=None,
        help="Maximum number of events to load from the dataset (optional)",
    )

    return parser.parse_args()


def load_yaml_config(file_path):
    """Load a YAML configuration file."""
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    args = parse_args()

    # Load configurations
    load_config = LoadConfig(**load_yaml_config(args.load_config)["LoadConfig"])
    train_config = TrainConfig(**load_yaml_config(args.train_config)["TrainConfig"])
    model_config = ModelConfig(**load_yaml_config(args.model_config)["ModelConfig"])
    model_config.compile_options = CompileOptions(**model_config.compile_options)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load data
    data_preprocessor = DataPreprocessor(load_config)
    data_config = data_preprocessor.load_from_npz(
        load_config.data_path["nominal"],
        event_numbers=args.event_numbers,
        max_events=args.max_events,
    )
    X, y = data_preprocessor.get_data()

    # Build model
    model = keras_models._get_model(model_config.model_type)
    even_trained_model = model(data_config)

    build_options = model_config.model_options
    build_options.update(**model_config.model_params)

    print(build_options)
    even_trained_model.build_model(**(build_options))

    # Compile Models
    even_trained_model.compile_model(
        optimizer=keras.optimizers.get(model_config.compile_options.optimizer),
        loss={
            key: getattr(utils, value["class_name"])(**value.get("config", {}))
            for key, value in model_config.compile_options.loss.items()
        },
        metrics={
            key: [
                getattr(utils, metric["class_name"])(**metric.get("config", {}))
                for metric in value
            ]
            for key, value in model_config.compile_options.metrics.items()
        },
    )
    # Adapt normalization layers
    even_trained_model.adapt_normalization_layers(X)

    train_options = train_config.to_dict()
    # Prepare Callbacks
    callbacks = []
    for callback_name, callback_params in train_config.callbacks.items():
        callback_class = getattr(keras.callbacks, callback_name)
        callbacks.append(callback_class(**callback_params))
    train_options["callbacks"] = callbacks

    # Train models
    even_history = even_trained_model.train_model(
        X,
        y,
        **train_options,
    )

    # Save models and training history
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.event_numbers == "even":
        even_trained_model.save_model(os.path.join(args.output_dir, f"odd_model.keras"))
        even_trained_model.export_to_onnx(
            os.path.join(args.output_dir, "odd_model.onnx")
        )
    elif args.event_numbers == "odd":
        even_trained_model.save_model(
            os.path.join(args.output_dir, f"even_model.keras")
        )
        even_trained_model.export_to_onnx(
            os.path.join(args.output_dir, "even_model.onnx")
        )
    else:
        even_trained_model.save_model(os.path.join(args.output_dir, f"model.keras"))
        even_trained_model.export_to_onnx(os.path.join(args.output_dir, "model.onnx"))
