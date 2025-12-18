import sys
import argparse
import os
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import yaml
import tensorflow as tf


def parse_args():
    """Parse command line arguments for hyperparameter tuning."""
    parser = argparse.ArgumentParser(
        description="Train Transformer model with specified hyperparameters"
    )

    # Hyperparameters
    parser.add_argument(
        "--hidden_dim",
        type=int,
        required=True,
        help="Hidden dimension size for the transformer",
    )
    parser.add_argument(
        "--num_layers", type=int, required=True, help="Number of transformer layers"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="FeatureConcatTransformer",
        help="Model architecture to use (default: FeatureConcatTransformer)",
    )
    # Optional hyperparameters with defaults
    parser.add_argument(
        "--dropout_rate", type=float, default=0.1, help="Dropout rate (default: 0.1)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1028, help="Batch size (default: 128)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs (default: 50)"
    )
    parser.add_argument(
        "--patience", type=int, default=50, help="Early stopping patience (default: 50)"
    )
    parser.add_argument(
        "--num_heads", type=int, required=True, help="Number of attention heads"
    )

    # Data and directory parameters
    parser.add_argument(
        "--max_events",
        type=int,
        default=4000000,
        help="Maximum number of events to load (default: 4000000)",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=".",
        help="Root directory for outputs",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="nominal",
        help="Type of data to use (default: nominal)",
    )

    return parser.parse_args()


def setup_directories(root_dir, model_name):
    """Create necessary directories if they don't exist."""
    if not os.path.isabs(root_dir):
        root_dir = os.path.abspath(root_dir)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)
    plots_dir = os.path.join(root_dir, "plots", model_name)
    model_dir = os.path.join(root_dir, "models", model_name)

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    return plots_dir, model_dir


def main():
    # Parse arguments
    args = parse_args()

    sys.path.append(args.root_dir)  # Ensure root directory is in the path
    import core.assignment_models as Models
    from core.DataLoader import (
        DataPreprocessor,
        DataConfig,
        LoadConfig,
        get_load_config_from_yaml,
    )
    import core

    # Load data configuration
    config_dir = os.path.join(args.root_dir, "config")
    CONFIG_PATH = os.path.join(config_dir, "workspace_config.yaml")

    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Data configuration file not found: {CONFIG_PATH}")

    load_config = get_load_config_from_yaml(CONFIG_PATH)
    # Create model name with hyperparameters
    MODEL_NAME = (
        f"{args.architecture}_d{args.hidden_dim}_l{args.num_layers}_h{args.num_heads}"
    )

    # Setup directories
    PLOTS_DIR, MODEL_DIR = setup_directories(args.root_dir, MODEL_NAME)

    print(f"Starting training with hyperparameters:")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Dropout rate: {args.dropout_rate}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Model name: {MODEL_NAME}")

    # Configure data
    with open(CONFIG_PATH, "r") as f:
        data_config_yaml = yaml.safe_load(f)

    # Load and preprocess data
    print("Loading data...")
    DataProcessor = DataPreprocessor(load_config)
    config = DataProcessor.load_from_npz(
        data_config_yaml["data_path"][args.data_type],
        max_events=args.max_events,
    )

    print("Splitting data...")
    X_train, y_train, X_val, y_val = DataProcessor.split_data(
        test_size=0.1, random_state=42
    )
    del DataProcessor  # Free memory

    # Build model
    print("Building model...")
    if args.architecture == "FeatureConcatTransformer":
        Model = Models.FeatureConcatTransformer(config, name="Transformer")
        Model.build_model(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout_rate=args.dropout_rate,
            log_variables=True,
        )
    elif args.architecture == "FeatureConcatRNN":
        Model = Models.FeatureConcatRNN(config, name="RNN")
        Model.build_model(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate,
        )
    elif args.architecture == "CrossAttentionTransformer":
        Model = Models.CrossAttentionModel(config, name="RNN")
        Model.build_model(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout_rate=args.dropout_rate,
        )
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")

    # Adapt normalization and compile
    print("Adapting normalization layers...")
    Model.adapt_normalization_layers(X_train)

    print("Compiling model...")
    Model.compile_model(
        loss=core.utils.AssignmentLoss(lambda_excl=0),
        optimizer=keras.optimizers.AdamW(
            learning_rate=args.learning_rate, weight_decay=args.weight_decay
        ),
        metrics=[core.utils.AssignmentAccuracy()],
    )

    # Count trainable parameters
    trainable_params = sum(
        [np.prod(var.shape) for var in Model.model.trainable_variables]
    )
    print(f"Total trainable parameters: {trainable_params:,}")

    # Train model
    print("Training model...")
    history = Model.train_model(
        epochs=args.epochs,
        X_train=X_train,
        y_train=y_train,
        sample_weights=core.utils.compute_sample_weights(X_train, y_train),
        batch_size=args.batch_size,
        callbacks=keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            mode="min",
        ),
        verbose=0,
    )

    # Save model
    model_path = os.path.join(MODEL_DIR, f"model.keras")
    onnx_path = os.path.join(MODEL_DIR, f"model.onnx")

    print(f"Saving model to {model_path}...")
    Model.save_model(model_path)

    print(f"Exporting to ONNX: {onnx_path}...")
    Model.export_to_onnx(onnx_path)

    # Make predictions and create confusion matrix
    print("Generating predictions and confusion matrix...")
    predicted_indices = Model.predict_indices(X_val)
    true_indices = y_val["assignment_labels"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(
        true_indices[:, :, 0].argmax(axis=1),
        predicted_indices[:, :, 0].argmax(axis=1),
        normalize="true",
        ax=ax,
    )

    confusion_matrix_path = os.path.join(PLOTS_DIR, "confusion_matrix_lepton.pdf")
    plt.savefig(confusion_matrix_path)
    print(f"Confusion matrix saved to {confusion_matrix_path}")

    # Save training history and model info
    history_path = os.path.join(MODEL_DIR, f"history.npz")
    np.savez(history_path, trainable_params=trainable_params, **history.history)
    print(f"Training history saved to {history_path}")

    # Save model configuration summary
    CONFIG_PATH = os.path.join(MODEL_DIR, f"config.txt")
    with open(CONFIG_PATH, "w") as f:
        f.write(f"Model Configuration:\n")
        f.write(f"==================\n")
        f.write(f"Hidden Dimension: {args.hidden_dim}\n")
        f.write(f"Number of Layers: {args.num_layers}\n")
        f.write(f"Dropout Rate: {args.dropout_rate}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
    print(f"Model config saved to {CONFIG_PATH}")

    # Print final metrics
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Trainable Parameters: {trainable_params:,}")

    if "val_loss" in history.history:
        best_val_loss = min(history.history["val_loss"])
        print(f"Best validation loss: {best_val_loss:.6f}")

    if "val_assignment_accuracy" in history.history or "val_acc" in history.history:
        metric_key = (
            "val_assignment_accuracy"
            if "val_assignment_accuracy" in history.history
            else "val_acc"
        )
        best_val_acc = max(history.history[metric_key])
        print(f"Best validation accuracy: {best_val_acc:.6f}")

    print("=" * 60)

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
