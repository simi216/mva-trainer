import sys
import argparse
import os
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import yaml


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
        "--num_central_layers", type=int, required=True, help="Number of central transformer layers"
    )
    parser.add_argument(
        "--num_regression_layers", type=int, required=True, help="Number of regression transformer layers"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="FeatureConcatTransformer",
        help="Model architecture to use (default: FeatureConcatTransformer)",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="workspace_config.yaml",
        help="Path to data configuration YAML file",
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
    import core.regression_models as Models
    from core.DataLoader import (
        DataPreprocessor,
        DataConfig,
        LoadConfig,
        get_load_config_from_yaml,
    )
    import core

    # Load data configuration
    load_config = get_load_config_from_yaml(args.data_config)

    # Create model name with hyperparameters
    MODEL_NAME = (
        f"Regression{args.architecture}_d{args.hidden_dim}_cl{args.num_central_layers}_rl{args.num_regression_layers}"
    )

    # Setup directories
    PLOTS_DIR, MODEL_DIR = setup_directories(args.root_dir, MODEL_NAME)

    print(f"Starting training with hyperparameters:")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Central Transformer layers: {args.num_central_layers}")
    print(f"  Regression Layers {args.num_regression_layers}")
    print(f"  Dropout rate: {args.dropout_rate}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Model name: {MODEL_NAME}")

    # Configure data
    with open(args.data_config, "r") as f:
        data_config_yaml = yaml.safe_load(f)

    # Load and preprocess data
    print("Loading data...")
    DataProcessor = DataPreprocessor(load_config)
    config = DataProcessor.load_data(
        data_config_yaml["data_path"][args.data_type],
        "reco",
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
            central_transformer_stack_size=args.num_central_layers,
            regression_transformer_stack_size=args.num_regression_layers,
            num_heads=8,
            dropout_rate=args.dropout_rate,
            input_as_four_vector=True,
        )
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")
    # Adapt normalization and compile
    print("Adapting normalization layers...")
    Model.adapt_normalization_layers(X_train)

    print("Compiling model...")
    Model.compile_model(
        loss={
            "assignment": core.utils.AssignmentLoss(lambda_excl=0),
            "regression": core.utils.RegressionLoss()
        },
        optimizer=keras.optimizers.AdamW(
            learning_rate=args.learning_rate, weight_decay=args.weight_decay
        ),
        metrics={
            "assignment": core.utils.AssignmentAccuracy(),
            "regression": core.utils.RelativeRegressionLoss()
        }
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
    model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.keras")
    onnx_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.onnx")

    print(f"Saving model to {model_path}...")
    Model.save_model(model_path)

    print(f"Exporting to ONNX: {onnx_path}...")
    Model.export_to_onnx(onnx_path)

    # Evaluate model
    print("Evaluating model...")
    import core.assignment_models.BaselineAssignmentMethods as BaselineMethods
    import core.reconstruction as Evaluation

    chi_square_true_nu = BaselineMethods.MassCombinatoricsAssigner(
        config,
        top_mass=173.5e3,
    )
    ground_truth_assigner = Evaluation.GroundTruthReconstructor(config, name = "Perfect Reconstructor")
    evaluator = Evaluation.ReconstructionEvaluator(
        [
            chi_square_true_nu,
            Model,
            ground_truth_assigner,
        ],
        X_val,
        y_val,
    )

    fig, ax = evaluator.plot_binned_accuracy(
        feature_data_type="non_training",
        feature_name="truth_ttbar_mass",
        fancy_feature_label=r"$m(t\overline{t})$ [GeV]",
        xlims=(340e3, 800e3),
        bins=10,
    )
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(tick/1e3)}" for tick in ticks])
    ax.set_xlim(340e3, 800e3)
    fig.savefig(os.path.join(PLOTS_DIR , "binned_accuracy_ttbar_mass.pdf"))

    fig, ax = evaluator.plot_binned_accuracy(
        feature_data_type="non_training",
        feature_name="N_jets",
        fancy_feature_label=r"$\# \text{jets}$",
        xlims=(2, config.max_jets + 1),
        bins= config.max_jets -1,
    )
    ax.set_xticks([i + 0.5 for i in range(2, config.max_jets+ 1)])
    ax.set_xticklabels([i for i in range(2, config.max_jets + 1)])
    fig.savefig(os.path.join(PLOTS_DIR , "binned_accuracy_N_jets.pdf"))

    # Plot neutrino component deviation histograms
    fig, axes = evaluator.plot_neutrino_component_deviations(
        bins=50,
        xlims=(0, 5),  # Optional: limit x-axis range
        figsize=(15, 10),
        component_labels=["$p_x$", "$p_y$", "$p_z$"]
    )
    fig.savefig(os.path.join(PLOTS_DIR, "neutrino_component_deviation.pdf"))

    # Plot overall relative deviation distribution
    fig, ax = evaluator.plot_overall_neutrino_deviation_distribution(
        bins=50,
        xlims=(0, 2),  # Optional: limit x-axis range
        figsize=(12, 6)
    )
    fig.savefig(os.path.join(PLOTS_DIR, "overall_neutrino_deviation.pdf"))

    # Compute best regression deviation metrics
    best_val_rel_dev = None
    best_val_rel_dev_epoch = None
    if "val_assignment_relative_deviation" in history.history:
        val_rel_devs = history.history["val_assignment_relative_deviation"]
        best_val_rel_dev = min(val_rel_devs)
        best_val_rel_dev_epoch = np.argmin(val_rel_devs)

    # Save training history and model info
    history_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_history.npz")
    save_dict = {
        "trainable_params": trainable_params,
        **history.history
    }
    if best_val_rel_dev is not None:
        save_dict["best_val_rel_dev"] = best_val_rel_dev
        save_dict["best_val_rel_dev_epoch"] = best_val_rel_dev_epoch
    
    np.savez(history_path, **save_dict)
    print(f"Training history saved to {history_path}")

    # Save model configuration summary
    config_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_config.txt")
    with open(config_path, "w") as f:
        f.write(f"Model Configuration:\n")
        f.write(f"==================\n")
        f.write(f"Hidden Dimension: {args.hidden_dim}\n")
        f.write(f"Central Transformer Layers: {args.num_central_layers}")
        f.write(f"Regression Transformer Layers: {args.num_regression_layers}")
        f.write(f"Dropout Rate: {args.dropout_rate}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
    print(f"Model config saved to {config_path}")

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

    if "val_assignment_relative_deviation" in history.history:
        best_val_rel_dev = min(history.history["val_assignment_relative_deviation"])
        print(f"Best validation relative deviation: {best_val_rel_dev:.6f}")

    print("=" * 60)

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
