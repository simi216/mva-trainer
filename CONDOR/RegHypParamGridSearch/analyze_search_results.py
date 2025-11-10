import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from glob import glob
import argparse


def parse_model_name(model_name):
    """Extract hyperparameters from model name."""
    # Try regression model pattern first: Regression{arch}_d{dim}_cl{central_layers}_rl{regression_layers}
    regression_pattern = r"_d(\d+)_cl(\d+)_rl(\d+)"
    match = re.search(regression_pattern, model_name)
    if match:
        return {
            "hidden_dim": int(match.group(1)),
            "num_central_layers": int(match.group(2)),
            "num_regression_layers": int(match.group(3)),
            "model_type": "regression"
        }
    
    # Try assignment model pattern: {arch}_d{dim}_l{layers}_h{heads}
    assignment_pattern = r"_d(\d+)_l(\d+)_h(\d+)"
    match = re.search(assignment_pattern, model_name)
    if match:
        return {
            "hidden_dim": int(match.group(1)),
            "num_layers": int(match.group(2)),
            "num_heads": int(match.group(3)),
            "model_type": "assignment"
        }
    
    return None


def collect_results(models_dir, model_type="Raw_Transformer_Assignment"):
    """Collect training results from all model directories."""
    results = []
    # Find all model directories
    model_dirs = glob(os.path.join(models_dir, f"{model_type}_*"))

    is_regression = "Regression" in model_type

    print(f"Looking for {model_type} models in {models_dir}...")
    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        hyperparams = parse_model_name(model_name)
        print(f"Processing model: {model_name}")
        if hyperparams is None:
            continue

        # Load training history
        history_file = os.path.join(model_dir, f"{model_name}_history.npz")

        if not os.path.exists(history_file):
            print(f"Warning: History file not found for {model_name}")
            continue

        try:
            history = np.load(history_file)

            # Get best validation metrics
            val_loss = history["val_loss"]
            best_val_loss = np.min(val_loss)
            best_epoch = np.argmin(val_loss)

            # Get trainable parameters
            trainable_params = (
                int(history["trainable_params"])
                if "trainable_params" in history
                else None
            )

            # Try to get validation accuracy
            if "val_assignment_accuracy" in history:
                val_acc = history["val_assignment_accuracy"]
                best_val_acc = val_acc[best_epoch]
            elif "val_acc" in history:
                val_acc = history["val_acc"]
                best_val_acc = val_acc[best_epoch]
            else:
                best_val_acc = None

            # Build base result dictionary
            result = {
                "model_name": model_name,
                "hidden_dim": hyperparams["hidden_dim"],
                "trainable_params": trainable_params,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "total_epochs": len(val_loss),
            }

            # Add model-specific hyperparameters
            if hyperparams["model_type"] == "regression":
                result["num_central_layers"] = hyperparams["num_central_layers"]
                result["num_regression_layers"] = hyperparams["num_regression_layers"]
                
                # Get regression-specific metrics
                if "val_regression_relative_error" in history:
                    val_reg_err = history["val_regression_relative_error"]
                    result["best_val_regression_error"] = val_reg_err[best_epoch]
                else:
                    result["best_val_regression_error"] = None
                
                # Try to get saved best relative deviation
                if "best_val_rel_dev" in history:
                    result["best_val_rel_dev"] = float(history["best_val_rel_dev"])
                else:
                    result["best_val_rel_dev"] = None
                    
            else:  # assignment model
                result["num_layers"] = hyperparams["num_layers"]
                result["num_heads"] = hyperparams["num_heads"]

            results.append(result)

        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            continue

    return pd.DataFrame(results)


def plot_grid_search_results(df, output_dir):
    """Create visualization of grid search results."""
    os.makedirs(output_dir, exist_ok=True)

    # Determine if this is regression or assignment model
    is_regression = "num_central_layers" in df.columns
    
    if is_regression:
        layer_col = "num_central_layers"
        layer_label = "Number of Central Layers"
    else:
        layer_col = "num_layers"
        layer_label = "Number of Layers"

    # 1. Heatmap of validation loss
    plt.figure(figsize=(10, 8))
    pivot_loss = df.pivot_table(
        values="best_val_loss", index=layer_col, columns="hidden_dim", aggfunc="mean"
    )
    sns.heatmap(pivot_loss, annot=True, fmt=".4f", cmap="viridis_r")
    plt.title("Best Validation Loss by Hyperparameters")
    plt.ylabel(layer_label)
    plt.xlabel("Hidden Dimension")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grid_search_loss_heatmap.png"), dpi=150)
    plt.close()

    # 2. Heatmap of validation accuracy (if available)
    if df["best_val_acc"].notna().any():
        plt.figure(figsize=(10, 8))
        pivot_acc = df.pivot_table(
            values="best_val_acc",
            index=layer_col,
            columns="hidden_dim",
            aggfunc="mean",
        )
        sns.heatmap(pivot_acc, annot=True, fmt=".4f", cmap="viridis")
        plt.title("Best Validation Accuracy by Hyperparameters")
        plt.ylabel(layer_label)
        plt.xlabel("Hidden Dimension")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "grid_search_accuracy_heatmap.png"), dpi=150
        )
        plt.close()

    # 2b. Heatmap of regression relative error (if available)
    if is_regression and "best_val_regression_error" in df.columns and df["best_val_regression_error"].notna().any():
        plt.figure(figsize=(10, 8))
        pivot_reg_err = df.pivot_table(
            values="best_val_regression_error",
            index=layer_col,
            columns="hidden_dim",
            aggfunc="mean",
        )
        sns.heatmap(pivot_reg_err, annot=True, fmt=".4f", cmap="viridis_r")
        plt.title("Best Validation Regression Relative Error by Hyperparameters")
        plt.ylabel(layer_label)
        plt.xlabel("Hidden Dimension")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "grid_search_regression_error_heatmap.png"), dpi=150
        )
        plt.close()

    # 2c. Heatmap of regression relative deviation (if available)
    if is_regression and "best_val_rel_dev" in df.columns and df["best_val_rel_dev"].notna().any():
        plt.figure(figsize=(10, 8))
        pivot_rel_dev = df.pivot_table(
            values="best_val_rel_dev",
            index=layer_col,
            columns="hidden_dim",
            aggfunc="mean",
        )
        sns.heatmap(pivot_rel_dev, annot=True, fmt=".4f", cmap="viridis_r")
        plt.title("Best Validation Relative Deviation by Hyperparameters")
        plt.ylabel(layer_label)
        plt.xlabel("Hidden Dimension")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "grid_search_relative_deviation_heatmap.png"), dpi=150
        )
        plt.close()

    # 3. Heatmap of trainable parameters (if available)
    if df["trainable_params"].notna().any():
        plt.figure(figsize=(10, 8))
        pivot_params = df.pivot_table(
            values="trainable_params",
            index=layer_col,
            columns="hidden_dim",
            aggfunc="mean",
        )
        sns.heatmap(pivot_params, annot=True, fmt=".0f", cmap="YlOrRd")
        plt.title("Trainable Parameters by Hyperparameters")
        plt.ylabel(layer_label)
        plt.xlabel("Hidden Dimension")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "grid_search_params_heatmap.png"), dpi=150)
        plt.close()

    # 4. Bar plot of top models
    df_sorted = df.sort_values("best_val_loss")
    top_n = min(10, len(df))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Top models by loss
    top_models = df_sorted.head(top_n)
    if is_regression:
        model_labels = [
            f"d{row['hidden_dim']}_cl{row['num_central_layers']}_rl{row['num_regression_layers']}" 
            for _, row in top_models.iterrows()
        ]
    else:
        model_labels = [
            f"d{row['hidden_dim']}_l{row['num_layers']}_h{row['num_heads']}" 
            for _, row in top_models.iterrows()
        ]

    ax1.barh(range(top_n), top_models["best_val_loss"])
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(model_labels)
    ax1.set_xlabel("Best Validation Loss")
    ax1.set_title(f"Top {top_n} Models by Validation Loss")
    ax1.invert_yaxis()

    # Training epochs
    ax2.barh(range(top_n), top_models["best_epoch"])
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(model_labels)
    ax2.set_xlabel("Best Epoch")
    ax2.set_title("Convergence Speed")
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grid_search_top_models.png"), dpi=150)
    plt.close()

    # 5. Scatter plot: hidden_dim vs num_layers colored by loss
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        df["hidden_dim"],
        df[layer_col],
        c=df["best_val_loss"],
        s=200,
        cmap="viridis_r",
        alpha=0.7,
        edgecolors="black",
    )
    plt.colorbar(scatter, label="Best Validation Loss")
    plt.xlabel("Hidden Dimension")
    plt.ylabel(layer_label)
    plt.title("Grid Search Results: Loss Landscape")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grid_search_scatter.png"), dpi=150)
    plt.close()

    # 6. Efficiency plot: Loss vs Parameters
    if df["trainable_params"].notna().any():
        # Create subplots based on available metrics
        n_plots = 2 if df["best_val_acc"].notna().any() else 1
        if is_regression and "best_val_regression_error" in df.columns and df["best_val_regression_error"].notna().any():
            n_plots += 1
        
        fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6))
        if n_plots == 1:
            axes = [axes]

        plot_idx = 0
        
        # Plot 1: Loss vs Parameters
        scatter1 = axes[plot_idx].scatter(
            df["trainable_params"] / 1e6,  # Convert to millions
            df["best_val_loss"],
            c=df["hidden_dim"],
            s=100,
            cmap="viridis",
            alpha=0.7,
            edgecolors="black",
        )
        axes[plot_idx].set_xlabel("Trainable Parameters (Millions)")
        axes[plot_idx].set_ylabel("Best Validation Loss")
        axes[plot_idx].set_title("Model Efficiency: Loss vs Model Size")
        axes[plot_idx].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[plot_idx], label="Hidden Dimension")
        plot_idx += 1

        # Plot 2: Accuracy vs Parameters (if available)
        if df["best_val_acc"].notna().any():
            scatter2 = axes[plot_idx].scatter(
                df["trainable_params"] / 1e6,
                df["best_val_acc"],
                c=df["hidden_dim"],
                s=100,
                cmap="viridis",
                alpha=0.7,
                edgecolors="black",
            )
            axes[plot_idx].set_xlabel("Trainable Parameters (Millions)")
            axes[plot_idx].set_ylabel("Best Validation Accuracy")
            axes[plot_idx].set_title("Model Efficiency: Accuracy vs Model Size")
            axes[plot_idx].grid(True, alpha=0.3)
            plt.colorbar(scatter2, ax=axes[plot_idx], label="Hidden Dimension")
            plot_idx += 1

        # Plot 3: Regression Error vs Parameters (if available)
        if is_regression and "best_val_regression_error" in df.columns and df["best_val_regression_error"].notna().any():
            scatter3 = axes[plot_idx].scatter(
                df["trainable_params"] / 1e6,
                df["best_val_regression_error"],
                c=df["hidden_dim"],
                s=100,
                cmap="viridis",
                alpha=0.7,
                edgecolors="black",
            )
            axes[plot_idx].set_xlabel("Trainable Parameters (Millions)")
            axes[plot_idx].set_ylabel("Best Validation Regression Error")
            axes[plot_idx].set_title("Model Efficiency: Regression Error vs Model Size")
            axes[plot_idx].grid(True, alpha=0.3)
            plt.colorbar(scatter3, ax=axes[plot_idx], label="Hidden Dimension")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "grid_search_efficiency.png"), dpi=150)
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze Hyperparameter Grid Search Results"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="Raw_Transformer_Assignment",
        help="Type of model to analyze (e.g., Raw_Transformer_Assignment, RegressionFeatureConcatTransformer)",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/afs/desy.de/user/a/aulich/mva-trainer/models",
        help="Directory containing trained models (default: /afs/desy.de/user/a/aulich/mva-trainer/models)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_type = args.model_type
    root_dir = args.root_dir

    # Configuration
    ROOT_DIR = root_dir
    MODELS_DIR = ROOT_DIR
    OUTPUT_DIR = os.path.join(ROOT_DIR, "plots", "grid_search_analysis", model_type)

    print(f"Collecting results from grid search (directory {ROOT_DIR})...")
    results_df = collect_results(MODELS_DIR, model_type=model_type)

    if len(results_df) == 0:
        print("No results found! Check your models directory.")
        return

    print(f"\nFound {len(results_df)} completed training runs")

    # Save results to CSV
    csv_path = os.path.join(OUTPUT_DIR, "grid_search_results.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Print summary statistics
    is_regression = "num_central_layers" in results_df.columns
    
    print("\n" + "=" * 60)
    print("GRID SEARCH SUMMARY")
    print("=" * 60)
    print("\nBest model by validation loss:")
    best_model = results_df.loc[results_df["best_val_loss"].idxmin()]
    print(f"  Model: {best_model['model_name']}")
    print(f"  Hidden dim: {best_model['hidden_dim']}")
    if is_regression:
        print(f"  Central layers: {best_model['num_central_layers']}")
        print(f"  Regression layers: {best_model['num_regression_layers']}")
    else:
        print(f"  Num layers: {best_model['num_layers']}")
        print(f"  Num heads: {best_model['num_heads']}")
    print(f"  Val loss: {best_model['best_val_loss']:.6f}")
    if best_model["best_val_acc"] is not None:
        print(f"  Val accuracy: {best_model['best_val_acc']:.6f}")
    if is_regression and "best_val_regression_error" in best_model and best_model["best_val_regression_error"] is not None:
        print(f"  Val regression error: {best_model['best_val_regression_error']:.6f}")
    if is_regression and "best_val_rel_dev" in best_model and best_model["best_val_rel_dev"] is not None:
        print(f"  Val relative deviation: {best_model['best_val_rel_dev']:.6f}")
    if best_model["trainable_params"] is not None:
        print(f"  Trainable params: {best_model['trainable_params']:,}")
    print(f"  Best epoch: {best_model['best_epoch']}")

    print("\nTop 5 models:")
    display_cols = ["hidden_dim", "trainable_params", "best_val_loss", "best_val_acc", "best_epoch"]
    if is_regression:
        display_cols.insert(1, "num_central_layers")
        display_cols.insert(2, "num_regression_layers")
        if "best_val_regression_error" in results_df.columns:
            display_cols.insert(-1, "best_val_regression_error")
        if "best_val_rel_dev" in results_df.columns:
            display_cols.insert(-1, "best_val_rel_dev")
    else:
        display_cols.insert(1, "num_layers")
        display_cols.insert(2, "num_heads")
    # Only include columns that exist and have data
    display_cols = [col for col in display_cols if col in results_df.columns]
    print(results_df.nsmallest(5, "best_val_loss")[display_cols].to_string(index=False))

    # Print parameter efficiency analysis
    if results_df["trainable_params"].notna().any():
        print("\n" + "=" * 60)
        print("MODEL EFFICIENCY ANALYSIS")
        print("=" * 60)
        print("\nSmallest model:")
        smallest = results_df.loc[results_df["trainable_params"].idxmin()]
        if is_regression:
            print(f"  Config: d{smallest['hidden_dim']}_cl{smallest['num_central_layers']}_rl{smallest['num_regression_layers']}")
        else:
            print(f"  Config: d{smallest['hidden_dim']}_l{smallest['num_layers']}_h{smallest['num_heads']}")
        print(f"  Parameters: {smallest['trainable_params']:,}")
        print(f"  Val loss: {smallest['best_val_loss']:.6f}")

        print("\nLargest model:")
        largest = results_df.loc[results_df["trainable_params"].idxmax()]
        if is_regression:
            print(f"  Config: d{largest['hidden_dim']}_cl{largest['num_central_layers']}_rl{largest['num_regression_layers']}")
        else:
            print(f"  Config: d{largest['hidden_dim']}_l{largest['num_layers']}_h{largest['num_heads']}")
        print(f"  Parameters: {largest['trainable_params']:,}")
        print(f"  Val loss: {largest['best_val_loss']:.6f}")

        print("\nBest efficiency (lowest loss per million parameters):")
        results_df["loss_per_mparam"] = results_df["best_val_loss"] / (
            results_df["trainable_params"] / 1e6
        )
        most_efficient = results_df.loc[results_df["loss_per_mparam"].idxmin()]
        if is_regression:
            print(f"  Config: d{most_efficient['hidden_dim']}_cl{most_efficient['num_central_layers']}_rl{most_efficient['num_regression_layers']}")
        else:
            print(f"  Config: d{most_efficient['hidden_dim']}_l{most_efficient['num_layers']}_h{most_efficient['num_heads']}")
        print(f"  Parameters: {most_efficient['trainable_params']:,}")
        print(f"  Val loss: {most_efficient['best_val_loss']:.6f}")
        print(f"  Efficiency: {most_efficient['loss_per_mparam']:.6f} loss/Mparam")
        
    # Print regression-specific analysis
    if is_regression:
        print("\n" + "=" * 60)
        print("REGRESSION PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        if "best_val_regression_error" in results_df.columns and results_df["best_val_regression_error"].notna().any():
            print("\nBest model by regression error:")
            best_reg = results_df.loc[results_df["best_val_regression_error"].idxmin()]
            print(f"  Config: d{best_reg['hidden_dim']}_cl{best_reg['num_central_layers']}_rl{best_reg['num_regression_layers']}")
            print(f"  Regression error: {best_reg['best_val_regression_error']:.6f}")
            print(f"  Val loss: {best_reg['best_val_loss']:.6f}")
            print(f"  Val accuracy: {best_reg['best_val_acc']:.6f}" if best_reg['best_val_acc'] is not None else "")
            
        if "best_val_rel_dev" in results_df.columns and results_df["best_val_rel_dev"].notna().any():
            print("\nBest model by relative deviation:")
            best_dev = results_df.loc[results_df["best_val_rel_dev"].idxmin()]
            print(f"  Config: d{best_dev['hidden_dim']}_cl{best_dev['num_central_layers']}_rl{best_dev['num_regression_layers']}")
            print(f"  Relative deviation: {best_dev['best_val_rel_dev']:.6f}")
            print(f"  Val loss: {best_dev['best_val_loss']:.6f}")
            print(f"  Val accuracy: {best_dev['best_val_acc']:.6f}" if best_dev['best_val_acc'] is not None else "")

    # Create visualizations
    print("\nCreating visualizations...")
    plot_grid_search_results(results_df, OUTPUT_DIR)
    print(f"Plots saved to {OUTPUT_DIR}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
