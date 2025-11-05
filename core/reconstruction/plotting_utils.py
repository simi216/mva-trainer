"""Plotting utilities for evaluation."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, List

from .evaluator_base import (
    PlotConfig,
    BinningUtility,
    BootstrapCalculator,
    FeatureExtractor,
)


class AccuracyPlotter:
    """Handles plotting of accuracy metrics."""

    @staticmethod
    def plot_overall_accuracies(
        reconstructor_names: List[str],
        accuracies: List[Tuple[float, float, float]],
        config: PlotConfig,
    ):
        """
        Plot overall accuracies with error bars.

        Args:
            reconstructor_names: List of reconstructor names
            accuracies: List of (mean, lower, upper) tuples
            config: Plot configuration

        Returns:
            Tuple of (figure, axis)
        """
        names = reconstructor_names
        means = [acc[0] for acc in accuracies]
        lowers = [acc[1] for acc in accuracies]
        uppers = [acc[2] for acc in accuracies]

        errors_lower = [(mean - lower) for mean, lower in zip(means, lowers)]
        errors_upper = [(upper - mean) for mean, upper in zip(means, uppers)]

        fig, ax = plt.subplots(figsize=config.figsize)
        x_pos = np.arange(len(names))

        ax.bar(
            x_pos,
            means,
            yerr=[errors_lower, errors_upper],
            capsize=5,
            alpha=0.7,
            ecolor="black",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_title(
            f"Accuracy of Jet Reconstructors ({config.confidence*100:.0f}% CI)"
        )
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=config.alpha)
        fig.tight_layout()

        return fig, ax

    @staticmethod
    def plot_binned_accuracy(
        bin_centers: np.ndarray,
        binned_accuracies: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        reconstructor_names: List[str],
        bin_counts: np.ndarray,
        bins: np.ndarray,
        feature_label: str,
        config: PlotConfig,
        show_combinatoric: bool = True,
        combinatoric_accuracy: Optional[np.ndarray] = None,
    ):
        """Plot binned accuracy vs. a feature."""
        fig, ax = plt.subplots(figsize=config.figsize)
        color_map = plt.get_cmap("tab10")

        # Plot combinatoric baseline if requested
        if show_combinatoric and combinatoric_accuracy is not None:
            ax.plot(
                bin_centers,
                combinatoric_accuracy,
                label="Random Assignment",
                color="black",
                linestyle="--",
            )

        # Plot each reconstructor
        for index, (name, (mean_acc, lower, upper)) in enumerate(
            zip(reconstructor_names, binned_accuracies)
        ):
            if config.show_errorbar:
                errors_lower = mean_acc - lower
                errors_upper = upper - mean_acc
                ax.errorbar(
                    bin_centers,
                    mean_acc,
                    yerr=[errors_lower, errors_upper],
                    fmt="x",
                    label=name,
                    color=color_map(index),
                    linestyle="None",
                )
            else:
                ax.plot(
                    bin_centers,
                    mean_acc,
                    label=name,
                    color=color_map(index),
                )

        # Configure main axes
        ax.set_xlabel(feature_label)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.set_xlim(bins[0], bins[-1])
        ax.grid(alpha=config.alpha)
        ax.legend(loc="best")

        # Add event count histogram
        AccuracyPlotter._add_count_histogram(ax, bin_centers, bin_counts, bins)

        # Set title
        title = f"Accuracy per Bin vs {feature_label}"
        if config.show_errorbar:
            title += f" ({config.confidence*100:.0f}% CI)"
        ax.set_title(title)

        fig.tight_layout()
        return fig, ax

    @staticmethod
    def _add_count_histogram(ax, bin_centers, bin_counts, bins):
        """Add event count histogram to plot."""
        ax_twin = ax.twinx()
        ax_twin.bar(
            bin_centers,
            bin_counts,
            width=(bins[1] - bins[0]),
            alpha=0.2,
            color="red",
            label="Event Count",
        )
        ax_twin.set_ylabel("Event Count", color="red")
        ax_twin.tick_params(axis="y", labelcolor="red")

    @staticmethod
    def plot_feature_assignment_success(
        bin_centers: np.ndarray,
        correct_hist: np.ndarray,
        incorrect_hist: np.ndarray,
        feature_label: str,
        figsize: Tuple[int, int] = (8, 6),
    ):
        """Plot histogram of correct vs incorrect assignments per feature bin."""
        fig, ax = plt.subplots(figsize=figsize)

        width = bin_centers[1] - bin_centers[0]

        ax.bar(
            bin_centers,
            correct_hist,
            width=width,
            label="Correct Assignments",
            color="green",
            alpha=0.7,
        )
        ax.bar(
            bin_centers,
            incorrect_hist,
            width=width,
            label="Incorrect Assignments",
            color="red",
            alpha=0.7,
        )

        ax.set_xlabel(feature_label)
        ax.set_ylabel("a. u.")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

        fig.tight_layout()
        return fig, ax


class ConfusionMatrixPlotter:
    """Handles plotting of confusion matrices."""

    @staticmethod
    def plot_confusion_matrices(
        true_labels: np.ndarray,
        predictions_list: List[np.ndarray],
        reconstructor_names: List[str],
        normalize: bool = True,
        figsize_per_plot: Tuple[int, int] = (5, 5),
    ):
        """
        Plot confusion matrices for all reconstructors.

        Args:
            true_labels: True assignment labels
            predictions_list: List of prediction arrays
            reconstructor_names: List of reconstructor names
            normalize: Whether to normalize the confusion matrix
            figsize_per_plot: Size of each subplot

        Returns:
            Tuple of (figure, axes)
        """
        from sklearn.metrics import confusion_matrix

        n_reconstructors = len(reconstructor_names)
        rows = int(np.ceil(np.sqrt(n_reconstructors)))
        cols = int(np.ceil(n_reconstructors / rows))

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(figsize_per_plot[0] * cols, figsize_per_plot[1] * rows),
        )
        axes = axes.flatten() if n_reconstructors > 1 else [axes]

        true_indices = np.argmax(true_labels, axis=-2).flatten()

        for i, (name, predictions) in enumerate(
            zip(reconstructor_names, predictions_list)
        ):
            predicted_indices = np.argmax(predictions, axis=-2).flatten()

            cm = confusion_matrix(
                true_indices,
                predicted_indices,
                normalize="true" if normalize else None,
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt=".2f" if normalize else "d",
                ax=axes[i],
                cmap="Blues",
                cbar_kws={"label": "Normalized Count" if normalize else "Count"},
            )
            axes[i].set_title(f"Confusion Matrix: {name}")
            axes[i].set_xlabel("Predicted Label")
            axes[i].set_ylabel("True Label")

        # Remove unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        return fig, axes[: i + 1]


class ComplementarityPlotter:
    """Handles plotting of complementarity metrics."""

    @staticmethod
    def plot_complementarity_matrix(
        complementarity_matrix: np.ndarray,
        reconstructor_names: List[str],
        figsize: Tuple[int, int] = (8, 6),
    ):
        """Plot complementarity matrix between reconstructors."""
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            complementarity_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=reconstructor_names,
            yticklabels=reconstructor_names,
            cmap="viridis",
            cbar_kws={"label": "Complementarity"},
            ax=ax,
        )
        ax.set_title("Complementarity Matrix between Reconstructors")
        ax.set_xlabel("")
        ax.set_ylabel("")

        fig.tight_layout()
        return fig, ax

    @staticmethod
    def plot_binned_complementarity(
        bin_centers: np.ndarray,
        binned_complementarity: Tuple[np.ndarray, np.ndarray, np.ndarray],
        bin_counts: np.ndarray,
        bins: np.ndarray,
        feature_label: str,
        config: PlotConfig,
    ):
        """Plot binned complementarity vs. a feature."""
        fig, ax = plt.subplots(figsize=config.figsize)

        mean_comp, lower, upper = binned_complementarity

        if config.show_errorbar:
            errors_lower = mean_comp - lower
            errors_upper = upper - mean_comp
            ax.errorbar(
                bin_centers,
                mean_comp,
                yerr=[errors_lower, errors_upper],
                fmt="x",
                label="Complementarity",
                color="blue",
                linestyle="None",
            )
        else:
            ax.plot(
                bin_centers,
                mean_comp,
                label="Complementarity",
                color="blue",
            )

        # Configure axes
        ax.set_xlabel(feature_label)
        ax.set_ylabel("Complementarity")
        ax.set_ylim(0, 1)
        ax.set_xlim(bins[0], bins[-1])
        ax.grid(alpha=config.alpha)
        ax.legend(loc="best")

        # Add event count histogram
        ax_twin = ax.twinx()
        ax_twin.bar(
            bin_centers,
            bin_counts,
            width=(bins[1] - bins[0]),
            alpha=0.2,
            color="red",
            label="Event Count",
        )
        ax_twin.set_ylabel("Event Count", color="red")
        ax_twin.tick_params(axis="y", labelcolor="red")

        # Set title
        title = f"Complementarity per Bin vs {feature_label}"
        if config.show_errorbar:
            title += f" ({config.confidence*100:.0f}% CI)"
        ax.set_title(title)

        fig.tight_layout()
        return fig, ax


class ResolutionPlotter:
    """Handles plotting of mass resolution metrics."""

    @staticmethod
    def plot_binned_resolution(
        bin_centers: np.ndarray,
        binned_resolutions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        reconstructor_names: List[str],
        bin_counts: np.ndarray,
        bins: np.ndarray,
        feature_label: str,
        resolution_label: str,
        config: PlotConfig,
    ):
        """Plot binned resolution vs. a feature."""
        fig, ax = plt.subplots(figsize=config.figsize)
        color_map = plt.get_cmap("tab10")

        # Plot each reconstructor
        for index, (name, (mean_res, lower, upper)) in enumerate(
            zip(reconstructor_names, binned_resolutions)
        ):
            if config.show_errorbar:
                errors_lower = mean_res - lower
                errors_upper = upper - mean_res
                ax.errorbar(
                    bin_centers,
                    mean_res,
                    yerr=[errors_lower, errors_upper],
                    fmt="x",
                    label=name,
                    color=color_map(index),
                    linestyle="None",
                )
            else:
                ax.plot(
                    bin_centers,
                    mean_res,
                    label=name,
                    color=color_map(index),
                )

        # Configure axes
        ax.set_xlabel(feature_label)
        ax.set_ylabel(resolution_label)
        ax.set_ylim(0, None)
        ax.set_xlim(bins[0], bins[-1])
        ax.grid(alpha=config.alpha)
        ax.legend(loc="best")

        # Add event count histogram
        ax_twin = ax.twinx()
        ax_twin.bar(
            bin_centers,
            bin_counts,
            width=(bins[1] - bins[0]),
            alpha=0.2,
            color="red",
            label="Event Count",
        )
        ax_twin.set_ylabel("Event Count", color="red")
        ax_twin.tick_params(axis="y", labelcolor="red")

        # Set title
        title = f"{resolution_label} per Bin vs {feature_label}"
        if config.show_errorbar:
            title += f" ({config.confidence*100:.0f}% CI)"
        ax.set_title(title)

        fig.tight_layout()
        return fig, ax
