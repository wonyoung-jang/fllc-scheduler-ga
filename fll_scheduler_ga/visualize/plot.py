"""Methods to create plots."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from ..genetic.schedule import Population

if TYPE_CHECKING:
    from ..genetic.ga import GA

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Plot:
    """A class for creating and managing plots related to the GA run."""

    ga_instance: "GA"

    def plot_fitness(
        self,
        title: str = "Generation vs. Fitness",
        xlabel: str = "Generation",
        ylabel: str = "Fitness",
        save_dir: str | Path | None = None,
    ) -> None | plt.Figure:
        """Create figure that summarizes how the average fitness of the first Pareto front evolved by generation.

        Args:
            title: Figure title.
            xlabel: Label on the X-axis.
            ylabel: Label on the Y-axis.
            save_dir: Directory to save the figure. If None, the plot is shown.

        """
        if not self.ga_instance.fitness_history:
            logger.error("Cannot plot fitness. No generation history was recorded.")
            return None

        objective_names = list(self.ga_instance.fitness.soft_constraints)
        history_df = pd.DataFrame(self.ga_instance.fitness_history, columns=objective_names)

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 7))

        history_df.plot(kind="line", ax=ax, linewidth=2.5, alpha=0.8)

        ax.set_title(title, fontsize=18, pad=20)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(title="Objectives", fontsize=10)
        ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)

        plt.tight_layout()

        if save_dir:
            try:
                plt.savefig(save_dir, dpi=300, bbox_inches="tight")
                logger.info("Fitness plot saved to %s", save_dir)
            except OSError:
                logger.exception("Failed to save fitness plot.")
        else:
            plt.show()

        plt.close(fig)
        return fig

    def plot_pareto_front(
        self,
        title: str = "Pareto Front: Optimal Schedule Trade-offs",
        save_dir: str | Path | None = None,
    ) -> None:
        """Generate and saves a parallel coordinates plot of the final Pareto front."""
        if not self.ga_instance.pareto_front:
            logger.warning("Cannot plot an empty Pareto front.")
            return

        objective_names = list(self.ga_instance.fitness.soft_constraints)
        num_objectives = len(objective_names)

        self._plot_parallel_coordinates(self.ga_instance.pareto_front, objective_names, title, save_dir)

        if 2 <= num_objectives <= 3:
            scatter_title = "2D Pareto Front" if num_objectives == 2 else "3D Pareto Front"
            scatter_save_dir = Path(save_dir).with_name(f"{Path(save_dir).stem}_scatter.png") if save_dir else None
            self._plot_pareto_scatter(self.ga_instance.pareto_front, objective_names, scatter_title, scatter_save_dir)

    def _plot_parallel_coordinates(
        self,
        pareto_front: Population,
        objective_names: list[str],
        title: str,
        save_dir: str | None,
    ) -> None:
        """Create the parallel coordinates plot."""
        data = [p.fitness for p in pareto_front]
        dataframe = pd.DataFrame(data, columns=objective_names)

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(1, 1, figsize=(12, 7))

        pd.plotting.parallel_coordinates(
            dataframe,
            class_column=objective_names[0],
            colormap="viridis",
            ax=axes,
            linewidth=1.5,
            alpha=0.7,
        )

        axes.set_title(title, fontsize=18, pad=20)
        axes.set_ylabel("Objective Score", fontsize=12)
        axes.set_xlabel("Objectives", fontsize=12)
        plt.xticks(rotation=15, ha="right")
        axes.get_legend().remove()

        norm = plt.Normalize(dataframe[objective_names[0]].min(), dataframe[objective_names[0]].max())
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes)
        cbar.set_label(f"Score for: {objective_names[0]}", fontsize=12)

        plt.tight_layout()

        if save_dir:
            try:
                plt.savefig(save_dir, dpi=300, bbox_inches="tight")
                logger.info("Pareto front plot saved to %s", save_dir)
            except OSError:
                logger.exception("Failed to save Pareto front plot")
        else:
            plt.show()

        plt.close(fig)

    def _plot_pareto_scatter(
        self,
        pareto_front: Population,
        objective_names: list[str],
        title: str,
        save_dir: str | None,
    ) -> None:
        """Create a 2D or 3D scatter plot of the Pareto front."""
        logger.info("Generating %d D scatter plot for Pareto front.", len(objective_names))

        fitness_data = [p.fitness for p in pareto_front]
        dataframe = pd.DataFrame(fitness_data, columns=objective_names)

        plt.style.use("seaborn-v0_8-whitegrid")

        if len(objective_names) == 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            x_obj, y_obj = objective_names
            crowding_distances = [p.crowding_distance for p in pareto_front]
            sc = ax.scatter(dataframe[x_obj], dataframe[y_obj], c=crowding_distances, cmap="viridis", s=60, alpha=0.8)
            ax.set_xlabel(x_obj, fontsize=12)
            ax.set_ylabel(y_obj, fontsize=12)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Crowding Distance", fontsize=12)
        elif len(objective_names) == 3:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")
            x_obj, y_obj, z_obj = objective_names

            crowding_distances = [p.crowding_distance for p in pareto_front]
            sc = ax.scatter(
                dataframe[x_obj], dataframe[y_obj], dataframe[z_obj], c=crowding_distances, cmap="viridis", s=60
            )

            ax.set_xlabel(x_obj, fontsize=10)
            ax.set_ylabel(y_obj, fontsize=10)
            ax.set_zlabel(z_obj, fontsize=10)
            cbar = plt.colorbar(sc, ax=ax, pad=0.1)
            cbar.set_label("Crowding Distance", fontsize=12)
        else:
            logger.info("Scatter plot is only supported for 2 or 3 objectives.")
            return

        ax.set_title(title, fontsize=18, pad=20)
        plt.tight_layout()

        if save_dir:
            try:
                plt.savefig(save_dir, dpi=300)
            except OSError:
                logger.exception("Failed to save Pareto scatter plot")
        else:
            plt.show()

        plt.close(fig)
