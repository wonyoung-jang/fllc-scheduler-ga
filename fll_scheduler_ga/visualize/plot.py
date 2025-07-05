"""Methods to create plots."""

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..genetic.ga import GA
from ..genetic.schedule import Population

logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")


@dataclass(slots=True)
class Plot:
    """A class for creating and managing plots related to the GA run."""

    ga_instance: GA

    def plot_fitness(
        self,
        title: str = "Average Fitness over Generations",
        xlabel: str = "Generation",
        ylabel: str = "Average Fitnesses",
        save_dir: str | Path | None = None,
    ) -> plt.Figure | None:
        """Create figure that summarizes how the average fitness of the first Pareto front evolved by generation.

        Args:
            title: Figure title.
            xlabel: Label on the X-axis.
            ylabel: Label on the Y-axis.
            save_dir: Directory to save the figure. If None, the plot is shown.

        """
        if not (history := self.ga_instance.fitness_history):
            logger.error("Cannot plot fitness. No generation history was recorded.")
            return None
        history_df = pd.DataFrame(data=history, columns=self.ga_instance.fitness.soft_constraints)
        fig, ax = plt.subplots(figsize=(12, 7))
        history_df.plot(kind="line", ax=ax, linewidth=2.5, alpha=0.8)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend(title="Objectives", fontsize=10)
        return self._finalize(fig, save_dir, "fitness_plot.png")

    def plot_pareto_front(
        self, title: str = "Pareto Front: Trade-offs", save_dir: str | Path | None = None
    ) -> plt.Figure | None:
        """Generate and saves a parallel coordinates plot of the final Pareto front."""
        if not (front := self.ga_instance.pareto_front):
            logger.warning("Cannot plot an empty Pareto front.")
            return None

        objectives = self.ga_instance.fitness.soft_constraints
        fig = self._plot_parallel(front, objectives, title)
        self._finalize(fig, save_dir, "pareto_parallel.png")

        if len(objectives) in (2, 3):
            scatter_name = f"pareto_scatter_{len(objectives)}d.png"
            return self._plot_pareto_scatter(
                front, objectives, title=title, save_dir=Path(save_dir).with_name(scatter_name) if save_dir else None
            )
        return fig

    def _plot_parallel(self, front: Population, objectives: list[str], title: str) -> None:
        """Create the parallel coordinates plot."""
        data = [[p.fitness[i] for i, _ in enumerate(objectives)] + [p.crowding_distance] for p in front]
        objectives = [*objectives, "Crowding Distance"]
        dataframe = pd.DataFrame(data=data, columns=objectives)
        fig, ax = plt.subplots(figsize=(12, 7))
        pd.plotting.parallel_coordinates(
            frame=dataframe, class_column=objectives[-1], ax=ax, linewidth=1.5, alpha=0.7, colormap="viridis"
        )
        ax.set(title=title, xlabel="Objectives", ylabel="Score")
        plt.xticks(rotation=15, ha="right")
        ax.get_legend().remove()
        self._attach_colorbar(ax, dataframe[objectives[-1]], label="Crowding Distance")
        return fig

    def _plot_pareto_scatter(
        self, front: Population, objectives: list[str], title: str, save_dir: str | None
    ) -> plt.Figure | None:
        """Create a 2D or 3D scatter plot of the Pareto front."""
        dataframe = pd.DataFrame(data=[p.fitness for p in front], columns=objectives)
        distances = [p.crowding_distance for p in front]

        if len(objectives) == 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            x_obj, y_obj = objectives
            ax.scatter(dataframe[x_obj], dataframe[y_obj], c=distances, cmap="viridis", s=60, alpha=0.8)
            ax.set(title=title, xlabel=x_obj, ylabel=y_obj)
            self._attach_colorbar(ax, distances, label="Crowding Distance")
        elif len(objectives) == 3:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")
            x_obj, y_obj, z_obj = objectives
            ax.scatter(dataframe[x_obj], dataframe[y_obj], dataframe[z_obj], c=distances, cmap="viridis", s=60)
            ax.set(title=title, xlabel=x_obj, ylabel=y_obj, zlabel=z_obj)
            self._attach_colorbar(ax, distances, label="Crowding Distance")
        else:
            logger.info("Scatter plot is only supported for 2 or 3 objectives.")
            return None

        return self._finalize(fig, save_dir, "pareto_scatter.png")

    def _attach_colorbar(self, ax: plt.Axes, values: list[float], label: str | None = None) -> None:
        """Attach a colorbar to the given axes."""
        norm = plt.Normalize(min(values), max(values))
        sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
        sm.set_array([])
        cbar = plt.colorbar(mappable=sm, ax=ax)
        if label:
            cbar.set_label(label, fontsize=12)

    def _finalize(self, fig: plt.Figure, save_dir: str | Path | None, default_name: str) -> plt.Figure:
        """Finalize the plot by saving or showing it."""
        if save_dir:
            path = Path(save_dir)
            if path.is_dir():
                path = path / default_name
            try:
                fig.savefig(path, dpi=300, bbox_inches="tight")
                logger.info("Saved plot: %s", path)
            except Exception:
                logger.exception("Error saving plot to %s", path)
        else:
            plt.show()

        plt.close(fig)
        return fig
