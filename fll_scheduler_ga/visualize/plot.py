"""Methods to create plots."""

import logging
from ast import Module
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..genetic.ga import GA
from ..genetic.schedule import Population

logger = logging.getLogger("visualize.plot")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


def get_matplotlib() -> Module:
    """Get the matplotlib module for plotting."""
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    return plt


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
    ) -> Any | None:
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
        history_df = pd.DataFrame(data=history, columns=self.ga_instance.evaluator.objectives)
        plt = get_matplotlib()
        fig, ax = plt.subplots(figsize=(12, 7))
        history_df.plot(kind="line", ax=ax, linewidth=2.5, alpha=0.8)
        x = np.arange(len(history_df))
        for col in history_df.columns:
            y = history_df[col].to_numpy()
            z = np.polyfit(x, y, 5)
            p = np.poly1d(z)
            ax.plot(x, p(x), linestyle="--", linewidth=2, label=f"{col} Trend")
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend(title="Objectives", fontsize=10)
        return _finalize(fig, save_dir, "fitness_plot.png")

    def plot_pareto_front(
        self, title: str = "Pareto Front: Trade-offs", save_dir: str | Path | None = None
    ) -> Any | None:
        """Generate and saves a parallel coordinates plot of the final Pareto front."""
        if not (front := self.ga_instance.pareto_front()):
            logger.warning("Cannot plot an empty Pareto front.")
            return None

        objectives = self.ga_instance.evaluator.objectives
        fig = _plot_parallel(front, objectives, title)
        _finalize(fig, save_dir, "pareto_parallel.png")

        if len(objectives) in (2, 3):
            scatter_name = f"pareto_scatter_{len(objectives)}d.png"
            return _plot_pareto_scatter(
                front, objectives, title=title, save_dir=Path(save_dir).with_name(scatter_name) if save_dir else None
            )
        return fig


def _plot_parallel(front: Population, objectives: list[str], title: str) -> None:
    """Create the parallel coordinates plot."""
    data = [[p.fitness[i] for i, _ in enumerate(objectives)] + [p.crowding] for p in front]
    objectives = [*objectives, "Crowding Distance"]
    dataframe = pd.DataFrame(data=data, columns=objectives)
    plt = get_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 7))
    pd.plotting.parallel_coordinates(
        frame=dataframe, class_column=objectives[-1], ax=ax, linewidth=1.5, alpha=0.7, colormap="viridis"
    )
    ax.set(title=title, xlabel="Objectives", ylabel="Score")
    plt.xticks(rotation=15, ha="right")
    ax.get_legend().remove()
    _attach_colorbar(ax, dataframe[objectives[-1]], label="Crowding Distance")
    return fig


def _plot_pareto_scatter(front: Population, objectives: list[str], title: str, save_dir: str | None) -> Any | None:
    """Create a 2D or 3D scatter plot of the Pareto front."""
    dataframe = pd.DataFrame(data=[p.fitness for p in front], columns=objectives)
    distances = [p.crowding for p in front]
    plt = get_matplotlib()

    if len(objectives) == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        x_obj, y_obj = objectives
        ax.scatter(dataframe[x_obj], dataframe[y_obj], c=distances, cmap="viridis", s=60, alpha=0.8)
        ax.set(title=title, xlabel=x_obj, ylabel=y_obj)
        _attach_colorbar(ax, distances, label="Crowding Distance")
    elif len(objectives) == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        x_obj, y_obj, z_obj = objectives
        ax.scatter(dataframe[x_obj], dataframe[y_obj], dataframe[z_obj], c=distances, cmap="viridis", s=60)
        ax.set(title=title, xlabel=x_obj, ylabel=y_obj, zlabel=z_obj)
        _attach_colorbar(ax, distances, label="Crowding Distance")
    else:
        logger.info("Scatter plot is only supported for 2 or 3 objectives.")
        return None

    return _finalize(fig, save_dir, "pareto_scatter.png")


def _finalize(fig: Any, save_dir: str | Path | None, default_name: str) -> Any:
    """Finalize the plot by saving or showing it."""
    plt = get_matplotlib()
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


def _attach_colorbar(ax: Any, values: list[float], label: str | None = None) -> None:
    """Attach a colorbar to the given axes."""
    plt = get_matplotlib()
    norm = plt.Normalize(min(values), max(values))
    sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    cbar = plt.colorbar(mappable=sm, ax=ax)
    if label:
        cbar.set_label(label, fontsize=12)
