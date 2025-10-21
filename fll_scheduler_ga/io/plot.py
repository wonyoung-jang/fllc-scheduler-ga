"""Methods to create plots."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ..config.constants import FitnessObjective
    from ..genetic.ga import GA


logger = logging.getLogger("visualize.plot")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
plt.style.use("seaborn-v0_8-whitegrid")


@dataclass(slots=True)
class Plot:
    """A class for creating and managing plots related to the GA run."""

    ga: GA
    save_dir: str | Path | None
    cmap_name: str
    objectives: list[FitnessObjective]
    ref_points: np.ndarray

    def plot(self) -> None:
        """Create all plots."""
        self.plot_fitness()
        self.plot_parallel()
        self.plot_scatter()

    def plot_fitness(self) -> None:
        """Create figure that summarizes how the average fitness of the first Pareto front evolved by generation.

        Args:
            title: Figure title.
            xlabel: Label on the X-axis.
            ylabel: Label on the Y-axis.
            save_dir: Directory to save the figure. If None, the plot is shown.

        """
        history = self.ga.fitness_history
        history = history[history[:, 0] >= 0]  # Filter out generations (if program terminated early)
        if not history.any():
            logger.error("Cannot plot fitness. No generation history was recorded.")
            return

        fig, ax = plt.subplots(figsize=(12, 7))
        columns = [f.value for f in self.objectives]
        history_df = pd.DataFrame(data=history, columns=columns)
        history_df.plot(kind="line", ax=ax, linewidth=2.5, alpha=0.9)

        x = np.arange(len(history_df))
        for col in history_df.columns:
            y = history_df[col].to_numpy()
            z = np.polyfit(x, y, 3)
            p = np.poly1d(z)
            ax.plot(x, p(x), linestyle="--", linewidth=0.8, label=f"{col} Trend (deg={len(z) - 1})")

        ax.set(title="Fitness over time", xlabel="Generations", ylabel="Average fitnesses")
        ax.legend(title="Objectives", fontsize=10)
        fig.tight_layout()
        self.finalize(fig, "fitness_vs_generation.png")

    def plot_parallel(self) -> None:
        """Create the parallel coordinates plot."""
        data = [[p.fitness[i] for i, _ in enumerate(self.objectives)] + [p.rank] for p in self.ga.total_population]
        objectives = [*self.objectives, "Rank"]
        dataframe = pd.DataFrame(data=data, columns=objectives)
        fig, ax = plt.subplots(figsize=(12, 7))
        pd.plotting.parallel_coordinates(
            frame=dataframe,
            class_column=objectives[-1],
            ax=ax,
            linewidth=1.5,
            alpha=0.7,
            colormap=self.cmap_name,
        )
        ax.set(title="Trade-off parallel coordinates", xlabel="Objectives", ylabel="Score")
        ax.get_legend().remove()
        plt.xticks(rotation=15, ha="right")
        ranks = dataframe[objectives[-1]].tolist()
        self._attach_colorbar(ax, ranks, label="Rank")
        self.finalize(fig, "pareto_parallel.png")

    def plot_scatter(self) -> None:
        """Create a 2D or 3D scatter plot of the Pareto front."""
        len_obj = len(self.objectives)
        if len_obj not in (2, 3):
            logger.error("Cannot plot Pareto scatter for %d objectives. Only 2D and 3D supported.", len_obj)
            return

        dataframe = pd.DataFrame(data=[p.fitness for p in self.ga.total_population], columns=self.objectives)
        ranks = [p.rank for p in self.ga.total_population]

        if len_obj == 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            x_obj, y_obj = self.objectives
            ax.scatter(dataframe[x_obj], dataframe[y_obj], c=ranks, cmap=self.cmap_name, s=60, alpha=0.8)
            ax.set(
                title=f"{len(self.ga.context.evaluator.objectives)}D scatter plot of schedules",
                xlabel=x_obj,
                ylabel=y_obj,
            )
            self._attach_colorbar(ax, ranks, label="Rank")
            self.finalize(fig, f"pareto_scatter_{len_obj}d.png")
        elif len_obj == 3:
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": "3d"})
            x_obj, y_obj, z_obj = self.objectives
            ax.view_init(azim=45, elev=40)
            ax.scatter(dataframe[x_obj], dataframe[y_obj], dataframe[z_obj], c=ranks, cmap=self.cmap_name, s=60)
            ax.set(
                title=f"{len(self.ga.context.evaluator.objectives)}D scatter plot of schedules",
                xlabel=x_obj,
                ylabel=y_obj,
                zlabel=z_obj,
                box_aspect=[1, 1, 1],
            )
            self._attach_colorbar(ax, ranks, label="Rank")
            self.finalize(fig, f"pareto_scatter_{len_obj}d.png")
            ax.scatter(
                self.ref_points[:, 0],
                self.ref_points[:, 1],
                self.ref_points[:, 2],
                c="red",
                s=30,
                label="Reference Points",
            )
            ax.legend()
            self.finalize(fig, f"pareto_scatter_{len_obj}d_ref.png")

    def finalize(self, fig: Figure, filename: str) -> None:
        """Finalize the plot by saving or showing it."""
        try:
            if self.save_dir:
                path = Path(self.save_dir) / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path, dpi=300)
                logger.debug("Saved plot: %s", path)
                plt.close(fig)
            else:
                plt.show()
                plt.close(fig)
        except Exception:
            logger.exception("Error saving plot to %s", path)
            plt.close(fig)

    def _attach_colorbar(self, ax: Axes, values: list[int], label: str | None = None) -> None:
        """Attach a colorbar to the given axes."""
        unique_values = sorted(set(values))

        if len(unique_values) <= 10:
            cmap = plt.get_cmap(self.cmap_name, len(unique_values))
            norm = mcolors.BoundaryNorm(np.arange(min(values) - 0.5, max(values) + 1.5), cmap.N)
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            cbar = plt.colorbar(mappable=sm, ax=ax, ticks=unique_values)
        else:
            norm = plt.Normalize(min(values), max(values))
            sm = plt.cm.ScalarMappable(norm=norm, cmap=self.cmap_name)
            cbar = plt.colorbar(mappable=sm, ax=ax)

        sm.set_array([])
        if label:
            cbar.set_label(label, fontsize=12)
