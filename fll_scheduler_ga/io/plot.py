"""Methods to create plots."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ..config.constants import FitnessObjective
    from ..config.schemas import ExportModel
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
    objectives: list[FitnessObjective]
    ref_points: np.ndarray
    export_model: ExportModel

    def plot(self) -> None:
        """Create all plots."""
        if self.export_model.plot_fitness:
            self.plot_fitness()
        if self.export_model.plot_parallel:
            self.plot_parallel()
        if self.export_model.plot_scatter:
            self.plot_scatter()

    def plot_fitness(self) -> None:
        """Create figure that summarizes how the average fitness of the first Pareto front evolved by generation.

        Args:
            title: Figure title.
            xlabel: Label on the X-axis.
            ylabel: Label on the Y-axis.
            save_dir: Directory to save the figure. If None, the plot is shown.

        """
        history = self.ga.fitness_history.history
        history = history[history[:, 0] >= 0]  # Filter out generations (if program terminated early)
        if not history.any():
            logger.error("Cannot plot fitness. No generation history was recorded.")
            return

        fig, ax = plt.subplots(figsize=(12, 7))
        columns = [f.value for f in self.objectives]

        x = np.arange(len(history))
        for i, col in enumerate(columns):
            y = history[:, i]
            ax.plot(x, y, linewidth=2.5, alpha=0.9, label=col)

            z = np.polyfit(x, y, 3)
            p = np.poly1d(z)
            ax.plot(x, p(x), linestyle="--", linewidth=0.8, label=f"{col} Trend (deg={len(z) - 1})")

        ax.set(title="Fitness over time", xlabel="Generations", ylabel="Average fitnesses")
        ax.legend(title="Objectives", fontsize=10)
        fig.tight_layout()
        self.finalize(fig, "fitness_vs_generation.png")

    def plot_parallel(self) -> None:
        """Create the parallel coordinates plot."""
        data = np.array([p.fitness for p in self.ga.total_population])
        ranks = np.array([p.rank for p in self.ga.total_population])
        fig, ax = plt.subplots(figsize=(12, 7))

        x = range(len(self.objectives))
        colors = plt.get_cmap(self.export_model.cmap_name)(np.linspace(0, 1, len(self.ga.total_population)))
        for i, ind_fitness in enumerate(data):
            ax.plot(x, ind_fitness, color=colors[i], alpha=0.7, linewidth=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(self.objectives, rotation=15, ha="right")
        ax.set(title="Trade-off parallel coordinates", xlabel="Objectives", ylabel="Score")
        plt.xticks(rotation=15, ha="right")
        self._attach_colorbar(ax, ranks, label="Rank")
        self.finalize(fig, "pareto_parallel.png")

    def plot_scatter(self) -> None:
        """Create a 2D or 3D scatter plot of the Pareto front."""
        n_objectives = len(self.objectives)
        if n_objectives not in (2, 3):
            logger.error("Cannot plot Pareto scatter for %d objectives. Only 2D and 3D supported.", n_objectives)
            return

        cmap_name = self.export_model.cmap_name
        data = np.array([p.fitness for p in self.ga.total_population])
        ranks = np.array([p.rank for p in self.ga.total_population])

        if n_objectives == 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            x_obj, y_obj = self.objectives
            ax.scatter(data[:, 0], data[:, 1], c=ranks, cmap=cmap_name, s=60, alpha=0.8)
            ax.set(
                title=f"{n_objectives}D scatter plot of schedules",
                xlabel=x_obj,
                ylabel=y_obj,
            )
            self._attach_colorbar(ax, ranks, label="Rank")
            self.finalize(fig, f"pareto_scatter_{n_objectives}d.png")
        elif n_objectives == 3:
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": "3d"})
            x_obj, y_obj, z_obj = self.objectives
            ax.view_init(azim=45, elev=40)
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=ranks, cmap=cmap_name, s=60)
            ax.set(
                title=f"{n_objectives}D scatter plot of schedules",
                xlabel=x_obj,
                ylabel=y_obj,
                zlabel=z_obj,
                box_aspect=[1, 1, 1],
            )
            self._attach_colorbar(ax, ranks, label="Rank")
            self.finalize(fig, f"pareto_scatter_{n_objectives}d.png")
            ax.scatter(
                self.ref_points[:, 0],
                self.ref_points[:, 1],
                self.ref_points[:, 2],
                c="red",
                s=30,
                label="Reference Points",
            )
            ax.legend()
            self.finalize(fig, f"pareto_scatter_{n_objectives}d_ref.png")

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
            cmap = plt.get_cmap(self.export_model.cmap_name, len(unique_values))
            norm = mcolors.BoundaryNorm(np.arange(min(values) - 0.5, max(values) + 1.5), cmap.N)
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            cbar = plt.colorbar(mappable=sm, ax=ax, ticks=unique_values)
        else:
            norm = plt.Normalize(min(values), max(values))
            sm = plt.cm.ScalarMappable(norm=norm, cmap=self.export_model.cmap_name)
            cbar = plt.colorbar(mappable=sm, ax=ax)

        sm.set_array([])
        if label:
            cbar.set_label(label, fontsize=12)
