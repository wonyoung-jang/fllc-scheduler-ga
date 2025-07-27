"""Methods to create plots."""

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..genetic.ga import GA
from .utils import finalize, plot_parallel, plot_pareto_scatter

logger = logging.getLogger("visualize.plot")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
plt.style.use("seaborn-v0_8-whitegrid")


@dataclass(slots=True)
class Plot:
    """A class for creating and managing plots related to the GA run."""

    ga_instance: GA

    def plot_fitness(self, title: str, xlabel: str, ylabel: str, save_dir: str | Path | None) -> None:
        """Create figure that summarizes how the average fitness of the first Pareto front evolved by generation.

        Args:
            title: Figure title.
            xlabel: Label on the X-axis.
            ylabel: Label on the Y-axis.
            save_dir: Directory to save the figure. If None, the plot is shown.

        """
        if not (history := self.ga_instance.fitness_history):
            logger.error("Cannot plot fitness. No generation history was recorded.")
            return
        history_df = pd.DataFrame(data=history, columns=self.ga_instance.context.evaluator.objectives)
        fig, ax = plt.subplots(figsize=(12, 7))
        history_df.plot(kind="line", ax=ax, linewidth=2.5, alpha=0.8)
        x = np.arange(len(history_df))
        for col in history_df.columns:
            y = history_df[col].to_numpy()
            z = np.polyfit(x, y, 3)
            p = np.poly1d(z)
            ax.plot(x, p(x), linestyle="--", linewidth=0.5, label=f"{col} Trend (^3)")
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend(title="Objectives", fontsize=10)

        finalize(fig, save_dir, "fitness_plot.png")

    def plot_pareto_front(self, title: str, save_dir: str | Path | None) -> None:
        """Generate and saves a parallel coordinates plot of the final Pareto front."""
        if not (pop := self.ga_instance.total_population):
            logger.warning("Cannot plot an empty Pareto front.")
            return

        objectives = self.ga_instance.context.evaluator.objectives
        fig = plot_parallel(pop, objectives, title)
        finalize(fig, save_dir, "pareto_parallel.png")

        if len(objectives) in (2, 3):
            scatter_name = f"pareto_scatter_{len(objectives)}d.png"
            plot_pareto_scatter(
                pop,
                objectives,
                title=title,
                save_dir=Path(save_dir).with_name(scatter_name) if save_dir else None,
            )
        else:
            logger.warning(
                "Cannot plot Pareto scatter for %d objectives. Only 2D and 3D plots are supported.", len(objectives)
            )
