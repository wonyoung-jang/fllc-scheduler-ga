"""Helper functions for plotting GA results."""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..data_model.schedule import Population

logger = logging.getLogger(__name__)


def plot_parallel(front: Population, objectives: list[str], title: str) -> None:
    """Create the parallel coordinates plot."""
    data = [[p.fitness[i] for i, _ in enumerate(objectives)] + [p.rank] for p in front]
    objectives = [*objectives, "Rank"]
    dataframe = pd.DataFrame(data=data, columns=objectives)
    fig, ax = plt.subplots(figsize=(12, 7))
    pd.plotting.parallel_coordinates(
        frame=dataframe, class_column=objectives[-1], ax=ax, linewidth=1.5, alpha=0.7, colormap="viridis"
    )
    ax.set(title=title, xlabel="Objectives", ylabel="Score")
    plt.xticks(rotation=15, ha="right")
    ax.get_legend().remove()
    _attach_colorbar(ax, dataframe[objectives[-1]].tolist(), label="Rank")
    return fig


def plot_pareto_scatter(front: Population, objectives: list[str], title: str, save_dir: str | None) -> Any | None:
    """Create a 2D or 3D scatter plot of the Pareto front."""
    dataframe = pd.DataFrame(data=[p.fitness for p in front], columns=objectives)
    ranks = [p.rank for p in front]

    if len(objectives) == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        x_obj, y_obj = objectives
        ax.scatter(dataframe[x_obj], dataframe[y_obj], c=ranks, cmap="viridis", s=60, alpha=0.8)
        ax.set(title=title, xlabel=x_obj, ylabel=y_obj)
        _attach_colorbar(ax, ranks, label="Rank")
    elif len(objectives) == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(azim=45, elev=40)
        x_obj, y_obj, z_obj = objectives
        ax.scatter(dataframe[x_obj], dataframe[y_obj], dataframe[z_obj], c=ranks, cmap="viridis", s=60)
        ax.set(title=title, xlabel=x_obj, ylabel=y_obj, zlabel=z_obj)
        _attach_colorbar(ax, ranks, label="Rank")
    else:
        logger.error("Scatter plot is only supported for 2 or 3 objectives.")
        return None

    return finalize(fig, save_dir, "pareto_scatter.png")


def finalize(fig: Figure, save_dir: str | Path | None, default_name: str) -> Any:
    """Finalize the plot by saving or showing it."""
    if save_dir:
        path = Path(save_dir)
        if path.is_dir():
            path = path / default_name
        try:
            fig.savefig(path, dpi=300)
            logger.debug("Saved plot: %s", path)
        except Exception:
            logger.exception("Error saving plot to %s", path)
    else:
        plt.show()

    plt.close(fig)
    return fig


def _attach_colorbar(ax: Axes, values: list[float], label: str | None = None) -> None:
    """Attach a colorbar to the given axes."""
    unique_values = sorted(set(values))
    cmap_name = "viridis"

    if len(unique_values) <= 10:
        cmap = plt.get_cmap(cmap_name, len(unique_values))
        norm = plt.matplotlib.colors.BoundaryNorm(np.arange(min(values) - 0.5, max(values) + 1.5), cmap.N)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(mappable=sm, ax=ax, ticks=unique_values)
    else:
        norm = plt.Normalize(min(values), max(values))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_name)
        cbar = plt.colorbar(mappable=sm, ax=ax)

    sm.set_array([])
    if label:
        cbar.set_label(label, fontsize=12)
