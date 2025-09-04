"""Progress bar observer for the FLL Scheduler GA."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tqdm import tqdm

from .base_observer import GaObserver

if TYPE_CHECKING:
    from ..data_model.schedule import Population


@dataclass(slots=True)
class TqdmObserver(GaObserver):
    """Observer that displays a tqdm progress bar for generations."""

    _progress_bar: tqdm = field(init=False, repr=False)

    def on_start(self, num_generations: int) -> None:
        """Initialize the progress bar at the start of the GA run."""
        self._progress_bar = tqdm(
            total=num_generations,
            unit="gen",
            desc="Initializing...",
            colour="MAGENTA",
        )

    def on_generation_end(self, generation: int, num_generations: int, best_fitness: tuple[float, ...]) -> None:
        """Update progress bar with no new best."""
        if best_fitness:
            fitness_str = ", ".join([f"{s:.3f}" for s in best_fitness])
            fitness_str += f" | Σ={sum(best_fitness):.4f} ({sum(best_fitness) / len(best_fitness):.3%})"
            self._progress_bar.set_description(f"Fitness: {fitness_str}")
        self._progress_bar.update()

    def on_finish(self, pop: Population, front: Population) -> None:
        """Close the progress bar."""
        self._progress_bar.close()
