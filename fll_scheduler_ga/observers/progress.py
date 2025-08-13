"""Progress bar observer for the FLL Scheduler GA."""

from dataclasses import dataclass, field

from tqdm import tqdm

from ..data_model.schedule import Population
from .base_observer import GaObserver


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

    def on_generation_end(
        self,
        generation: int,
        num_generations: int,
        population_size: int,
        best_fitness: tuple[float, ...],
        front_size: int,
    ) -> None:
        """Update progress bar with no new best."""
        if best_fitness:
            fitness_str = ", ".join([f"{s:.3f}" for s in best_fitness])
            fitness_str += f" | Σ={sum(best_fitness):.3f} ({sum(best_fitness) / len(best_fitness):.2%})"
            self._progress_bar.set_description(f"Front: {front_size:<3}/{population_size:<3} | Avg: {fitness_str}")
        self._progress_bar.update(1)

    def on_finish(self, pop: Population, front: Population) -> None:
        """Close the progress bar."""
        self._progress_bar.close()
