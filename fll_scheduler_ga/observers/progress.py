"""Progress bar observer for the FLL Scheduler GA."""

from dataclasses import dataclass, field
from logging import Logger

from tqdm import tqdm

from ..genetic.schedule import Population
from .base_observer import GaObserver


@dataclass(slots=True)
class TqdmObserver(GaObserver):
    """Observer that displays a tqdm progress bar for generations."""

    logger: Logger
    _progress_bar: tqdm = field(init=False, repr=False)

    def on_start(self, num_generations: int) -> None:
        """Initialize the progress bar at the start of the GA run."""
        self._progress_bar = tqdm(total=num_generations, unit="gen")

    def on_generation_end(
        self,
        generation: int,
        num_generations: int,
        population_size: int,
        best_fitness: tuple[float, ...],
        front_size: int,
    ) -> None:
        """Update progress bar with no new best."""
        fitness_str = ", ".join([f"{s:.3f}" for s in best_fitness])
        self._progress_bar.set_description(f"Front size: {front_size:<3} | Avg Fitness: ({fitness_str})")
        self._progress_bar.update(1)

    def on_finish(self, pop: Population, front: Population) -> None:
        """Close the progress bar."""
        self._progress_bar.close()

    def on_mutation(self, mutation_name: str, *, successful: bool) -> None:
        """Handle mutation events, currently a no-op."""

    def on_crossover(self, crossover_name: str, *, successful: bool) -> None:
        """Handle crossover events, currently a no-op."""
