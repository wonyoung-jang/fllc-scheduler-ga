"""Tools for Non-dominated Sorting Genetic Algorithm III (NSGA-III)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from logging import getLogger
from math import comb
from typing import TYPE_CHECKING

import numpy as np

from ..config.constants import EPSILON

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = getLogger(__name__)


def calc_ref_points(n_obj: int, n_pop: int) -> np.ndarray:
    """Generate a set of structured reference points."""
    m = n_obj
    p = 1
    while comb(m + p - 1, m - 1) < n_pop:
        p += 1

    def _generate_coordinates() -> Iterator[np.ndarray]:
        for dividers in combinations(range(m + p - 1), m - 1):
            coords = np.zeros(m, dtype=float)
            prev = -1
            for i, divider in enumerate(dividers):
                coords[i] = divider - prev - 1
                prev = divider
            coords[-1] = m + p - 1 - dividers[-1] - 1
            yield (coords / p)

    coordinates = tuple(_generate_coordinates())
    points = np.array(coordinates, dtype=float)
    logger.debug("Generated %d reference points:\n%s", points.shape[0], points)
    return points


def calc_norm_sq_of_refs(points: np.ndarray) -> np.ndarray:
    """Calculate the squared norms of the reference points."""
    norm_sq = (points**2).sum(axis=1)
    norm_sq[norm_sq == 0.0] = EPSILON
    logger.debug("Computed squared norms of reference points:\n%s", norm_sq)
    return norm_sq


@dataclass(slots=True)
class ReferenceDirections:
    """Structured reference points for NSGA-III."""

    n_refs: int
    points: np.ndarray
    norm_sq: np.ndarray


@dataclass(slots=True)
class NonDominatedSorting:
    """Non-dominated sorting utility."""

    @staticmethod
    def get_fronts(fits: np.ndarray, n_pop: int) -> list[np.ndarray]:
        """Perform non-dominated sorting on the population."""
        n_fit = fits.shape[0]
        if n_fit == 0:
            return []

        # Pairwise comparisons using broadcasting
        all_ge = (fits[:, None, :] >= fits[None, :, :]).all(axis=2)
        any_gt = (fits[:, None, :] > fits[None, :, :]).any(axis=2)
        # dom[i,j] = True if i dominates j (>= on all and > on at least one)
        dom = np.logical_and(all_ge, any_gt)
        # Number of individuals that dominate j = sum over i dom[i,j]
        dom_count = dom.sum(axis=0)
        # Adjacency lists: who each i dominates
        assigned = np.zeros(n_fit, dtype=bool)
        fronts: list[np.ndarray] = []

        # Initial front: those not dominated by anybody
        current_front: np.ndarray = (dom_count == 0).nonzero()[0]
        assigned[current_front] = True
        fronts.append(current_front)
        n_ranked = current_front.size

        # Build subsequent fronts
        while n_ranked < n_pop and current_front.size > 0:
            # Sum of domination relationships from current_front to each j
            decrement = dom[current_front, :].sum(axis=0)
            dom_count = dom_count - decrement

            # Next front: those now not dominated by anybody
            next_front: np.ndarray = ((dom_count == 0) & (~assigned)).nonzero()[0]
            if next_front.size == 0:
                break

            assigned[next_front] = True
            fronts.append(next_front)
            n_ranked += next_front.size
            current_front = next_front
        return fronts


@dataclass(slots=True)
class NSGA3:
    """Non-dominated Sorting Genetic Algorithm III (NSGA-III)."""

    rng: np.random.Generator
    refs: ReferenceDirections
    sorting: NonDominatedSorting

    def select(self, fits: np.ndarray, n_pop: int) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
        """Select the next generation using NSGA-III principles."""
        fronts = self.sorting.get_fronts(fits, n_pop)
        last_idx = len(fronts) - 1
        selected_indices = np.array([i for f in fronts for i in f], dtype=int)
        selected_fits = fits[selected_indices]
        refs, distances = self.norm_and_associate(selected_fits)

        if len(fronts) == 1:
            fronts[0] = self.rng.permutation(selected_indices)[:n_pop]
            fronts = tuple(fronts)
            flat = np.concatenate(fronts)
            ranks = self.ranks_from_fronts(fronts, fits.shape[0])
            return fronts, flat, ranks[flat]

        last_front_indices = fronts[last_idx]
        n_last_front = last_front_indices.size

        fronts = fronts[:last_idx]

        selected = selected_indices[:-n_last_front]
        n_remaining = n_pop - selected.size
        niche_selected = refs[:-n_last_front]
        counts = self.count(niche_selected)
        niches = self.niche(
            counts=counts,
            n_last_front=n_last_front,
            n_remaining=n_remaining,
            niche_refs=refs[-n_last_front:],
            niche_dists=distances[-n_last_front:],
        )
        last_front_indices = last_front_indices[niches]
        fronts.append(last_front_indices)
        fronts = tuple(fronts)
        flat = np.concatenate(fronts)
        ranks = self.ranks_from_fronts(fronts, fits.shape[0])
        return fronts, flat, ranks[flat]

    def ranks_from_fronts(self, fronts: tuple[np.ndarray, ...], n_individuals: int) -> np.ndarray:
        """Assign ranks to individuals based on their fronts."""
        ranks = np.full(n_individuals, fill_value=-1, dtype=int)
        for rank, front in enumerate(fronts):
            ranks[front] = rank
        return ranks

    def niche(
        self,
        counts: np.ndarray,
        n_last_front: int,
        n_remaining: int,
        niche_refs: np.ndarray,
        niche_dists: np.ndarray,
    ) -> np.ndarray:
        """Select k individuals from the last front using a niching mechanism."""
        # Mask of individuals in the last front still available for selection
        mask = np.full(n_last_front, fill_value=True, dtype=bool)
        n_selected = 0
        while n_selected < n_remaining:
            # All reference points associated with individuals still available
            available_refs = np.unique(niche_refs[mask])
            ref_counts = counts[available_refs]

            # Minimum count among those reference points
            min_count = ref_counts.min()

            # Number of individuals to select from this niche
            n_select = n_remaining - n_selected
            niche_indices = available_refs[(ref_counts == min_count).nonzero()[0]]
            niche_indices = niche_indices[self.rng.permutation(niche_indices.size)[:n_select]]

            for niche_idx in niche_indices:
                # Indices of individuals in this niche still available
                next_i = ((niche_refs == niche_idx) & mask).nonzero()[0]
                self.rng.shuffle(next_i)
                index = next_i[niche_dists[next_i].argmin()] if counts[niche_idx] == 0 else next_i[0]
                mask[index] = False
                counts[niche_idx] += 1
                n_selected += 1
                if n_selected >= n_remaining:
                    break

        # Return the masked indices
        return (~mask).nonzero()[0]

    def norm_and_associate(self, fits: np.ndarray) -> tuple[np.ndarray, ...]:
        """Normalize objectives then associate individuals with nearest reference points."""
        ideal = fits.max(axis=0)
        nadir = fits.min(axis=0)

        span = ideal - nadir
        span[span == 0.0] = EPSILON

        norm = (ideal - fits) / span

        coeffs = (norm @ self.refs.points.T) / self.refs.norm_sq
        coeffs[coeffs < 0.0] = 0.0

        proj = coeffs[:, :, None] * self.refs.points[None, :, :]
        residuals = norm[:, None, :] - proj
        dists: np.ndarray = np.linalg.norm(residuals, axis=2)
        min_dists = dists.min(axis=1)

        # To break ties uniformly and vectorized:
        ties = dists == min_dists[:, None]
        rand_matrix = self.rng.random(dists.shape)

        # Mask tied positions with random values
        rand_matrix[~ties] = -1.0
        chosen_refs = rand_matrix.argmax(axis=1)  # Index of chosen ref per individual

        return chosen_refs, min_dists

    def count(self, niche_selected: np.ndarray) -> np.ndarray:
        """Count how many individuals are associated with each reference point."""
        counts = np.zeros(self.refs.n_refs, dtype=int)
        indices, count = np.unique(niche_selected, return_counts=True)
        counts[indices] = count
        return counts
