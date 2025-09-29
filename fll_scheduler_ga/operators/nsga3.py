"""Tools for Non-dominated Sorting Genetic Algorithm III (NSGA-III)."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from logging import getLogger
from math import comb

import numpy as np

from ..config.constants import EPSILON

logger = getLogger(__name__)


@dataclass(slots=True)
class NSGA3:
    """Non-dominated Sorting Genetic Algorithm III (NSGA-III)."""

    rng: np.random.Generator
    n_objectives: int
    n_total_pop: int
    ref_points: np.ndarray = field(default=None, repr=False)
    norm_sq: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to generate reference points."""
        self.ref_points = self.init_ref_points()
        ns = np.sum(self.ref_points**2, axis=1)
        self.norm_sq = np.where(ns == 0.0, EPSILON, ns)

    def init_ref_points(self) -> np.ndarray:
        """Generate a set of structured reference points."""
        m = self.n_objectives
        p = 1
        while comb(m + p - 1, m - 1) < self.n_total_pop:
            p += 1

        ref_pts = np.array([], dtype=float).reshape(0, m)
        for dividers in combinations(range(m + p - 1), m - 1):
            coords = np.zeros(m, dtype=float)
            prev = -1
            for i, divider in enumerate(dividers):
                coords[i] = divider - prev - 1
                prev = divider
            coords[-1] = m + p - 1 - dividers[-1] - 1
            ref_pts = np.vstack([ref_pts, coords / p])

        logger.debug("Generated %d reference points:\n%s", len(ref_pts), ref_pts)
        return ref_pts

    def select(self, fits: np.ndarray[float], n: int) -> list[np.ndarray[int]]:
        """Select the next generation using NSGA-III principles."""
        fronts = self.non_dominated_sort(fits, n)
        last_idx = len(fronts) - 1
        selected_indices = np.array([i for f in fronts for i in f], dtype=int)
        selected_fits = fits[selected_indices]
        refs, distances = self.norm_and_associate(selected_fits)

        if len(fronts) == 1:
            fronts[0] = self.rng.permutation(selected_indices)[:n]
            return fronts

        last_front_indices = fronts[last_idx]
        n_last_front = len(last_front_indices)

        fronts = fronts[:last_idx]

        selected = selected_indices[:-n_last_front]
        n_remaining = n - len(selected)
        selected_refs = refs[:-n_last_front]
        counts = self.count(selected_refs)
        niches = self.niche(
            counts=counts,
            n_last_front=n_last_front,
            n_remaining=n_remaining,
            niche_refs=refs[-n_last_front:],
            niche_dists=distances[-n_last_front:],
        )
        last_front_indices = last_front_indices[niches]
        fronts.append(last_front_indices)
        return fronts

    def non_dominated_sort(self, fits: np.ndarray[float], n: int) -> list[np.ndarray[int]]:
        """Perform non-dominated sorting on the population."""
        n_pop = fits.shape[0]
        if n_pop == 0:
            return []

        # Pairwise comparisons using broadcasting
        all_ge = np.all(fits[:, None, :] >= fits[None, :, :], axis=2)
        any_gt = np.any(fits[:, None, :] > fits[None, :, :], axis=2)
        # dom[i,j] = True if i dominates j (>= on all and > on at least one)
        dom = np.logical_and(all_ge, any_gt)
        # Number of individuals that dominate j = sum over i dom[i,j]
        dom_count = np.sum(dom, axis=0)
        # Adjacency lists: who each i dominates
        assigned = np.zeros(n_pop, dtype=bool)
        fronts: list[np.ndarray[int]] = []

        # Initial front: those not dominated by anybody
        current_front = np.where(dom_count == 0)[0]
        assigned[current_front] = True
        fronts.append(current_front)
        n_ranked = current_front.size

        # Build subsequent fronts
        while n_ranked < n and current_front.size > 0:
            # Sum of domination relationships from current_front to each j
            decrement = np.sum(dom[current_front, :], axis=0)
            dom_count = dom_count - decrement

            # Next front: those now not dominated by anybody
            next_front = np.where((dom_count == 0) & (~assigned))[0]
            if next_front.size == 0:
                break

            assigned[next_front] = True
            fronts.append(next_front)
            n_ranked += next_front.size
            current_front = next_front
        return fronts

    def niche(
        self,
        counts: np.ndarray,
        n_last_front: int,
        n_remaining: int,
        niche_refs: np.ndarray,
        niche_dists: np.ndarray,
    ) -> np.ndarray[int]:
        """Select k individuals from the last front using a niching mechanism."""
        shuffle = self.rng.shuffle
        permutation = self.rng.permutation
        # Mask of individuals in the last front still available for selection
        mask = np.full(n_last_front, fill_value=True)
        n_selected = 0
        while n_selected < n_remaining:
            # All reference points associated with individuals still available
            available_refs = np.unique(niche_refs[mask])
            ref_counts = counts[available_refs]

            # Minimum count among those reference points
            min_count = ref_counts.min()

            # Number of individuals to select from this niche
            n_select = n_remaining - n_selected
            niche_indices = available_refs[np.where(ref_counts == min_count)[0]]
            niche_indices = niche_indices[permutation(len(niche_indices))[:n_select]]

            for niche_idx in niche_indices:
                # Indices of individuals in this niche still available
                next_i = np.where((niche_refs == niche_idx) & mask)[0]
                shuffle(next_i)
                index = next_i[np.argmin(niche_dists[next_i])] if counts[niche_idx] == 0 else next_i[0]
                mask[index] = False
                counts[niche_idx] += 1
                n_selected += 1

        # Return the masked indices
        return np.where(~mask)[0]

    def norm_and_associate(self, fits: np.ndarray[float]) -> tuple[np.ndarray, np.ndarray]:
        """Normalize objectives then associate individuals with nearest reference points."""
        ideal = fits.max(axis=0)
        nadir = fits.min(axis=0)

        span = ideal - nadir
        span = np.where(span == 0.0, EPSILON, span)

        norm = (ideal - fits) / span

        coeffs = (norm @ self.ref_points.T) / self.norm_sq
        coeffs = np.where(coeffs < 0.0, 0.0, coeffs)

        proj = coeffs[:, :, None] * self.ref_points[None, :, :]

        residuals = norm[:, None, :] - proj
        dists: np.ndarray = np.linalg.norm(residuals, axis=2)
        min_dists = dists.min(axis=1)

        # To break ties uniformly and vectorized:
        ties = dists == min_dists[:, None]
        rand_matrix = self.rng.random(dists.shape)
        # Mask tied positions with random values
        masked_random = np.where(ties, rand_matrix, -1.0)
        chosen_refs = masked_random.argmax(axis=1)  # Index of chosen ref per individual

        return chosen_refs, min_dists

    def count(self, idx_to_count: np.ndarray[int]) -> np.ndarray[int]:
        """Count how many individuals are associated with each reference point."""
        counts = np.zeros(len(self.ref_points), dtype=int)
        if idx_to_count.size > 0:
            indices, count = np.unique(idx_to_count, return_counts=True)
            counts[indices] = count
        return counts
