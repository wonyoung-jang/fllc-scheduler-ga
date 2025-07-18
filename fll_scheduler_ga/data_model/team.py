"""Team data model for FLL Scheduler GA."""

from dataclasses import dataclass, field
from logging import getLogger
from math import sqrt

from ..config.config import RoundType, TournamentConfig
from ..data_model.event import Event
from .location import Location

logger = getLogger(__name__)


type TeamMap = dict[int, Team]

PENALTY = 1e-16


@dataclass(slots=True, frozen=True)
class TeamInfo:
    """Data model for team information in the FLL Scheduler GA."""

    identity: int


@dataclass(slots=True)
class TeamFactory:
    """Factory class to create Team instances."""

    config: TournamentConfig
    conflicts: dict[int, set[int]]
    base_teams_info: frozenset[TeamInfo] = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to validate the base team information."""
        self.base_teams_info = frozenset(TeamInfo(i) for i in range(1, self.config.num_teams + 1))

    def build(self) -> TeamMap:
        """Create a mapping of team identities to Team instances.

        Returns:
            TeamMap: A mapping of team identities to Team instances.

        """
        return {
            i.identity: Team(
                info=i,
                round_types=self.config.round_requirements.copy(),
                event_conflict_map=self.conflicts,
            )
            for i in self.base_teams_info
        }


@dataclass(slots=True)
class Team:
    """Data model for a team in the FLL Scheduler GA."""

    info: TeamInfo
    round_types: dict[RoundType, int]
    event_conflict_map: dict[int, set[int]]

    identity: int = field(init=False, repr=False)
    events: list[Event] = field(default_factory=list)
    opponents: list[int] = field(default_factory=list, repr=False)
    locations: list[Location] = field(default_factory=list, repr=False)

    fitness: tuple[float] = field(default=None, init=False, repr=False)

    _event_ids: set[int] = field(default_factory=set, repr=False)
    _rounds_needed: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to sort events and time slots."""
        self.identity = self.info.identity
        self._rounds_needed = sum(self.round_types.values())

    def __hash__(self) -> int:
        """Hash function for the team based on its identity."""
        return hash(self.info.identity)

    def __len__(self) -> int:
        """Get the count a team is counted as."""
        return 1

    def rounds_needed(self) -> int:
        """Get the total number of rounds still needed for the team."""
        return self._rounds_needed

    def needs_round(self, round_type: RoundType) -> int:
        """Check if the team still needs to participate in a given round type."""
        return self.round_types[round_type]

    def switch_event(self, old_event: Event, new_event: Event) -> None:
        """Switch an event for the team."""
        self.remove_event(old_event)
        self.add_event(new_event)

    def remove_event(self, event: Event) -> None:
        """Unbook a team from an event."""
        self.round_types[event.round_type] += 1
        self._rounds_needed += 1
        self.events.remove(event)
        self._event_ids.remove(event.identity)
        if event.location.teams_per_round == 2:
            self.locations.remove(event.location)
        self.fitness = None

    def add_event(self, event: Event) -> None:
        """Book a team for an event."""
        self.round_types[event.round_type] -= 1
        self._rounds_needed -= 1
        self.events.append(event)
        self._event_ids.add(event.identity)
        if event.location.teams_per_round == 2:
            self.locations.append(event.location)
        self.fitness = None

    def switch_opponent(self, old_opponent: "Team", new_opponent: "Team") -> None:
        """Switch the opponent for a given event."""
        self.remove_opponent(old_opponent)
        self.add_opponent(new_opponent)

    def remove_opponent(self, opponent: "Team") -> None:
        """Remove an opponent for a given event."""
        self.opponents.remove(opponent.identity)
        self.fitness = None

    def add_opponent(self, opponent: "Team") -> None:
        """Add an opponent for a given event."""
        self.opponents.append(opponent.identity)
        self.fitness = None

    def conflicts(self, new_event: Event) -> bool:
        """Check if adding a new event would cause a time conflict.

        Args:
            new_event (Event): The new event to check for conflicts.

        Returns:
            bool: True if there is a conflict, False otherwise.

        """
        if new_event.identity in self._event_ids:
            return True

        if not (potential_conflicts := self.event_conflict_map.get(new_event.identity)):
            return False

        return any(existing_event_id in potential_conflicts for existing_event_id in self._event_ids)

    def score(self) -> None:
        """Calculate the fitness score for the team based on various criteria."""
        if self.fitness is None:
            self.fitness = (
                self.score_break_time(PENALTY),
                self.score_table_consistency(PENALTY),
                self.score_opponent_variety(PENALTY),
            )

    def score_break_time(self, penalty: float = 0.01) -> float:
        """Calculate a score based on the break times between events."""
        if len(self.events) < 2:
            return 1
        _events = self.events
        _events.sort(key=lambda e: e.identity)
        break_times = []
        for i in range(1, len(_events)):
            start = _events[i].timeslot.start
            stop = _events[i - 1].timeslot.stop
            duration_seconds = (start - stop).total_seconds()
            break_times.append(duration_seconds // 60)
        if not (n := len(break_times)):
            return 1
        sum_x = sum(break_times)
        mean_x = sum_x / n
        if mean_x == 0:
            return 0
        sum_sq_diff = sum((b - mean_x) ** 2 for b in break_times)
        variance = sum_sq_diff / n
        stdev_x = sqrt(variance)
        coeff_of_variation = stdev_x / mean_x

        break_ratio = 1 / (1 + coeff_of_variation)
        break_penalty = penalty ** break_times.count(0)
        return break_ratio * break_penalty

    def score_table_consistency(self, penalty: float = 0.01) -> float:
        """Calculate a score based on the consistency of table assignments."""
        unique, total = self._get_unique_and_total(self.locations)
        table_ratio = unique / total if total else 1
        table_ratio = 1 / (1 + table_ratio)
        table_penalty = penalty**total if unique == total else 1
        return table_ratio * table_penalty

    def score_opponent_variety(self, penalty: float = 0.01) -> float:
        """Calculate a score based on the variety of opponents faced."""
        unique, total = self._get_unique_and_total(self.opponents)
        opponent_ratio = unique / total if total else 1
        opponent_penalty = penalty ** (total - unique) if unique != total else 1
        return opponent_ratio * opponent_penalty

    @staticmethod
    def _get_unique_and_total(items: list[int]) -> tuple[int, int]:
        """Get the count of unique and total items."""
        unique_items = set(items)
        return len(unique_items), len(items)
