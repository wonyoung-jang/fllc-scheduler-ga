"""Team data model for FLL Scheduler GA."""

import math
from collections.abc import Generator
from dataclasses import dataclass, field
from logging import getLogger

from ..config.config import RoundType, TournamentConfig
from ..data_model.event import Event
from .location import Location

logger = getLogger(__name__)


type TeamMap = dict[int, Team]
type Individual = dict[Event, Team]

ZERO_BREAK_PENALTY = 0.0001


@dataclass(slots=True, frozen=True)
class TeamInfo:
    """Data model for team information in the FLL Scheduler GA."""

    identity: int


@dataclass(slots=True)
class TeamFactory:
    """Factory class to create Team instances."""

    config: TournamentConfig
    event_conflict_map: dict[int, set[int]]
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
                event_conflict_map=self.event_conflict_map,
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

    _cached_break_time_score: float | None = field(default=None, init=False, repr=False)
    _cached_opponent_score: float | None = field(default=None, init=False, repr=False)
    _cached_table_score: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to sort events and time slots."""
        self.identity = self.info.identity

    def __hash__(self) -> int:
        """Hash function for the team based on its identity."""
        return hash(self.info.identity)

    def __len__(self) -> int:
        """Get the count a team is counted as."""
        return 1

    def __deepcopy__(self, memo: dict[int, object]) -> "Team":
        """Create a deep copy of the Team instance."""
        if id(self) in memo:
            return memo[id(self)]

        new_team = Team(
            info=self.info,
            round_types=self.round_types.copy(),
            event_conflict_map=self.event_conflict_map.copy(),
            events=self.events.copy(),
            opponents=self.opponents.copy(),
            locations=self.locations.copy(),
        )
        memo[id(self)] = new_team
        return new_team

    def rounds_needed(self) -> int:
        """Get the total number of rounds still needed for the team."""
        return sum(self.round_types.values())

    def needs_round(self, round_type: RoundType) -> int:
        """Check if the team still needs to participate in a given round type."""
        return self.round_types[round_type]

    def has_location(self, event: Event) -> bool:
        """Check if the team has a location for its events."""
        return any(event.location == e.location for e in self.events)

    def switch_event(self, event_to_unbook: Event, new_event: Event) -> None:
        """Switch booking for a team, unbooking the current event and booking the new one."""
        self.remove_event(event_to_unbook)
        self.add_event(new_event)

    def remove_event(self, event: Event) -> None:
        """Unbook a team from an event."""
        self.round_types[event.round_type] += 1
        self.events.remove(event)
        if event.location.teams_per_round == 2:
            self.locations.remove(event.location)

        if self._cached_break_time_score is not None:
            self._cached_break_time_score = None
        if self._cached_table_score is not None:
            self._cached_table_score = None

    def add_event(self, event: Event) -> None:
        """Book a team for an event."""
        if self.round_types[event.round_type] <= 0:
            logger.debug("Team %d already has %s", self.identity, event)
        self.round_types[event.round_type] -= 1
        self.events.append(event)
        if event.location.teams_per_round == 2:
            self.locations.append(event.location)

        if self._cached_break_time_score is not None:
            self._cached_break_time_score = None
        if self._cached_table_score is not None:
            self._cached_table_score = None

    def switch_opponent(self, old_opponent: "Team", new_opponent: "Team") -> None:
        """Switch the opponent for a given event."""
        self.remove_opponent(old_opponent)
        self.add_opponent(new_opponent)

    def remove_opponent(self, opponent: "Team") -> None:
        """Remove an opponent for a given event."""
        self.opponents.remove(opponent.identity)
        if self._cached_opponent_score is not None:
            self._cached_opponent_score = None

    def add_opponent(self, opponent: "Team") -> None:
        """Add an opponent for a given event."""
        self.opponents.append(opponent.identity)
        if self._cached_opponent_score is not None:
            self._cached_opponent_score = None

    def conflicts(self, new_event: Event) -> bool:
        """Check if adding a new event would cause a time conflict.

        Args:
            new_event (Event): The new event to check for conflicts.

        Returns:
            bool: True if there is a conflict, False otherwise.

        """
        if not (potential_conflicts := self.event_conflict_map.get(new_event.identity, set())):
            return False

        if any(e.identity in potential_conflicts for e in self.events):
            return True

        return new_event in self.events

    def _get_break_times(self) -> Generator[int]:
        """Calculate break times between events.

        Returns:
            Generator[int]: The break time in minutes between consecutive events.

        """
        if len(self.events) < 2:
            return

        self.events.sort(key=lambda e: e.timeslot.start)
        for i in range(1, len(self.events)):
            yield (self.events[i].timeslot.start - self.events[i - 1].timeslot.stop).total_seconds() // 60

    def score_break_time(self) -> float:
        """Calculate a score based on the break times between events."""
        if self._cached_break_time_score is not None:
            return self._cached_break_time_score

        break_times = list(self._get_break_times())
        n = len(break_times)
        if n == 0:
            return 1.0

        sum_x = 0.0
        zero_breaks = 0

        for b in break_times:
            if b == 0:
                zero_breaks += 1
            sum_x += b

        if sum_x <= 0:
            return 0.0

        mean_x = sum_x / n
        sum_sq_diff = sum((b - mean_x) ** 2 for b in break_times)
        variance = sum_sq_diff / n
        stdev_x = math.sqrt(variance)
        penalty = ZERO_BREAK_PENALTY**zero_breaks
        coeff_of_variation = stdev_x / mean_x if mean_x > 0 else 0

        self._cached_break_time_score = penalty * (1.0 / (1.0 + coeff_of_variation))
        return self._cached_break_time_score

    def score_opponent_variety(self) -> float:
        """Calculate a score based on the variety of opponents faced."""
        if self._cached_opponent_score is not None:
            return self._cached_opponent_score

        self._cached_opponent_score = len(set(self.opponents)) / len(self.opponents) if self.opponents else 1.0
        return self._cached_opponent_score

    def score_table_consistency(self) -> float:
        """Calculate a score based on the consistency of table assignments."""
        if self._cached_table_score is not None:
            return self._cached_table_score

        self._cached_table_score = 1.0 / len(set(self.locations)) if self.locations else 1.0
        return self._cached_table_score
