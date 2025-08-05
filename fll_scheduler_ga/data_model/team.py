"""Team data model for FLL Scheduler GA."""

from dataclasses import dataclass, field
from logging import getLogger

from ..config.config import RoundType, TournamentConfig
from ..data_model.event import Event
from .time import TimeSlot

logger = getLogger(__name__)


type TeamMap = dict[int, Team]


@dataclass(slots=True, frozen=True)
class TeamInfo:
    """Data model for team information in the FLL Scheduler GA."""

    identity: int


@dataclass(slots=True)
class Team:
    """Data model for a team in the FLL Scheduler GA."""

    info: TeamInfo
    identity: int
    roundreqs: dict[RoundType, int]
    fitness: tuple[float, ...] = field(init=False, repr=False)
    events: list[int] = field(default_factory=list, repr=False)
    timeslots: list[TimeSlot] = field(default_factory=list, repr=False)
    opponents: list[int] = field(default_factory=list, repr=False)
    tables: list[int] = field(default_factory=list, repr=False)

    def __hash__(self) -> int:
        """Hash function for the team based on its identity."""
        return self.identity

    def rounds_needed(self) -> bool:
        """Get the total number of rounds still needed for the team."""
        return sum(self.roundreqs.values()) > 0

    def needs_round(self, round_type: RoundType) -> bool:
        """Check if the team still needs to participate in a given round type."""
        return self.roundreqs[round_type] > 0

    def switch_event(self, old_event: Event, new_event: Event) -> None:
        """Switch an event for the team."""
        self.remove_event(old_event)
        self.add_event(new_event)

    def remove_event(self, event: Event) -> None:
        """Unbook a team from an event."""
        self.roundreqs[event.roundtype] += 1
        self.events.remove(event.identity)
        self.timeslots.remove(event.timeslot)
        if event.paired:
            self.tables.remove(event.location)

    def add_event(self, event: Event) -> None:
        """Book a team for an event."""
        self.roundreqs[event.roundtype] -= 1
        self.events.append(event.identity)
        self.timeslots.append(event.timeslot)
        if event.paired:
            self.tables.append(event.location)

    def switch_opponent(self, old_opponent: "Team", new_opponent: "Team") -> None:
        """Switch the opponent for a given event."""
        self.remove_opponent(old_opponent)
        self.add_opponent(new_opponent)

    def remove_opponent(self, opponent: "Team") -> None:
        """Remove an opponent for a given event."""
        self.opponents.remove(opponent.identity)

    def add_opponent(self, opponent: "Team") -> None:
        """Add an opponent for a given event."""
        self.opponents.append(opponent.identity)

    def conflicts(self, new_event: Event) -> bool:
        """Check if adding a new event would cause a time conflict.

        Args:
            new_event (Event): The new event to check for conflicts.

        Returns:
            bool: True if there is a conflict, False otherwise.

        """
        _events = self.events
        if new_event.identity in _events:
            return True

        if not (potential_conflicts := new_event.conflicts):
            return False

        return any(existing_event_id in potential_conflicts for existing_event_id in _events)

    def break_time_key(self) -> frozenset[int]:
        """Get a key for the break time cache based on the team's events."""
        return frozenset(self.timeslots)

    def table_consistency_key(self) -> int:
        """Get a key for the table consistency cache based on the team's events."""
        return len(set(self.tables))

    def opponent_variety_key(self) -> int:
        """Get a key for the opponent variety cache based on the team's events."""
        return len(set(self.opponents))


@dataclass(slots=True)
class TeamFactory:
    """Factory class to create Team instances."""

    config: TournamentConfig
    _base_teams_info: frozenset[TeamInfo] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self._base_teams_info = frozenset(TeamInfo(i) for i in range(1, self.config.num_teams + 1))

    def build(self) -> TeamMap:
        """Create a mapping of team identities to Team instances.

        Returns:
            TeamMap: A mapping of team identities to Team instances.

        """
        return {
            info.identity: Team(
                info=info,
                identity=info.identity,
                roundreqs=self.config.round_requirements.copy(),
            )
            for info in self._base_teams_info
        }
