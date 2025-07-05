"""Configuration for the tournament scheduler GA."""

import logging
import math
from configparser import ConfigParser
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from pathlib import Path

from ..data_model.location import Room, Table

logger = logging.getLogger(__name__)


class RoundType(StrEnum):
    """Enumeration for different types of rounds in the FLL Scheduler GA."""

    JUDGING = "Judging"
    PRACTICE = "Practice"
    TABLE = "Table"


@dataclass(slots=True, frozen=True)
class Round:
    """Representation of a round in the FLL tournament."""

    round_type: RoundType
    rounds_per_team: int
    teams_per_round: int
    start_time: str
    duration_minutes: timedelta
    num_locations: int
    num_teams: int

    @property
    def num_slots(self) -> int:
        """Calculate the number of slots needed for a given Round configuration."""
        total_num_teams = self.num_teams * self.rounds_per_team
        slots_per_timeslot = self.num_locations * self.teams_per_round

        if slots_per_timeslot == 0:
            return 0

        return math.ceil(total_num_teams / slots_per_timeslot)


@dataclass(slots=True, frozen=True)
class TournamentConfig:
    """Configuration for the tournament."""

    num_teams: int
    rounds: frozenset[Round]
    round_requirements: dict[RoundType, int]

    def __str__(self) -> str:
        """Represent the TournamentConfig."""
        rounds_str = ", ".join(f"{r.round_type}" for r in sorted(self.rounds, key=lambda x: x.round_type))
        round_reqs_str = ", ".join(f"{k}: {v}" for k, v in self.round_requirements.items())
        return (
            f"\nTournamentConfig:\n"
            f"\tnum_teams: {self.num_teams}\n"
            f"\trounds: {rounds_str}\n"
            f"\tround_requirements: {round_reqs_str}"
        )


def load_tournament_config(config_path: str) -> TournamentConfig:
    """Load, parse, and return the tournament configuration.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        TournamentConfig: The parsed tournament configuration.

    """
    parser = ConfigParser()

    if isinstance(config_path, str):
        config_path = Path(config_path)

    if not config_path.is_file():
        msg = f"Config file not found at {config_path}"
        raise FileNotFoundError(msg)

    parser.read(config_path)

    num_teams = parser["DEFAULT"].getint("num_teams")

    parsed_rounds = set()
    round_reqs = {}

    for section in parser.sections():
        if section.startswith("round"):
            r_type = RoundType(parser[section]["round_type"])
            r_per_team = parser[section].getint("rounds_per_team")
            round_reqs[r_type] = r_per_team
            parsed_rounds.add(
                Round(
                    round_type=r_type,
                    rounds_per_team=r_per_team,
                    teams_per_round=parser[section].getint("teams_per_round"),
                    start_time=parser[section]["start_time"],
                    duration_minutes=timedelta(minutes=parser[section].getint("duration_minutes")),
                    num_locations=parser[section].getint("num_locations"),
                    num_teams=num_teams,
                )
            )

    if not parsed_rounds:
        msg = "No rounds defined in the configuration file."
        raise ValueError(msg)

    return TournamentConfig(
        num_teams=num_teams,
        rounds=frozenset(parsed_rounds),
        round_requirements=round_reqs,
    )


def get_location_type(round_type: RoundType) -> Room | Table:
    """Get the location type based on the round type.

    Args:
        round_type (RoundType): The type of the round.

    Returns:
        Room | Table: The corresponding location type for the round.

    """
    return {
        RoundType.JUDGING: Room,
        RoundType.PRACTICE: Table,
        RoundType.TABLE: Table,
    }.get(round_type)
