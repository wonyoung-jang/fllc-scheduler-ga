"""Configuration for the tournament scheduler GA."""

import logging
import math
from configparser import ConfigParser
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..data_model.time import HHMM_FMT

logger = logging.getLogger(__name__)

type RoundType = str


@dataclass(slots=True)
class Round:
    """Representation of a round in the FLL tournament."""

    round_type: RoundType
    rounds_per_team: int
    teams_per_round: int
    start_time: datetime
    stop_time: datetime
    duration_minutes: timedelta
    num_locations: int
    num_teams: int
    num_slots: int = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to calculate the number of slots."""
        self.num_slots = self.get_num_slots()

    def get_num_slots(self) -> int:
        """Get the number of slots available for this round."""
        total_num_teams = self.num_teams * self.rounds_per_team
        slots_per_timeslot = self.num_locations * self.teams_per_round

        if slots_per_timeslot == 0:
            return 0

        minimum_slots = math.ceil(total_num_teams / slots_per_timeslot)

        if self.stop_time:
            total_available = self.stop_time - self.start_time
            slots_in_window = int(total_available / self.duration_minutes)
            return max(minimum_slots, slots_in_window)

        return minimum_slots


@dataclass(slots=True, frozen=True)
class TournamentConfig:
    """Configuration for the tournament."""

    num_teams: int
    rounds: list[Round]
    round_requirements: dict[RoundType, int]
    total_slots: int
    unique_opponents_possible: bool

    def __str__(self) -> str:
        """Represent the TournamentConfig."""
        rounds_str = ", ".join(f"{r.round_type}" for r in sorted(self.rounds, key=lambda x: x.start_time))
        round_reqs_str = ", ".join(f"{k}: {v}" for k, v in self.round_requirements.items())

        return (
            f"TournamentConfig:\n"
            f"\tNumber of Teams: {self.num_teams}\n"
            f"\tRound Types: {rounds_str}\n"
            f"\tRound Requirements: {round_reqs_str}\n"
            f"\tTotal Slots: {self.total_slots}\n"
            f"\tUnique Opponents Possible: {self.unique_opponents_possible}"
        )


def get_config_parser(path: Path | None = None) -> ConfigParser:
    """Get a ConfigParser instance for the given config file path.

    Args:
        path (Path | None): Path to the configuration file.

    Returns:
        ConfigParser: The configured ConfigParser instance.

    """
    if path is None:
        try:
            path = Path("fll_scheduler_ga/config.ini").resolve()
        except FileNotFoundError:
            logger.exception("Configuration file not found. Please provide a valid path.")

    parser = ConfigParser(inline_comment_prefixes=("#", ";"))
    parser.read(path)
    logger.debug("Configuration file loaded from %s", path)
    return parser


def parse_rounds(parser: ConfigParser) -> tuple[list[Round], dict[RoundType, int], int]:
    """Parse and return a list of Round objects from the configuration.

    Args:
        parser (ConfigParser): The ConfigParser instance with tournament configuration.

    Returns:
        list[Round]: A list of Round objects parsed from the configuration.
        dict[RoundType, int]: A dictionary mapping round types to the number of rounds per team.
        int: The total number of teams in the tournament.

    """
    num_teams = parser["DEFAULT"].getint("num_teams")
    parsed_rounds: list[Round] = []
    round_reqs = {}

    for section in parser.sections():
        if not section.startswith("round"):
            continue

        r_type = parser[section].get("round_type")
        r_per_team = parser[section].getint("rounds_per_team")
        round_reqs[r_type] = r_per_team

        if start_time := parser[section].get("start_time", ""):
            start_time = datetime.strptime(start_time, HHMM_FMT).replace(tzinfo=UTC)

        if stop_time := parser[section].get("stop_time", ""):
            stop_time = datetime.strptime(stop_time, HHMM_FMT).replace(tzinfo=UTC)

        parsed_rounds.append(
            Round(
                r_type,
                r_per_team,
                parser[section].getint("teams_per_round"),
                start_time,
                stop_time,
                timedelta(minutes=parser[section].getint("duration_minutes")),
                parser[section].getint("num_locations"),
                num_teams,
            )
        )

    if not parsed_rounds:
        msg = "No rounds defined in the configuration file."
        raise ValueError(msg)

    return parsed_rounds, round_reqs, num_teams


@dataclass(slots=True)
class OperatorConfig:
    """Configuration for the genetic algorithm operators."""

    selection_types: list[str]
    crossover_types: list[str]
    crossover_ks: list[int]
    mutation_types: list[str]


def parse_operator_config(parser: ConfigParser) -> OperatorConfig:
    """Parse and return the operator configuration from the provided ConfigParser.

    Args:
        parser (ConfigParser): The ConfigParser instance with operator configuration.

    Returns:
        OperatorConfig: The parsed operator configuration.

    """
    if "genetic.operator.selection" not in parser:
        msg = "No selection configuration section '[genetic.operator.selection]' found."
        raise ValueError(msg)

    parser_selections = parser["genetic.operator.selection"].get("selection_types", "")
    selection_types = [s.strip() for s in parser_selections.split(",") if s.strip()]

    if not selection_types:
        logger.warning("No selection types enabled in the configuration. Selection will not occur.")
        return None

    if "genetic.operator.crossover" not in parser:
        msg = "No crossover configuration section '[genetic.operator.crossover]' found."
        raise ValueError(msg)

    parser_crossovers = parser["genetic.operator.crossover"].get("crossover_types", "")
    crossover_types = [c.strip() for c in parser_crossovers.split(",") if c.strip()]

    if not crossover_types:
        logger.warning("No crossover types enabled in the configuration. Crossover will not occur.")
        return None

    parser_crossover_ks = parser["genetic.operator.crossover"].get("crossover_ks", "")
    crossover_ks = [int(k) for k in parser_crossover_ks.split(",") if k.strip()]

    if "genetic.operator.mutation" not in parser:
        msg = "No mutation configuration section '[genetic.operator.mutation]' found."
        raise ValueError(msg)

    parser_mutations = parser["genetic.operator.mutation"].get("mutation_types", "")
    mutation_types = [m.strip() for m in parser_mutations.split(",") if m.strip()]

    if not mutation_types:
        logger.warning("No mutation types enabled in the configuration. Mutation will not occur.")
        return None

    return OperatorConfig(
        selection_types,
        crossover_types,
        crossover_ks,
        mutation_types,
    )


def load_tournament_config(path: Path | None = None) -> tuple[TournamentConfig, ConfigParser, OperatorConfig]:
    """Load, parse, and return the tournament configuration.

    Args:
        path (Path | None): The path to the configuration file.

    Returns:
        TournamentConfig: The parsed tournament configuration.
        ConfigParser: The ConfigParser instance used to read the configuration.

    """
    parser = get_config_parser(path)
    parsed_rounds, round_reqs, num_teams = parse_rounds(parser)
    operator_config = parse_operator_config(parser)
    all_rounds_per_team = [r.rounds_per_team for r in parsed_rounds]
    total_slots = sum(num_teams * rpt for rpt in all_rounds_per_team)
    unique_opponents_possible = 1 <= max(all_rounds_per_team) <= num_teams - 1

    config = TournamentConfig(
        num_teams,
        parsed_rounds,
        round_reqs,
        total_slots,
        unique_opponents_possible,
    )

    logger.debug("Loaded tournament configuration: %s", config)

    return config, parser, operator_config
