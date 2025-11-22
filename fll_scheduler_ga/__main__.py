"""Main entry point for the fll-scheduler-ga package."""

from __future__ import annotations

import logging
import sys

from fll_scheduler_ga.config._config_manager import ConfigManager, parse_config_args
from fll_scheduler_ga.config.app_config import AppConfig
from fll_scheduler_ga.engine import init_logging, run_ga_instance

logger = logging.getLogger(__name__)


def main_cli() -> None:
    """Run the fll-scheduler-ga application from the command line interface."""
    try:
        args = parse_config_args()
        manager = ConfigManager()

        # Management Commands (Exits after execution)
        if args.list:
            manager.list_configs()
            sys.exit(0)

        if args.add:
            if len(args.add) > 2:
                print("ERROR: Too many arguments for --add. Provide PATH [NAME].")
                sys.exit(1)
            src = args.add[0]
            dest = args.add[1] if len(args.add) == 2 else None
            manager.add_config(src, dest)
            manager.list_configs()
            sys.exit(0)

        if args.remove:
            manager.remove_config(args.remove)
            manager.list_configs()
            sys.exit(0)

        if args.set:
            selected = manager.get_config(args.set)
            with manager.active_config.open("w") as f:
                f.write(selected.name)
            print(f"Set active configuration to {selected.name}")
            sys.exit(0)

        # Main Execution
        active_config_path = manager.get_config(args.config)
        app_config = AppConfig.build(active_config_path)
        init_logging(app_config)
        run_ga_instance(app_config)
        logger.debug("Starting GA with configuration: %s", active_config_path.name)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except Exception:
        logger.exception("GA process failed unexpectedly.")
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
