"""Main entry point for the fll-scheduler-ga package."""

from fll_scheduler_ga.entrypoints.cli.cli import app


def main() -> None:
    """Run the CLI application."""
    app()


if __name__ == "__main__":
    main()
