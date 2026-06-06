"""Main entry point for the fll-scheduler-ga package."""

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from fll_scheduler_ga.constants import CONFIG_FILE_DEFAULT

CONFIG_DIR = Path(".configs/").resolve()
CONFIG_FILE_ACTIVE = Path(".configs/_active_config.txt").resolve()


@dataclass(slots=True)
class ConfigManager:
    """Manages selection and maintenance of configuration files."""

    dirpth: Path = CONFIG_DIR
    default: Path = CONFIG_FILE_DEFAULT
    active: Path = CONFIG_FILE_ACTIVE
    available: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Ensure directory structure exists and is populated."""
        self.dirpth.mkdir(parents=True, exist_ok=True)
        self.refresh_list()
        if not self.available:
            if not self.default.exists():
                msg = f"Critical: Default template not found at {self.default}"
                raise FileNotFoundError(msg)
            dest = self.dirpth / "default.json"
            shutil.copy(self.default, dest)
            print(f"Initialized configuration directory with {dest.name}")
            self.refresh_list()
            self.set_active_config("0")

    def refresh_list(self) -> None:
        """Refresh the internal list of JSON files."""
        self.available = sorted((f for f in self.dirpth.iterdir() if f.suffix == ".json"), key=lambda f: f.name)

    def get_active_config(self) -> Path:
        """Retrieve the last used configuration if it still exists."""
        if not self.active.exists():
            return Path()
        try:
            active = self.active.read_text(encoding="utf-8").strip()
            active_path = Path(active)
            for path in self.available:
                if path.name == active_path.name:
                    return path
        except OSError:
            return Path()
        return Path()

    def list_configs(self) -> None:
        """Print available configurations with indices."""
        active = self.get_active_config()
        print(f"\nAvailable Configurations in '{self.dirpth}':")
        print("-" * 50)
        for idx, path in enumerate(self.available):
            marker = "*" if active and path.name == active.name else " "
            print(f"{marker}[{idx}] {path.name}")
        print("-" * 50)
        print("(*) = active configuration")

    def get_config(self, identifier: str | None) -> Path:
        """Select a config based on index (int) or filename (str)."""
        if not self.available:
            msg = "No configuration files found."
            raise FileNotFoundError(msg)
        # Default to active config
        if identifier is None:
            active = self.get_active_config()
            if active:
                return active
            # Fallback to first available
            print(f"No last active config found. Using first available: {self.available[0].name}")
            return self.available[0]
        # Select by index
        if identifier.isdigit():
            idx = int(identifier)
            if 0 <= idx < len(self.available):
                return self.available[idx]
            msg = f"Index {idx} out of range 0-{len(self.available) - 1}"
            raise ValueError(msg)
        # Select by name, stem or path
        identifier_path = Path(identifier).resolve()
        for path in self.available:
            if path == identifier_path:
                return path
            if identifier in (path.name, path.stem):
                return path
        # If the path exists but is not in the config directory, import it
        if identifier_path.exists() and identifier_path.suffix == ".json":
            self.add_config(identifier_path, identifier_path.name)
            self.refresh_list()
            return self.get_config(identifier_path.name)
        msg = f"Configuration '{identifier}' not found."
        raise FileNotFoundError(msg)

    def set_active_config(self, identifier: str) -> None:
        """Set the active configuration file."""
        selected = self.get_config(identifier)
        with self.active.open("w") as f:
            f.write(str(selected.resolve()))

    def add_config(self, source_path: Path | str, dest_name: str = "") -> None:
        """Import a new configuration file."""
        src = Path(source_path) if isinstance(source_path, str) else source_path
        if not src.exists():
            msg = f"Source file {src} does not exist."
            raise FileNotFoundError(msg)
        # Determine destination filename
        if dest_name:
            new_filename = dest_name
            if not new_filename.endswith(".json"):
                new_filename += ".json"
        else:
            new_filename = src.name
        dest = self.dirpth / new_filename
        if dest.exists():
            user_input = input(f"File {dest.name} already exists. Overwrite? (y/n): ")
            if user_input.lower() not in ("y", "yes"):
                print("Aborted adding configuration.")
                return
            print(f"Overwriting existing config {dest.name}")
        shutil.copy(src, dest)
        self.refresh_list()
        print(f"Successfully added: {dest.name}")
