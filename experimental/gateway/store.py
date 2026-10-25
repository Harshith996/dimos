# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import yaml

from dimos.constants import RECORDINGS_DIR, STATE_DIR
from dimos.utils.logging_config import setup_logger

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logger = setup_logger()


def gateway_dir() -> Path:
    return STATE_DIR / "gateway"


def logs_dir() -> Path:
    return gateway_dir() / "logs"


def app_env() -> dict[str, Any]:
    """`DIMOS_APP`, as Desktop started this gateway with it: the home it belongs to, and the config overrides for this
    component and nothing else. One variable carrying what it is told, instead of a variable per setting.

    Unusable JSON is said out loud and then ignored: an override is a debugging aid, and refusing to start over one
    would be worse than starting without it.
    """
    text = os.environ.get("DIMOS_APP")
    if not text:
        return {}
    try:
        loaded = json.loads(text)
    except ValueError as error:
        logger.warning(
            "DIMOS_APP is not usable, so its home and overrides are ignored", error=str(error)
        )
        return {}
    return loaded if isinstance(loaded, dict) else {}


def overrides() -> dict[str, Any]:
    """`DIMOS_APP.overrides`: config paths (`recordings.dir`) this process is to run with instead of the file's.

    Desktop hands each component its own section and strips the rest, so there is nothing here to pass on to a run:
    an override does not propagate, which is what went wrong when `ZENOH_CONNECT` was inherited by every module of
    every launch. Keys are config paths and never environment variable names on purpose, because `GlobalConfig` is a
    pydantic `BaseSettings` with no `env_prefix` and would pick those up by their bare names.
    """
    found = app_env().get("overrides")
    return found if isinstance(found, dict) else {}


def dimos_home() -> Path:
    """The home whose config.yaml this gateway reads: `DIMOS_APP.home`, else `DIMOS_HOME`, else ~/.dimos.

    `DIMOS_HOME` is still read because that is what every Desktop before 0.2.228 set, and because a person running the
    gateway by hand exports it.
    """
    home = app_env().get("home") or os.environ.get("DIMOS_HOME")
    return Path(home) if home else Path.home() / ".dimos"


def config_file() -> Path:
    """The shared `config.yaml` itself, which may not exist."""
    return dimos_home() / "config.yaml"


def setting(section: str, key: str) -> Any:
    """A value from the shared `$DIMOS_HOME/config.yaml` (default ~/.dimos/config.yaml), None when unset.

    An override for that path (`DIMOS_APP.overrides`) wins over the file, and nothing here ever writes the file, so an
    override is never saved by the gateway either.
    """
    override = overrides().get(f"{section}.{key}")
    if override is not None:
        return None if override == "" else override
    try:
        loaded = yaml.safe_load(config_file().read_text()) or {}
    except (OSError, yaml.YAMLError):
        return None
    value = (loaded.get(section) or {}).get(key) if isinstance(loaded, dict) else None
    return None if value == "" else value


def recordings_dir() -> Path:
    configured = setting("recordings", "dir")
    return Path(configured).expanduser() if configured else RECORDINGS_DIR


def shared_settings() -> dict[str, Any]:
    """What `GET /config` answers, and what a `config` event says has changed: the shared
    `config.yaml`'s settings as the gateway resolves them for each use.

    The settings fixed when the gateway started (`zenoh.namespace`, `dimos-gateway.socket`, where a
    flag wins over the file) are not here: they cannot change while it runs, and the running ones are
    `GET /info`'s. So everything here is read fresh, and a change to any of it is news.
    """
    return {
        "path": str(config_file()),
        "recordingsDir": str(recordings_dir()),
        # provenance: a reader of these cannot otherwise tell a value this process was handed from one the file holds
        "overridden": sorted(overrides()),
    }


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def venv_dir(dimos_dir: Path) -> Path:
    return dimos_dir / ".venv"


def python_for(dimos_dir: Path) -> str:
    python = venv_dir(dimos_dir) / "bin" / "python"
    return str(python) if python.exists() else sys.executable


def dimos_bin(dimos_dir: Path) -> Path:
    return venv_dir(dimos_dir) / "bin" / "dimos"


def checkout_version(dimos_dir: Path) -> tuple[bool, str | None]:
    try:
        project = tomllib.loads((dimos_dir / "pyproject.toml").read_text()).get("project", {})
    except (OSError, tomllib.TOMLDecodeError):
        return False, None
    if project.get("name") != "dimos":
        return False, None
    version = project.get("version")
    return True, str(version) if version else None


def git(dimos_dir: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(["git", "-C", str(dimos_dir), *args], capture_output=True, text=True)
    except OSError:
        return None
    return done.stdout.strip() or None if done.returncode == 0 else None


def info(dimos_dir: Path) -> dict[str, Any]:
    from experimental.gateway.state import API_VERSION

    found, version = checkout_version(dimos_dir)
    return {
        "dir": str(dimos_dir),
        "found": found,
        "installed": found and dimos_bin(dimos_dir).exists(),
        "version": version,
        "api": API_VERSION,
        "commit": git(dimos_dir, "rev-parse", "--short=10", "HEAD"),
        "tag": git(
            dimos_dir, "describe", "--tags", "--exact-match", "--match", "desktop-gateway-v*"
        ),
    }
