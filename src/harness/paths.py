"""Where Harness keeps its own data.

The user's project stays clean: only the **working memory** is project-scoped
and lives directly at ``<workdir>/WORK.md``. Everything global (database, tool
manifest, permissions, skills, traces, long-term memory) lives in a single
per-user data directory, ``~/.OAgent``.

Layout::

    ~/.OAgent/                      global data directory
        harness.db                  sessions / messages
        capabilities.json           tool + skill manifest
        permissions.json            permission grants
        skills/<name>/SKILL.md      installed skills
        traces/<session>.jsonl      traces
        MEMORY.md                   long-term memory (C7)
        extract.md                  extraction principle (C7)

    <workdir>/WORK.md               working memory (project-scoped, C4)

``<home>`` resolution order (first hit wins):

1. ``HARNESS_HOME`` -- explicit override; also lets tests redirect everything
   to a temporary directory.
2. ``~/.OAgent`` -- the default on every platform.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "APP_DIR_NAME",
    "ENV_HARNESS_HOME",
    "LEGACY_DIR_NAME",
    "harness_home",
    "project_agent_dir",
    "capabilities_path",
    "permissions_path",
    "skills_root",
    "session_db_path",
    "work_memory_path",
    "traces_dir",
    "long_term_memory_path",
    "extract_principle_path",
]

#: Application directory name: ``~/.OAgent``.
APP_DIR_NAME = ".OAgent"

#: Environment variable that overrides the data directory.
ENV_HARNESS_HOME = "HARNESS_HOME"

#: Legacy directory name: recognised as a read-only source for a manifest that
#: a repository committed, never created for Harness state.
LEGACY_DIR_NAME = ".agent"

#: Working-memory file name, directly in the working directory (C4).
WORK_MEMORY_FILENAME = "WORK.md"


def _platform_default() -> Path:
    """Data directory: ``~/.OAgent`` on every platform."""
    return Path.home() / APP_DIR_NAME


_home_cache: Path | None = None


def harness_home(*, create: bool = True) -> Path:
    """Return the data directory (shared by every project on this machine).

    Args:
        create: Create the directory when missing. Failures fall back to the
            home-directory location instead of raising, because storage must
            still work in restricted environments.
    """
    global _home_cache
    override = os.environ.get(ENV_HARNESS_HOME)
    candidates = [Path(override).expanduser()] if override else [_platform_default()]

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if create:
            try:
                resolved.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
        if not create or resolved.is_dir():
            _home_cache = resolved
            return resolved
    # Nothing is creatable: return the first candidate so callers surface a
    # concrete error instead of writing into the project directory.
    return candidates[0]


def project_agent_dir(workdir: str | Path) -> Path | None:
    """Return a project-local ``.agent`` directory when it already exists.

    Read-only compatibility hook: manifests committed with a repository stay
    usable, but we never create the directory inside a user's project.
    """
    candidate = Path(workdir).resolve() / LEGACY_DIR_NAME
    return candidate if candidate.is_dir() else None


def capabilities_path(workdir: str | Path) -> Path:
    """Manifest file: project-local ``.agent/capabilities.json`` if present, else ``<home>``."""
    legacy = project_agent_dir(workdir)
    if legacy is not None and (legacy / "capabilities.json").is_file():
        return legacy / "capabilities.json"
    return harness_home() / "capabilities.json"


def permissions_path(workdir: str | Path) -> Path:
    """Permission file: project-local override first, else ``<home>/permissions.json``."""
    legacy = project_agent_dir(workdir)
    if legacy is not None and (legacy / "permissions.json").is_file():
        return legacy / "permissions.json"
    return harness_home() / "permissions.json"


def skills_root(workdir: str | Path) -> Path:
    """Skill root: project-local ``.agent/skills`` if present, else ``<home>/skills``."""
    legacy = project_agent_dir(workdir)
    if legacy is not None and (legacy / "skills").is_dir():
        return legacy / "skills"
    return harness_home() / "skills"


def session_db_path() -> Path:
    """SQLite file holding every session of every project."""
    return harness_home() / "harness.db"


def work_memory_path(workdir: str | Path) -> Path:
    """Working memory: ``<workdir>/WORK.md`` (project-scoped, C4).

    Working memory describes *this project*, so it lives next to the project
    files. It is written **directly** into the working directory -- no
    ``.agent`` folder is created.
    """
    return Path(workdir).resolve() / WORK_MEMORY_FILENAME


def traces_dir() -> Path:
    """Directory holding one JSONL trace per session."""
    return harness_home() / "traces"


def long_term_memory_path() -> Path:
    """Long-term memory (one file for the whole installation, C7)."""
    return harness_home() / "MEMORY.md"


def extract_principle_path() -> Path:
    """Extraction principle (maintained by the user)."""
    return harness_home() / "extract.md"
