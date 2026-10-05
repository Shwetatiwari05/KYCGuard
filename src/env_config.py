"""Environment/secret loading for local development.

Cloud Run (and any container/CI runtime) injects `GEMINI_API_KEY` / `GROQ_API_KEY`
as real environment variables, so those always win. The `.env` files below are a
local-development convenience only: they are loaded with ``override=False`` and
are silently skipped when absent (as they are in a deployed image).
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# src/env_config.py -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Anchored to PROJECT_ROOT (not the CWD) so the app behaves the same whether it
# is started from the project root or from another directory.
DOTENV_FILES = (
    PROJECT_ROOT / "backend" / ".env",
    PROJECT_ROOT / ".env",
)

_loaded = False


def load_project_env(force: bool = False) -> list[Path]:
    """Load local `.env` files into `os.environ` without overriding real env vars.

    Returns the files that were found and applied. Safe to call repeatedly.
    """
    global _loaded
    if _loaded and not force:
        return []

    loaded: list[Path] = []
    for path in DOTENV_FILES:
        if path.is_file():
            load_dotenv(path, override=False)
            loaded.append(path)

    _loaded = True
    return loaded