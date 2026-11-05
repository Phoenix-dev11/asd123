"""Find the Zils checkout so the kit calls the project's real code instead of copying it.

Set ZILS_REPO, or keep the kit beside `zils/`, or use the installer's ~/.zils/zils.
"""

import os
import sys
from pathlib import Path


def add_zils():
    here = Path(__file__).resolve().parent
    for candidate in (
        os.environ.get("ZILS_REPO"),
        here.parent / "zils",
        Path.home() / ".zils" / "zils",
    ):
        if candidate and (Path(candidate) / "zils" / "__init__.py").exists():
            path = str(Path(candidate).resolve())
            if path not in sys.path:
                sys.path.insert(0, path)
            return path
    raise SystemExit("Zils checkout not found: set ZILS_REPO")


POOL_MODELS = {
    "d3-lite": "d3-lite-0.8b-v1",
    "d3-mini": "d3-mini-4.5b-v1",
    "d3-flash": "d3-flash-8.4b-v1",
    "d3": "d3-ultra-26b-v1",
}
