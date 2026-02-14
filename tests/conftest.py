from __future__ import annotations

"""Pytest configuration.

This file is imported during *collection*, so it's the right place to set
process-wide environment variables needed for stable imports.
"""

import os
import tempfile

# Force a non-interactive backend in test environments.
os.environ.setdefault("MPLBACKEND", "Agg")

# Isolate matplotlib cache to avoid flaky font-cache locking (stale locks in
# ~/.cache/matplotlib can break collection).
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig-"))

# Headless Qt platform (only matters if PyQt6 is installed and GUI tests run).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Rationale:
# Prevents Matplotlib from writing into ~/.cache/matplotlib and hitting stale lock files.
