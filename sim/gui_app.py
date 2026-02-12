"""Deprecated Streamlit entrypoint.

Use `python -m sim.desktop_gui` for the native desktop GUI.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit("Deprecated: use `python -m sim.desktop_gui`.")


if __name__ == "__main__":
    main()
