#!/usr/bin/env python3
"""Server entry point for OpenEnv multi-mode deployment checks."""

from __future__ import annotations

from healthcheck import main as healthcheck_main


def main() -> None:
    """Launch the lightweight health server."""
    healthcheck_main()


if __name__ == "__main__":
    main()
