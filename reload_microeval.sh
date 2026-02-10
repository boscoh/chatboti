#!/bin/bash
# Helper script to reload microeval after making changes
# Usage: ./reload_microeval.sh

cd "$(dirname "$0")"
echo "Reinstalling microeval to pick up changes..."
uv sync --reinstall-package microeval
echo "Done! Changes should now be available."
