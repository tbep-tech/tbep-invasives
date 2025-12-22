from __future__ import annotations

import os
from pathlib import Path

from tbep_invasives.paths import load_config
from tbep_invasives.app.dashboard import create_app

# Allow overriding config file path in Docker/CI:
#   TBEP_CONFIG=/app/config/settings_ci.yaml
_config_env = os.environ.get("TBEP_CONFIG", "").strip()

# If your load_config() already uses TBEP_CONFIG internally, this is redundant but harmless.
# If load_config() accepts a path argument, pass it; otherwise just call load_config().
try:
    cfg = load_config(_config_env) if _config_env else load_config()
except TypeError:
    cfg = load_config()

dash_app = create_app(cfg)

# Gunicorn entrypoint
server = dash_app.server
