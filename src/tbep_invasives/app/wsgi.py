from tbep_invasives.paths import load_config
from tbep_invasives.app.dashboard import create_app

cfg = load_config()
dash_app = create_app(cfg)

# Gunicorn looks for "server" by convention
server = dash_app.server
