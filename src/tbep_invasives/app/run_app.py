from __future__ import annotations

from pathlib import Path

try:
    from tbep_invasives.paths import load_config
except ImportError:  # pragma: no cover
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from tbep_invasives.paths import load_config

from tbep_invasives.app.dashboard import create_app


def main() -> None:
    cfg = load_config()
    app = create_app(cfg)

    host = str(cfg.get("app", {}).get("host", "127.0.0.1"))
    port = int(cfg.get("app", {}).get("port", 8430))
    debug = bool(cfg.get("app", {}).get("debug", True))

    # Dash v3+: app.run(); older Dash: app.run_server()
    if callable(getattr(app, "run", None)):
        app.run(host=host, port=port, debug=debug, use_reloader=False)
    else:
        app.run_server(host=host, port=port, debug=debug, use_reloader=False)



if __name__ == "__main__":
    main()
