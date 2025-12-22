from __future__ import annotations

import logging
from typing import Any, Dict

from tbep_invasives.paths import load_config
from tbep_invasives.pipeline.preflight import preflight
from tbep_invasives.steps.download_invasives import run as run_download
from tbep_invasives.steps.flam_overlay import run as run_flam
from tbep_invasives.steps.hex_enrichment import run as run_hex
from tbep_invasives.steps.report_cards import run as run_cards


logger = logging.getLogger(__name__)


def run(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quarterly pipeline:
      1) download invasives
      2) build FLAM overlay
      3) enrich hexbins
      4) generate report cards
    """
    results: Dict[str, Any] = {}
    
    preflight(cfg, mode="quarterly")

    logger.info("STEP 1/4: download invasives")
    results["download_invasives"] = run_download(cfg)

    logger.info("STEP 2/4: build FLAM overlay")
    results["flam_overlay"] = run_flam(cfg)

    logger.info("STEP 3/4: enrich hexbins")
    results["hex_enrichment"] = run_hex(cfg)

    logger.info("STEP 4/4: report cards")
    results["report_cards"] = run_cards(cfg)

    logger.info("Pipeline complete.")
    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cfg = load_config()
    run(cfg)


if __name__ == "__main__":
    main()
