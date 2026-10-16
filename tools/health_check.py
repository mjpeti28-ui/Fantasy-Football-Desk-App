"""Print the input health check for the configured rankings/projections.

    python tools/health_check.py            # uses RANKINGS_CSV / PROJECTIONS_CSV from .env.local
    python tools/health_check.py --no-espn  # skip the ESPN injury/availability checks

Exit code 1 if any ERROR-level finding, so it can gate other scripts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env import load_local_env  # noqa: E402

load_local_env()

from health import run_health_check, summarize  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rankings")
    parser.add_argument("--projections")
    parser.add_argument("--no-espn", action="store_true")
    args = parser.parse_args(argv)
    findings = run_health_check(args.rankings, args.projections, use_espn=not args.no_espn)
    print(summarize(findings))
    return 1 if any(f.level == "ERROR" for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
