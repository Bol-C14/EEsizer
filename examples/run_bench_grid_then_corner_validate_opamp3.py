from __future__ import annotations

import sys

from run_bench_grid_then_corner_validate import main


if __name__ == "__main__":
    sys.argv[1:1] = ["--bench", "opamp3"]
    main()

