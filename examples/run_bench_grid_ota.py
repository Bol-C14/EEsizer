from __future__ import annotations

import sys

from run_grid_search_bench import main


if __name__ == "__main__":
    sys.argv[1:1] = ["--bench", "ota"]
    main()

