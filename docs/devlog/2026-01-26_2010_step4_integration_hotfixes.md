# 2026-01-26 2010 Step4 integration hotfixes

- Deck builder now sets `wr_vecnames` and `wr_singlescale` to stabilize wrdata headers/scale layout.
- OTA/opamp3 AC stop_hz increased to 1e8 to avoid unity-crossing missing in baseline metrics.
- Patch loop evaluation pipeline now honors `sim_plan` overrides from spec/config across baseline + attempts (grid/corner/patch loop).
- Updated docs to reflect wrdata header behavior and plot/insight artifacts.

Notes:
- This unblocks ngspice integration tests and benchmark grid search runs that rely on vout/vinp/vinn probes.
