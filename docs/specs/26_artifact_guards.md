# 26. Artifact: Guard Checks and Reports

Guards produce structured results that can be logged, audited, and reused across strategies.

## 26.1 GuardCheck

**Type:** `eesizer_core.contracts.guards.GuardCheck`

Fields:
- `name: str` — unique identifier for the guard (e.g., `"patch_guard"`).
- `ok: bool` — whether the guard passed.
- `severity: "hard" | "soft"` — hard failures must block progress; soft failures are warnings.
- `reasons: tuple[str, ...]` — human-readable reasons for failures (empty when `ok=True`).
- `data: dict[str, Any]` — optional structured data for thresholds, detected values, or diagnostics.

Rules:
- `reasons` MUST be serializable and stable enough for auditing.
- `data` MUST be JSON-serializable.

## 26.2 GuardReport

**Type:** `eesizer_core.contracts.guards.GuardReport`

Fields:
- `checks: tuple[GuardCheck, ...]`
- `ok: bool` — `True` if and only if all **hard** checks are ok.
- `hard_fails: tuple[GuardCheck, ...]` — derived subset where `severity=="hard"` and `ok=False`.
- `soft_fails: tuple[GuardCheck, ...]` — derived subset where `severity=="soft"` and `ok=False`.

Rules:
- `ok` is derived from `checks` (hard checks only).
- `hard_fails` / `soft_fails` MUST be consistent with `checks`.
