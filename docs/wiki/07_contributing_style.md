# Contributing & Style Guide

## Code style
- Prefer small, typed functions over mega-functions
- Avoid global mutable state
- All file I/O must be scoped under a run directory
- Operators should be easy to mock and unit-test

## Logging
- Use structured logging (JSON or key-value)
- No bare prints in core modules
- Operators log: inputs hash, outputs hash, execution time, tool version (if any)

## Errors
- Raise structured exceptions with context (operator name, file path, metric name)
- Do not swallow simulator errors; wrap them with actionable messages

## Documentation
- New artifact/operator must be documented in this wiki (or linked)
- Keep docs short but precise
