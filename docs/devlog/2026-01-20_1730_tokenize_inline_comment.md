# 2026-01-20 17:30 — Tokenize Inline Comments

## Scope

- Treat semicolon (`;`) as an inline comment delimiter during tokenization.

## Files touched

- `src/eesizer_core/domain/spice/tokenize.py`
- `tests/test_step3_patch_domain.py`

## Rationale & notes

- Keeps inline comments intact in stored lines while ensuring patch/apply uses the same
  token boundaries as the parser (which already strips inline comments).
- Fixes cases like `w=1u;comment` where token spans previously included the comment.

## Migration / compatibility

- Tokenization now ignores text after the first `;` in a line, matching existing parsing behavior.
