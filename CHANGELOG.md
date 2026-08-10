# Changelog

## Unreleased

- Add a modern terminal paid-work preflight with live provider pricing,
  projected calls and costs, recovery-ceiling balance enforcement, and
  interactive acceptance. Payment failures retain safe, actionable billing
  details.
- Make five-view semantic fidelity review opt-in through `--review` or
  `PAPERCLEAN_REVIEW=true`, with explicit unreviewed provenance.
- Discover user-level or repository `.env` and `.keyenv.toml` configuration for
  installed and checkout-based use.
- Recreate documents as pristine, straight, high-resolution pages without
  holes, damaged edges, dirt, stains, discoloration, or blur; sharpen model
  output at the source page size and make deterministic OCR/foreground checks
  tolerant of intentional restoration geometry. Preserve registered authored
  marks and edge microprint while clearing duplicate generated footer content
  and edge-connected scanner artifacts.
- Strengthen full-page and regional prompts around footer sharpness, level
  baselines, and single-instance microprint. Retain sharper generated edge text
  only after exact registered OCR confirms alignment and rejects duplicates.
- Move every generation, repair, review, and retry-feedback prompt into packaged
  Markdown resources, with wheel-level inclusion tests.
- Detect edge microprint and its cleanup regions from each page's OCR distribution
  and registration geometry instead of document-specific coordinates.
- Prevent GPT-5.6-sol review reasoning from exhausting the structured-verdict
  budget by using medium reasoning with a larger completion allowance.

## 0.1.1 - 2026-08-09

## 0.1.0 - 2026-08-09

- Initial PaperClean implementation.
