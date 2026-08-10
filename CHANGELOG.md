# Changelog

## Unreleased

- Add an AgentBridge backend that runs generation and mandatory structured
  five-view verification through an authenticated local Codex CLI. Include capability
  preflight, non-persistent requests, subscription call-count projections,
  observable token accounting, and backend-aware provenance.
- Add a modern terminal paid-work preflight with live provider pricing,
  projected calls and costs, recovery-ceiling balance enforcement, and
  interactive acceptance. Payment failures retain safe, actionable billing
  details.
- Replace the Tesseract runtime dependency with mandatory five-view multimodal
  verification through the selected OpenRouter or AgentBridge model. Retain local
  registration, foreground, canvas, and resolution safeguards and fail closed when
  verification cannot produce a valid verdict.
- Discover user-level or repository `.env` and `.keyenv.toml` configuration for
  installed and checkout-based use.
- Recreate documents as pristine, straight, high-resolution pages without
  holes, damaged edges, dirt, stains, discoloration, or blur; sharpen model
  output at the source page size and make local foreground checks
  tolerant of intentional restoration geometry. Preserve registered authored
  marks and edge microprint while clearing duplicate generated footer content
  and edge-connected scanner artifacts.
- Strengthen full-page and regional prompts around footer sharpness, level
  baselines, and single-instance microprint. Restore reviewer-identified edge
  discrepancies from registered source pixels and verify the repaired page again.
- Move every generation, repair, review, and retry-feedback prompt into packaged
  Markdown resources, with wheel-level inclusion tests.
- Prevent GPT-5.6-sol review reasoning from exhausting the structured-verdict
  budget by using medium reasoning with a larger completion allowance.

## 0.1.1 - 2026-08-09

## 0.1.0 - 2026-08-09

- Initial PaperClean implementation.
