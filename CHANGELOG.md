# Changelog

## Unreleased

- Rename the GitHub repository and PyPI distribution to `paperclean-cli` while
  preserving the `paperclean` command and Python import package.
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
  discrepancies, including edge-local layout changes, from registered source
  pixels and verify the repaired page again. Confirm isolated scanner-quality
  rejections once so a stochastic verdict cannot discard an otherwise verified page.
- Retry a page-scoped reviewer timeout exactly once and include that bounded retry in
  conservative recovery call projections.
- Restore all localized reviewer-identified text, table, diagram, layout, and authored
  mark discrepancies from registered cleaned source pixels before re-verification,
  keeping model repair only for unclassified content.
- Move every generation, repair, review, and retry-feedback prompt into packaged
  Markdown resources, with wheel-level inclusion tests.
- Prevent GPT-5.6-sol review reasoning from exhausting the structured-verdict
  budget by using medium reasoning with a larger completion allowance.
- Recover dense reports and tables with a provenance-visible, source-preserving
  white-paper cleanup after generative attempts are exhausted. Remove boundary rails
  and punched holes without synthesizing text, then require the same five-view model
  verification before publication. Confirm every rejection once so stochastic content
  alarms cannot discard a deterministic recovery, and record residual quality limitations
  and expected global deskew/layout rectification in legacy artwork without letting them
  veto this content-exact path. Record confirmed non-specific alerts without treating them
  as evidence of a semantic mismatch.
- Preserve detected photographic, diagnostic-image, and large shaded form panels
  pixel-for-pixel during source cleanup, while still whitening and rectifying the
  surrounding paper. Treat a
  `changed_diagram` alert as expected only on pages where this preservation mask exists,
  and prevent thin scan-noise bridges from merging a panel into the page border.
- Make binder-hole removal context-aware so holes touching text, table lines, or authored
  marks are never erased blindly. Attempt a localized model restoration only when the
  obscured continuation is highly probable, then publish it only after full-page and
  regional verification. Treat vague content alerts as rejection on this synthetic path;
  otherwise retain the hole. Combine connected-component and circle geometry across the
  outer ten percent of either side margin, classify physical ink connectivity instead of
  broad proximity, scale erasure to include the visible hole halo, and exclude
  diagnostic-panel pixels from hole candidates and restoration.
- Normalize discrepancy-free reviewer booleans per view so later regional failures retain
  their actual category in reports. For deterministic source cleanup, restore localized
  normalized source evidence after explicit text-like discrepancies and require a fresh
  five-view verification before publication.

## 0.1.1 - 2026-08-09

## 0.1.0 - 2026-08-09

- Initial PaperClean implementation.
