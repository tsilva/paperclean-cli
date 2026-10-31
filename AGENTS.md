# PaperClean repository instructions

## Development

- Use Python 3.11+ and `uv`; keep `uv.lock` committed and run locked/frozen checks.
- Preserve the dependency-source and minimum-release-age controls in `pyproject.toml`.
- Never log API keys, base64 document data, or verbatim document text.
- Live OpenRouter tests are opt-in and must remain marked `live`.

## Releases

Use the repository-level `$paperclean-cli_build-release` skill in `.codex/skills/paperclean-cli_build-release`
for Actions release candidates, local metadata-only versioning/tagging, PyPI Trusted Publishing,
GitHub Release creation, monitoring, and exact-version verification. Never
upload to PyPI manually or put PyPI credentials on a command line.

## Product Specifications

When specifications exist, use `$specs-author` for substantive work and
specification changes. Follow its shared read/reuse cadence, provisional-scenario
guidance, proportional detail, decision-based approval, and closing intent check.
The maintained procedure is `/Users/tsilva/.codex/skills/private-skills/specs-author/SKILL.md`.

## Shared release procedure

The project `paperclean-cli_build-release` skill composes `$release-workflow` from
`/Users/tsilva/.codex/skills/release-workflow/SKILL.md`.
Read both for release work; keep project commands, version policy, artifact
requirements, and approval gates in the project adapter.
