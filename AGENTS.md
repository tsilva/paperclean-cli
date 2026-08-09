# PaperClean repository instructions

## Development

- Use Python 3.11+ and `uv`; keep `uv.lock` committed and run locked/frozen checks.
- Preserve the dependency-source and minimum-release-age controls in `pyproject.toml`.
- Never log API keys, base64 document data, or verbatim document text.
- Live OpenRouter tests are opt-in and must remain marked `live`.

## Releases

Use the repository-level `$build-release` skill in `.codex/skills/build-release`
for local release candidates, versioning, tagging, PyPI Trusted Publishing,
GitHub Release creation, monitoring, and exact-version verification. Never
upload to PyPI manually or put PyPI credentials on a command line.
