# PaperClean

PaperClean turns poorly photographed or scanned documents into conservative,
scanner-like PDFs and images. It uses an image model to clean each page, then
requires deterministic content checks and a multimodal reviewer to approve the
result. Rejected pages fall back to the original and are recorded in embedded
provenance and a JSON report.

PaperClean sends complete document pixels to OpenRouter and the selected model
providers. Do not process material whose external transmission is prohibited.
`--zdr` is available only when both selected model endpoints appear in
OpenRouter's published zero-data-retention endpoint list; the default image
model may not satisfy that requirement. OpenRouter account logging and privacy
settings remain the account owner's responsibility.

## Install

PaperClean requires Python 3.11 or newer and Tesseract 5.5 or newer. On macOS:

```bash
brew install tesseract
uv tool install keyenv
uv tool install paperclean
```

For a checkout:

```bash
uv sync --frozen --all-groups
keyenv authorize OPENROUTER_API_KEY
keyenv set OPENROUTER_API_KEY
keyenv doctor
keyenv run -- uv run paperclean --help
```

Install additional Tesseract language packs through the platform package
manager and select them with `--ocr-lang` or `PAPERCLEAN_OCR_LANG`.

## Usage

```bash
keyenv run -- uv run paperclean document.pdf
keyenv run -- uv run paperclean scans/
keyenv run -- uv run paperclean photo.jpg --max-attempts 3
```

Supported inputs are PDF, JPEG, and PNG. Directory traversal is recursive and
does not follow directory symlinks. A source named `document.pdf` produces:

```text
document.clean.pdf
document.clean.pdf.report.json
```

Exit status `0` means every page passed, `2` means one or more original pages
were used, and `1` means a fatal/file failure or incomplete batch.

CLI flags override environment variables, which override these defaults:

| Environment variable | Default |
| --- | --- |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` |
| `PAPERCLEAN_IMAGE_MODEL` | `openai/gpt-image-2` |
| `PAPERCLEAN_REVIEW_MODEL` | `openai/gpt-5.6-sol` |
| `PAPERCLEAN_MAX_ATTEMPTS` | `3` |
| `PAPERCLEAN_JOBS` | `1` |
| `PAPERCLEAN_OCR_LANG` | `eng` |
| `PAPERCLEAN_MAX_COST_USD` | unset |
| `PAPERCLEAN_ZDR` | `false` |

`OPENROUTER_API_KEY` has no default and is required. The valid OpenRouter slug
for the requested GPT Image 2 model is `openai/gpt-image-2`.

PaperClean preserves the original PDF text streams beneath an opaque page
overlay. This retains searchable text, but deskewing or perspective correction
can make old search-highlight coordinates imperfect. PaperClean rejects
encrypted PDFs, unapplied redaction annotations, XFA, JavaScript-driven forms,
and calculation-driven forms. Outputs are static sanitized PDFs; interactivity,
signatures, active actions, and original attachments are removed.

`--max-cost-usd` is a soft observed-cost ceiling. When enabled, paid requests
are serialized, but one completed request and an ambiguously billed timeout can
still exceed the value.

## Development

```bash
uv sync --frozen --all-groups
uv run ruff check .
uv run mypy -p paperclean
uv run pytest
uv build --no-sources
```

Live contract and end-to-end tests are never part of the default test run:

```bash
keyenv run -- uv run pytest -m live
```

## Release setup

Releases use the repository-local `$build-release` skill and PyPI Trusted
Publishing. Configure the existing `paperclean` PyPI project with owner
`tsilva`, repository `paperclean`, workflow `release.yml`, and environment
`pypi`. Protect the GitHub `pypi` environment with manual approval.
