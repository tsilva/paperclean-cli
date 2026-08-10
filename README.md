<div align="center">
  <img src="./logo.png" alt="PaperClean" width="300" />

  **🧹 Turn rough document photos into conservative, scanner-like files. 🧹**
</div>

PaperClean is a Python CLI for people who need cleaner PDFs or images from phone
photos and poor scans without silently accepting changed content. Give it a PDF,
JPEG, PNG, or folder; it recreates each page, runs deterministic fidelity checks,
and falls back to the original page when a candidate cannot be trusted.

Complete page pixels are sent to OpenRouter and the selected model providers.
Before paid work begins, PaperClean shows a live cost and credit preflight and
asks for confirmation. Optional `--review` adds five-view multimodal fidelity
review; it is disabled by default.

## Install

PaperClean requires Python 3.11 or newer and Tesseract 5.5 or newer. On macOS:

```bash
brew install tesseract
uv tool install keyenv-macos
uv tool install paperclean
```

Keyenv is the recommended way to keep `OPENROUTER_API_KEY` in the macOS
Keychain. Create `~/.config/keyenv/.keyenv.toml`:

```toml
[keyenv]
version = 1

[secrets.OPENROUTER_API_KEY]
account = "paperclean-user/OPENROUTER_API_KEY"
required = true
```

Authorize and store the key:

```bash
cd ~/.config/keyenv
keyenv authorize OPENROUTER_API_KEY
keyenv set OPENROUTER_API_KEY
keyenv doctor
```

Run PaperClean from the directory containing your document:

```bash
paperclean document.pdf
```

## Commands

```bash
paperclean document.pdf                         # clean one PDF
paperclean photo.jpg                            # clean one JPEG or PNG
paperclean scans/                               # recursively clean a directory
paperclean document.pdf --review                # add five-view semantic review
paperclean document.pdf --max-attempts 3        # set generation attempts per page
paperclean document.pdf --max-cost-usd 1.00     # set a soft observed-cost ceiling
paperclean document.pdf --ocr-lang eng+por      # select Tesseract languages
paperclean --help                               # show every CLI option
```

A source named `document.pdf` produces:

```text
document.clean.pdf
document.clean.pdf.report.json
```

Exit status `0` means every page passed, `2` means one or more original pages
were used, and `1` means a fatal file or batch failure.

## Configuration

CLI flags override environment variables, which override these defaults:

| Environment variable | Default |
| --- | --- |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` |
| `PAPERCLEAN_IMAGE_MODEL` | `openai/gpt-image-2` |
| `PAPERCLEAN_REVIEW` | `false` |
| `PAPERCLEAN_REVIEW_MODEL` | `openai/gpt-5.6-sol` |
| `PAPERCLEAN_MAX_ATTEMPTS` | `3` |
| `PAPERCLEAN_JOBS` | `1` |
| `PAPERCLEAN_OCR_LANG` | `eng` |
| `PAPERCLEAN_MAX_COST_USD` | unset |
| `PAPERCLEAN_ZDR` | `false` |

`OPENROUTER_API_KEY` is required. PaperClean reads supported values from the
process environment first, then user-level `.env` or Keyenv configuration,
then repository-level configuration.

## Notes

- Supported inputs are PDF, JPEG, and PNG. Directory traversal is recursive and
  does not follow directory symlinks.
- Local OCR, registration, foreground, and resolution checks are required before
  any generated page is accepted. Rejected candidates retry with feedback, then
  fall back to the original page when attempts are exhausted.
- PaperClean restores registered signatures, stamps, and edge microprint from
  the source without restoring stains, skew, shadows, or damaged paper edges.
- The cost preflight checks selected endpoints, live pricing, available credits,
  and the conservative recovery ceiling. `--yes` accepts the displayed
  preflight but never overrides insufficient known credits.
- `--max-cost-usd` is a soft ceiling. One completed request or an ambiguously
  billed timeout can exceed it.
- `--zdr` works only when every selected model endpoint is listed by OpenRouter
  as zero-data-retention capable. Do not process documents whose external
  transmission is prohibited.
- PDF outputs keep searchable text streams beneath an opaque page overlay.
  Active content and attachments are removed; encrypted PDFs, unapplied
  redactions, XFA, JavaScript-driven forms, and calculation-driven forms are
  rejected.

## Development

```bash
uv sync --frozen --all-groups   # install the locked development environment
uv run ruff check .             # lint
uv run mypy -p paperclean       # type-check
uv run pytest                   # run the offline test suite
uv build --no-sources           # build wheel and source distribution
keyenv run -- uv run pytest -m live  # run opt-in live tests
```

## Architecture

![PaperClean architecture](./architecture.png)

## License

No license file has been added yet.
