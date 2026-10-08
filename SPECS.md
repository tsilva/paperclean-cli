## PROJECT PURPOSE

PaperClean CLI serves people who need cleaner PDFs or images from phone photos and poor scans. It produces conservative, scanner-like files while preserving source content and refuses to substitute untrusted reconstructions.

## PROJECT REQUIREMENTS

### Inputs and outputs

- Accept individual PDF, JPEG, and PNG inputs plus recursively discovered supported files in directories.
- Do not follow directory symlinks.
- Produce a cleaned file and machine-readable report for each input.
- Distinguish complete success, source-fallback completion, and fatal failure through exit status.

### Fidelity

- Never silently accept changed content in a cleaned output.
- Require local safety checks and independent model verification for every generated page.
- Retry rejected cleanup conservatively and use the untouched original page when no candidate can be trusted.
- Preserve authored content, source ambiguity, photos, diagnostic images, shaded form regions, text, tables, diagrams, signatures, stamps, handwriting, and edge content while removing non-content scan defects.
- Never invent missing, obscured, or unresolved content.

### Safety and transparency

- Show a preflight and require confirmation before processing unless the user explicitly preapproves it.
- Report known provider availability, cost or call ceilings, and credit sufficiency without inventing unavailable USD usage or balances.
- Never proceed when known credits are insufficient.
- Disclose external transmission of complete page pixels and do not claim confidentiality that the selected backend cannot provide.

### PDF safety

- Preserve searchable text in PDF output while removing active content and attachments.
- Reject encrypted PDFs, unapplied redactions, XFA, JavaScript-driven forms, and calculation-driven forms.
