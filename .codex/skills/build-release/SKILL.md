---
name: build-release
description: Build, tag, publish, monitor, or verify a PaperClean PyPI release. Use when the user asks to cut a release, publish paperclean-cli, build release artifacts, invoke $build-release, or confirm that a version is live.
---

# Build Release

Read and apply the shared `$release-workflow` skill at
`/Users/tsilva/.codex/skills/release-workflow/SKILL.md` before execution.
It owns common preflight, publication safeguards, `$push` integration,
workflow monitoring, verification, and reporting. The rules below are this
project's adapter; they retain its invocation default and required gates.
If the shared skill is unavailable, stop and report the missing dependency.

A bare `$build-release` or `/build-release` invocation requests the full
publication flow. Explicitly local, dry-run, or inspection requests must not
launch `scripts/release.py`, which commits, tags, and pushes.

Use PaperClean's repository-owned release script. Publication uses GitHub
Actions and PyPI Trusted Publishing through the protected `pypi` environment.

## Release flow

1. From the repository root, inspect the current branch and worktree:

```bash
git status --short --branch
git log --oneline @{u}..HEAD
```

Stop if the tree is dirty or the branch is unsynchronized. Do not clean, commit,
pull, switch branches, or discard changes on the user's behalf.

2. Launch the metadata-only release operator:

```bash
python3 scripts/release.py
```

For an explicitly requested version:

```bash
python3 scripts/release.py --to <MAJOR.MINOR.PATCH>
```

The operator checks an unused version and tag, updates version/lock metadata,
commits that metadata, and atomically pushes synchronized main and the release
tag. It does not install dependencies, run source tests, or build artifacts.
GitHub Actions runs all formatting, lint, type, test, packaging, artifact-audit,
and isolated installed-wheel CLI gates before Trusted Publishing.

For validation without publication, run:

```bash
python3 scripts/release.py --validate
```

This dispatches the exact pushed main SHA, including when local unrelated work
is dirty. Monitor that SHA and download/audit its two distribution artifacts.
No version bump, tag, or publication occurs. Explicit local artifact inspection
can use the existing helpers, but normal release and validation builds run only
in Actions.

3. Follow the shared monitoring and verification procedure for the `release.yml`
tag-push run at the full `paperclean-cli-v<version>` commit SHA. A `workflow_dispatch` run
validates artifacts but never publishes. Verify PyPI project `paperclean-cli` and
the GitHub Release for the same tag.

Use the existing exact-version verifier:

```bash
python .codex/skills/build-release/scripts/release_build.py \
  wait-pypi --version <version>
```

Require `paperclean_cli-<version>-py3-none-any.whl` and
`paperclean_cli-<version>.tar.gz` on PyPI and the GitHub Release for the tag.
