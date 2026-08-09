---
name: build-release
description: Build, tag, publish, monitor, or verify a PaperClean PyPI release. Use when the user asks to cut a release, publish paperclean, build release artifacts, invoke $build-release, or confirm that a version is live.
---

# Build Release

Use PaperClean's repository-owned release script and monitor the exact tag until
the matching files are visible on PyPI. Never manually upload a distribution,
print a credential, or put a PyPI token on a command line. Publication uses
GitHub Actions and PyPI Trusted Publishing through the protected `pypi`
environment.

## Release flow

1. From the repository root, inspect the current branch and worktree:

```bash
git status --short --branch
git log --oneline @{u}..HEAD
```

Stop if the tree is dirty or the branch is unsynchronized. Do not clean, commit,
pull, switch branches, or discard changes on the user's behalf.

2. Prepare the frozen environment and launch the local release gate:

```bash
uv sync --frozen --all-groups
scripts/release.py
```

For an explicitly requested version:

```bash
scripts/release.py --to <MAJOR.MINOR.PATCH>
```

The script requires an unused PyPI version and tag, promotes the Unreleased
changelog, locks dependencies, runs formatting/lint/type/tests, builds exactly
one universal wheel and one sdist, audits both, commits the release metadata,
creates `paperclean-v<version>`, and atomically pushes the current branch and
tag. The artifact audit also installs the wheel in an isolated environment from
the committed lock and invokes its CLI. Report the exact failed gate and stop
if any step fails.

3. Resolve the release commit and monitor only the matching workflow:

```bash
release_sha="$(git rev-list -n 1 paperclean-v<version>)"
gh run list --workflow release.yml --commit "$release_sha" --limit 5 \
  --json databaseId,status,conclusion,event,headSha,url
gh run watch <run-id> --exit-status
```

A `workflow_dispatch` run audits but never publishes. If the tag run fails,
inspect `gh run view <run-id> --log-failed`; do not improvise a manual upload.

4. Poll the exact version until PyPI reports both distributions:

```bash
python .codex/skills/build-release/scripts/release_build.py \
  wait-pypi --version <version>
```

For post-publication verification, query
`https://pypi.org/pypi/paperclean/<version>/json` until the release contains
`paperclean-<version>-py3-none-any.whl` and
`paperclean-<version>.tar.gz`. Also verify the GitHub Release exists for the
same tag.

## Completion

Lead with `https://pypi.org/project/paperclean/<version>/`. Include the exact
tag, workflow URL and conclusion, GitHub Release URL, and both distribution
filenames. Do not report success before PyPI returns the files.
