# Releasing Zephon

Zephon uses `setuptools-scm`, so the package version comes from the git tag on
the release commit (`v0.1.0` → `0.1.0`). Versions follow
[PEP 440](https://peps.python.org/pep-0440/): `X.Y.ZrcN` for release
candidates, `X.Y.Z` for releases.

## Publishing to PyPI

Releases go to [PyPI](https://pypi.org/p/zephon) through the manually
triggered `Publish to PyPI` workflow (`.github/workflows/pypi-publish.yaml`),
which uses [trusted publishing](https://docs.pypi.org/trusted-publishers/): no
API token is stored anywhere. Pushing a `v*` tag does **not** publish anything
by itself.

One-time setup (already done for `zephon`):

- A trusted publisher on pypi.org and on test.pypi.org for owner `datologyai`,
  repository `zephon`, workflow `pypi-publish.yaml`, and environment `pypi`
  (`testpypi` on TestPyPI).
- GitHub environments `pypi` and `testpypi` on the repository, both limited to
  `v*` tags. `pypi` requires a reviewer's approval before anything uploads.

To release:

1. **Tag and push** the release commit:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
   PyPI never accepts the same version twice, even after a deletion, so pick
   the version deliberately. The PyPI project page shows the README from the
   uploaded sdist, so land README changes before tagging.
2. **Dry run on TestPyPI:** Actions → *Publish to PyPI* → *Run workflow*, set
   "Use workflow from" to the tag, and choose the `testpypi` target. Then
   check that it installs, in a fresh environment:
   ```bash
   uv pip install --no-deps --index-url https://test.pypi.org/simple/ zephon==0.1.0
   uv pip install zephon==0.1.0
   ```
   The first command takes only zephon from TestPyPI; the second keeps it and
   pulls its dependencies from PyPI. (A single command with both indexes
   doesn't work: uv checks `--extra-index-url` first and stops at the first
   index that has the package, so once zephon is on PyPI it never looks at
   TestPyPI.)
   Versions are permanent on TestPyPI too, so to rehearse ahead of a release
   without spending its version, dry-run an `rcN` tag instead.
3. **Publish:** run the workflow again from the same tag with the `pypi`
   target, and approve the `pypi` environment deployment when prompted.
4. **Verify** with `uv pip install zephon==0.1.0` in a fresh environment.

The workflow fails if it is run from anything but a tag, or if the built
wheel or sdist version does not match the tag.
