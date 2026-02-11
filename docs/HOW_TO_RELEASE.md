# How To Release

This document defines the release process for `trtx-rs`.

## Release Checklist

1. Bump versions using semantic versioning (`MAJOR.MINOR.PATCH`):
   - Update workspace version in `Cargo.toml` (`[workspace.package].version`).
   - Update `trtx/Cargo.toml` dependency version for `trtx-sys` when needed.
2. Add all relevant release notes to `CHANGELOG.md`.
3. Ensure tests pass.
4. Commit release changes, create tag `vMAJOR.MINOR.PATCH`, and push commit + tag.
5. Publish the GitHub release with `gh release`.

## Commands

Replace `X.Y.Z` with the target version.

```bash
# 1) update version(s) and changelog
$EDITOR Cargo.toml trtx/Cargo.toml CHANGELOG.md

# 2) validate
cargo test --features mock

# 3) commit + tag + push
git add Cargo.toml trtx/Cargo.toml CHANGELOG.md
git commit -m "release: vX.Y.Z"
git tag vX.Y.Z
git push origin main
git push origin vX.Y.Z

# 4) create GitHub release
gh release create vX.Y.Z --title "vX.Y.Z" --notes-file CHANGELOG.md
```

## Notes

- Use `mock` tests when TensorRT-RTX/CUDA are not available in the current environment.
- If the release notes are not at the top of `CHANGELOG.md`, create a release-specific notes file and pass it to `--notes-file`.
