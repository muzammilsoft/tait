# Release Process

1. Run the full test suite.
2. Run a tiny end-to-end train/load/chat smoke test.
3. Run `tait doctor` and a benchmark on a representative target when possible.
4. Update `CHANGELOG.md` and version metadata.
5. Build the package and inspect the archive contents.
6. Tag the release and publish release notes.
