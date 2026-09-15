# Test fixtures

`catalogue/` is a small wardrobe catalogue used by the test suite so that tests
do not depend on the full demo, which is distributed separately.

It was derived from the published demo catalogue by keeping the first three
grouped garments and everything they reference, remapping candidate-row and
group indices so the join keys stay consistent, and downscaling every image to
192px on its long edge. The file layout, JSON schema, and provenance of each
view (`frame` vs `plate`) match real pipeline output.

Regenerate it only if the catalogue format changes; the tests assert against
its structure, not its contents.
