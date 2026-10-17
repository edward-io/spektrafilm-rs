# Rust audit fixes — 2026-10-17

All ten findings from the Rust application audit are addressed. The adjacent
Python checkout and checked-in reference fixtures were left unchanged. Existing
uncommitted Rust work was preserved.

| Finding | Result | Regression coverage |
| --- | --- | --- |
| Failed image/LUT exports truncated existing files | Validate formats first, write to sibling temporary files, and publish only after pixel and metadata writes succeed. | `spektrafilm-io/tests/export_safety.rs`, `spektrafilm-core/tests/lut_export_safety.rs` |
| Valid unfamiliar ICC profiles were treated as linear ProPhoto | Convert other embedded RGB profiles to linear sRGB through the existing color-management dependency. Malformed and unsupported profiles fail explicitly. Gamma-1 float imports preserve negative values. | V4 sRGB, linear HDR/negative values and malformed ICC cases in `export_safety.rs` |
| Rebuilt bundles retained stale QA and ZIPs included unrelated files | Build in a separate directory, replace recognized bundle directories, and archive only new files. Unrelated files and unsafe manifest paths block replacement. Failed publication restores the previous bundle. | CLI bundle/ZIP rebuild, current-directory output, rollback, nested multi-paper ownership and manifest traversal cases |
| EXR always loaded as ProPhoto and lost its saving color space | Write standard RGB chromaticities and linear pixels. Recognize named primaries on import; untagged EXR defaults to linear sRGB. Unknown custom primaries fail explicitly. | Eight color spaces at half/full float precision, direct encoded save, untagged and custom-primary EXR cases |
| Autoexposure applied gain to encoded pixels | Apply exposure gain in linear light on CPU and resident GPU. Metering still precedes crop/rescale. | `autoexposure_encoding.rs`: equivalent encoded/linear inputs at four brightness levels, CPU and Metal |
| CLI RAW processing overwrote explicit input settings | Infer color space and decoding independently, only for fields absent from the parameter file. | Four combinations of explicit/omitted fields, comparing actual CLI output with the shared pipeline |
| Projects stored pending RAW controls | Save the settings that produced the loaded pixels. Explicit startup defaults can still retain pending control preferences. | Headless GUI loads a DNG, changes controls, saves/reopens a project and compares the applied settings and pixels |
| Relative image paths broke when projects were saved elsewhere | Resolve paths when loading and anchor saved relative paths to the project directory. | Project saved outside the original working directory reopens the same source |
| Multi-stage Lumix bundles labeled intermediate density as V-Log | Require a complete single-stage transform for Lumix delivery. | Reject 2/3/4-stage Lumix bundles before output creation; verify a valid 1-stage header |
| RAW comparison helper duplicated outdated decoding | Use shared application loading and saving; remove four redundant direct CLI dependencies. | Portrait cropped DNG helper output matches shared dimensions, orientation, color and every pixel |

Fourteen regression tests were added across core (5), I/O (4), CLI (3), and
GUI/project state (2). Existing saving-color and Python stage-oracle tests were
adjusted for intentional corrections without regenerating their fixtures.

Autoexposure intentionally differs from Python v0.3.4 for encoded inputs.
Same-space transport conversion also avoids multiplying by a rounded RGB
matrix: changing encoding must not mix color channels. Other deterministic
stage and transport reference checks remain in place.

## Validation

Run from the Rust repository, with the installed LibRaw explicitly enabled:

```sh
SPEKTRAFILM_LIBRAW_LIBRARY=/opt/homebrew/lib/libraw_r.dylib \
  CARGO_INCREMENTAL=0 cargo test --workspace --features precision-f64 --locked --offline
SPEKTRAFILM_LIBRAW_LIBRARY=/opt/homebrew/lib/libraw_r.dylib \
  CARGO_INCREMENTAL=0 cargo test --workspace --locked --offline

CARGO_INCREMENTAL=0 cargo test -p spektrafilm-core -p spektrafilm-cli \
  --test lut_export_safety --test regressions --locked --offline
CARGO_INCREMENTAL=0 cargo test -p spektrafilm-core -p spektrafilm-cli \
  --features precision-f64 --test lut_export_safety --test regressions --locked --offline
CARGO_INCREMENTAL=0 cargo check --workspace --all-features --locked --offline
```

- Full f32 workspace: **160 passed, 0 failed, 1 ignored**.
- Full f64 workspace: **169 passed, 0 failed, 1 ignored**. The additional nested
  bundle ownership case was added afterward and passed the final targeted run,
  giving **170 distinct f64 tests** checked.
- Final CLI/core regressions: **7 passed in each precision**, including the
  current-directory ZIP rebuild after its path-resolution correction.
- All-feature workspace compilation, including CUDA: **passed**.
- Installed LibRaw numeric and geometry tests ran in both full suites.
- Focused rustfmt checks and `git diff --check`: **passed**. Repo-wide formatting
  still reports pre-existing formatting in unrelated files and module ordering;
  those files were not reformatted as part of this fix.
- The adjacent Python repository's tracked diff remains empty.

## Limits

Validation used macOS/Metal and headless egui. Native pointer/dialog workflows,
Windows/Linux execution and CUDA runtime were not exercised. The real-camera
test body was not enabled with a personal RAW file; RAW regressions used the
synthetic DNG and recorded rawpy oracle. The live Lensfun database download test
remains intentionally ignored.

Bundle publication targets one local application/export process. It does not
provide coordination for simultaneous writers to the same output directory.
No release application bundle was rebuilt or installed by this change.
