# Rendering and application audit — 2026-10-17

Audited the native profile loader, CPU/wgpu rendering and grain dispatch,
preview calibration cache and worker lifecycle, RAW geometry/white balance,
and GPU versus f64 export source handling. This follows the earlier fix for
single-channel B&W physical grain layers. Existing uncommitted GUI work was
preserved. No application bundle or ZIP was created.

## Confirmed additional defects, fixed

### P1 — Missing B&W sensitivity became full spectral sensitivity

`ProfileData.log_sensitivity` used the null-to-zero deserializer. The renderer
then exponentiates this logarithmic data: `10^0 = 1`, whereas the Python
filming and printing stages apply `nan_to_num(10^log_sensitivity)` and therefore
give an unmeasured wavelength zero weight. Double-X contains 33 null entries,
Tri-X 30, and Kodak 2302 56. Both CPU and GPU inherited the same incorrect
data, so CPU/GPU agreement alone could not detect this error.

The loader now preserves NaN for missing log sensitivity, including through
B&W channel expansion. The existing linearization already maps these values
to zero. A regression checks the source's missing first wavelength in all
three B&W profiles and both accessor precisions. Color profiles contain no
null sensitivity entries; their data is unchanged.

### P1 — LibRaw included sensor padding in the Sony image

The Sony ILCE-7M5 sample decoded as 7168 × 5120, including the black strips at
the right and bottom. Applying its recorded vendor crop produces 7008 × 4672.
The Rust adapter now calls LibRaw's optional public crop API after unpacking
and before processing. The [LibRaw API documentation](https://github.com/LibRaw/LibRaw/blob/master/doc/API-CXX.html#adjust_to_raw_inset_crop)
specifies that order so masked-pixel black-level calculations remain correct.
The C entry point is available in LibRaw 0.22; older libraries retain their
own native crop behavior.

A synthetic DNG regression specifies a 28 × 24 default crop inside a 32 × 32
sensor. Before the fix LibRaw returned 32 × 32; afterward it returns 28 × 24.
The original uncropped DNG still passes the existing rawpy numeric fixtures.
The real Sony sample now opens at 7008 × 4672 with rawler, LibRaw and Auto.
This vendor-crop behavior deliberately differs from the previous rawpy-style
padded output. The GUI's decorative white border is a separate display setting.

### P2 — rawler ignored EXIF rotation and reflection

`RawDevelop` demosaics and crops but does not apply the orientation tag. The
adapter returned those untransformed pixels, while export later reset EXIF
orientation to 1. Portrait and mirrored RAW images could therefore be saved
with the wrong orientation.

The adapter now applies all eight EXIF orientations after lens correction,
while the correction still operates in sensor coordinates. A nonsquare DNG
regression checks output dimensions and every pixel's position for all seven
nonidentity transformations, independently of the manual rotation helper.

### P2 — Inactive custom tint changed As Shot images

rawler multiplied green by the stored tint in every white-balance mode.
LibRaw and the Python implementation apply that tint only in Custom mode.
Switching away from Custom could therefore keep its tint with rawler but
discard it with LibRaw.

Tint now applies only in Custom mode on both decoders. Its GUI slider appears
alongside Custom temperature, and the tooltip describes that scope. The DNG
regression checks that inactive tint leaves As Shot, Camera, Daylight and
Tungsten unchanged, and that Custom still multiplies green without changing
red or blue.

### P2 — Failed preview refinement retried and exposed stale output for saving

After a successful 640-pixel interactive frame, a failed high-resolution
render left `output_image` populated. The idle refinement condition continued
to see an incomplete preview and could retry the failing render repeatedly.
The previous result could also become saveable under the newly requested
settings after the worker finished.

Render failures and disconnected workers now invalidate the saveable result.
This stops automatic refinement until another edit/manual render, while
retaining the error message. A headless egui regression delivers an actual
worker error, verifies the old output cannot be saved, and runs another GUI
frame to ensure it does not launch a retry.

### P2 — f64 export used RAW edits that had not been reloaded

GPU export uses the loaded source pixels. f64 export reopens the source, but
previously passed the current RAW controls even if those edits had not been
applied with Reload. Merely changing export backend could therefore change
white balance, decoder or lens correction.

Successful loads now retain the RAW settings snapshot that produced the
pixels. Rotation preserves it. f64 export uses that snapshot, and a successful
reload replaces it. The GUI regression loads a real synthetic DNG through the
background worker, edits controls while decoding, rotates the loaded pixels,
then reloads and checks the export settings at each step.

## Additional checks

- All bundled profile grain tensors load and resolve to three physical
  layers and three storage channels. Multi-time B&W profiles preserve the
  selected time. Real Double-X and Tri-X profiles exercise CPU, staged GPU
  and resident GPU physical grain, including direct film scans and printing.
- A new cache regression changes 33 controls for both Kodak Gold/Portra Endura
  and Double-X/2302: 66 cases covering exposure, input/output color, bandpass,
  development time, gamma/morph, optical effects, spectral reconstruction,
  preview mode and scanner corrections. Every cached result matches a fresh
  pipeline within 1e-6 absolute channel error. No additional cache-key
  omission was found in these cases.
- Existing workspace regressions cover deterministic Python fixtures, LUT
  baking/QA, color transports, full-stage GPU comparisons, grain statistics,
  ICC/metadata handling, export cancellation and GUI project state.
- The legacy paper density gamma remains a stored-curve fallback; fitted
  papers use the print-curve morph controls. This matches the reference and
  is already explained by its tooltip.

## Validation

Final workspace results: **146 f32 tests passed** and **156 f64 tests passed**,
with no failures. Each run intentionally skips the one live Lensfun download
test. The installed-library LibRaw crop and rawpy regressions were enabled.
The strengthened RAW-mode and GUI-source-state regressions also pass in f64.

Commands run from the Rust repository, with the locally installed LibRaw
0.22.2 explicitly enabled for its opt-in tests:

```sh
SPEKTRAFILM_LIBRAW_LIBRARY=/opt/homebrew/lib/libraw_r.dylib cargo test --workspace
SPEKTRAFILM_LIBRAW_LIBRARY=/opt/homebrew/lib/libraw_r.dylib cargo test --workspace --features precision-f64
SPEKTRAFILM_RAW_TEST_IMAGE=/absolute/path/DSC00009.ARW \
  SPEKTRAFILM_LIBRAW_LIBRARY=/opt/homebrew/lib/libraw_r.dylib \
  cargo test -p spektrafilm-io --test camera_raw -- --nocapture
```

The GUI, ordinary CLI and f64 export helper were rebuilt in
`target/macos-app/release/`. Release smoke checks render B&W output and decode
the Sony sample through the shared RAW path. Detailed local logs are outside
the repository; no personal photograph is committed.

After the workspace runs finished, a quiet release benchmark decoded the Sony
sample in 1017 ms and resized it to 640 × 426 in 25.7 ms. With Kodak Gold 200,
Portra Endura and physical layered grain, the first render took 77.8 ms and
three warm renders took 23.9, 22.7 and 23.5 ms. All values were finite on
wgpu/Metal. This is a local observation, not a cross-hardware guarantee.

## Remaining limits

- CPU/GPU numerical agreement does not prove agreement with physical film.
  There is no independent end-to-end Python B&W photographic oracle here;
  the new B&W tests check profile interpretation and render invariants.
- f64 export still reads the original file from disk. It depends on that
  source remaining available and unchanged after loading. GPU export uses
  the in-memory source. The RAW-controls mismatch is fixed; external file
  replacement is a separate remaining source-lifecycle limitation.
- ICC embedding covers the documented bundled spaces; arbitrary named runtime
  spaces do not all receive generated ICC profiles. Consumers cannot infer
  those spaces from the pixel values alone.
- GPU execution was checked on this Mac's wgpu/Metal backend. CUDA and other
  platforms were not exercised. Full native file-dialog/pointer walkthroughs
  were not performed; GUI lifecycle checks use headless egui frames.

This audit found and repaired concrete defects; it is not a claim that every
profile, imported project, source file or GPU adapter is exhaustively tested.
