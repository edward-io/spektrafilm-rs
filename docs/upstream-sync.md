# Reference parity and application support

Reference: the adjacent Python `spektrafilm` v0.3.4 checkout at
`3bb2c2d2801ff68b92019cf1dbcbb133d60832bc`. The Rust baseline was v0.1.2,
`9dd59b0380194b93686aaa230a8bb9680aa270a4`.

## Implemented

| Area | Behavior |
| --- | --- |
| LUT baking | Calibrated spectral CPU pipeline; deterministic LUT regime disables image-dependent and stochastic effects. Input/output transport curves and exposure gains follow the Python registry. |
| LUT delivery | CUBE, Autodesk 10-bit 3DL, Lumix V-Log CUBE, 8-bit Hald PNG; 1/2/3/4-stage bundles, contiguous combinations, multiple papers, TOML/JSON specs, ZIP, wire ranges, optional OCIO and native scientific QA reports with local SVG/PNG diagnostics. |
| Color | Named colour-science RGB primaries, white points and CCTFs; CAT02 conversion; selected output encoding; input xy/OkLCh and output ACES RGC/OkLCh/OkLrab/CAM16-UCS/JzAzBz gamut compression. Unknown names fail explicitly. |
| Runtime settings | UV/IR bandpass, Hanatos adaptation surface and spectral blur, crop and bicubic rescale with physical pixel pitch, preview/debug regimes, dye channel ordering, named illuminants, intermediate inject/collect taps. Auto exposure precedes crop/rescale and encoded RGB is decoded at film exposure. |
| Grain | Three physical emulsion sublayers with per-layer density curves, particle fractions/scales, dye-cloud blur, microstructure and final grain blur. The simple model remains selectable. wgpu executes layered grain, dye-cloud blur and microstructure within the resident chain. |
| Application | Shared GUI/CLI raster and RAW I/O; JPEG/PNG/TIFF/EXR; unbounded float output; ICC for bundled profiles, EXIF/XMP and JPEG IPTC preservation; orientation handling; RAW white balance and optional Lensfun correction. |
| GUI | Background image loading, persistent loading/error status, advanced color/crop/spectral/grain/RAW/export controls, project/default persistence, and Python GUI state import. Display previews convert the selected output primaries/CCTF to sRGB. Saving converts to an independent output space. Full-resolution GPU export is the default; f64 CPU export remains selectable. Native LUT export controls cover formats, stage bundles, QA and BundleSpec builds. Manual preview, quarter-turn rotation, source/output/paper-back layers, opacity/split comparison, display transform, all seven GPU sampling modes and paper development/crossfade persist or operate natively. Profile presets seed once on selection; subsequent low-level DIR/halation edits survive rebuilds. Scan-for-print restores previous correction/glare settings when disabled. |

## Sony ILCE-7M5

The original application used rawler 0.7.2, which rejected this camera as
`Unsupported { what: "Unknown camera", model: "ILCE-7M5", ... }`.
The workspace now uses rawler 0.8, including its ILCE-7M5 camera profile and
ARW decoder support. This raises the Rust toolchain requirement to 1.89.
No external RAW library is needed for this rawler path.

The user's `DSC00009.ARW` was decoded successfully with all three selections:

- rawler: 7008 × 4672, linear sRGB, active-area crop.
- LibRaw: 7168 × 5120, linear ACES2065-1, matching rawpy's output geometry.
- Auto: LibRaw when installed; rawler if LibRaw is missing or cannot decode.

LibRaw uses the public C API through dynamic loading. Install a current system
LibRaw, or set `SPEKTRAFILM_LIBRAW_LIBRARY` to the library path. Explicit decoder
selection surfaces that decoder's errors. Auto reports both failures if neither
can decode. A separately installed library is optional for camera support, but
LibRaw gives the closest RAW-development match to the Python/rawpy application.

Personal photographs are not checked into the repository. Run the opt-in camera
regression with `SPEKTRAFILM_RAW_TEST_IMAGE=/absolute/path/to/photo.ARW cargo test
-p spektrafilm-io --test camera_raw -- --nocapture` (both libraries must be
available for that test). Ordinary tests use a small synthetic DNG instead.

The default Rust build includes Lensfun 0.3.4 as a static library. No separate Lensfun installation or library-path variable is needed. `SPEKTRAFILM_LENSFUN_LIBRARY` is an optional explicit override for comparisons with another installed library. Building from source requires GLib development headers / static archives and `pkg-config` (macOS: `brew install glib pkg-config`); Lensfun source is vendored in `crates/spektrafilm-io/vendor/`. macOS builds also link GLib/gettext/PCRE2 statically so those libraries are not runtime prerequisites. `spektrafilm-io --no-default-features` retains the optional dynamic-library path for custom builds. Database selection is automatic. An explicit folder or `SPEKTRAFILM_LENSFUN_DATABASE` takes priority, followed by the application's cache, standard Lensfun locations and the library's built-in search. If no database is found, the loading worker downloads the official format-1 archive from [Lensfun](https://lensfun.github.io/db/version_1.tar.bz2), validates the XML/archive and atomically publishes a local snapshot. Downloads have time and size limits; interrupted or invalid downloads do not replace a working snapshot. Subsequent loads and exports work from the cache without internet access.

On macOS the managed cache is `~/Library/Application Support/spektrafilm-rs/lensfun/`; Linux uses `$XDG_DATA_HOME/spektrafilm-rs/lensfun` (default `~/.local/share`), and Windows uses `%LOCALAPPDATA%/spektrafilm-rs/lensfun`. The current download is about 435 KB / 5.2 MB expanded, containing 56 XML files. No administrator access is needed. In the GUI enable **Lens correction**, leave **Database: automatic**, and click **Reload RAW with these settings**. The folder picker remains an optional override; **Use automatic Lensfun database** clears a previously selected override. A camera/lens without a matching calibration still receives no correction. Lens setup failures now retain a dedicated error category: Auto does not repeat lens setup with the other RAW decoder or attach advice to install LibRaw.

## Preview and GPU performance

Measured in release mode on this Apple M1 Max with the same Sony file, Kodak Gold 200,
Kodak Portra Endura, and physical layered grain. These are local observations,
not cross-hardware guarantees.

| Operation | Before | After |
| --- | ---: | ---: |
| Resize 7168 × 5120 to 2048 × 1462 | 5355 ms | 38 ms |
| Warm 2048 × 1462 render | 2125 ms | about 150 ms |
| Warm 640 × 457 interactive render | — | about 22 ms |
| Full 7168 × 5120 render | 31881 ms in old GUI | 4194 ms cold, packaged P3 export |

The old layered model forced staged processing with CPU grain. wgpu now keeps
its physical grain passes on the GPU. Preview resizing fuses anti-aliasing
with destination sampling, avoiding two full-resolution f64 intermediates.
The GUI caches scaled inputs and uses 640 pixels while a control is dragged,
then refines to its configured limit (2048 by default). Simplified preview
mode independently changes the effect model; it no longer controls whether the
GUI accidentally renders the entire original. Save preview and full export are
explicitly separate actions.

Reproduce timings with:

```sh
SPEKTRAFILM_STAGE_TIMINGS=1 cargo run --release -p spektrafilm-cli \
  --example preview_bench -- /absolute/path/photo.ARW 2048
```

The GUI reports the selected backend and actual resident/staged processing
path. Encoded input and every named output transport now remain GPU resident,
including linear output, general primaries / whites and JzAzBz compression.
Crop/rescale are preprocessed before the resident chain. Explicit enlarger /
scanner LUT modes use staged GPU spectral integration and tensor PCHIP; their
switches and resolution are honored. CPU f64 export preserves those settings.
Metering antialiases a bounded thumbnail before decoding it and performs the
full-frame exposure scale / decoding in the resident compute chain.

## Verification

Both expanded workspace suites passed (121 f32 and 131 f64 tests), followed by the new full-cube envelope regression in both modes and the final GUI geometry regressions. The 40-case real-image audit has max error 0.0004163 and zero values above 0.001. Independent f32 GPU / f64 CPU frontend builds pass 40 synthetic cases with max error 0.000585. Machine-readable results are in `docs/validation/`; see the completion checklist for the exact evidence and remaining verification limits. `cargo test --workspace --features precision-f64` covers the existing stage,
GPU comparison, RNG and model tests plus:

- 27 Python-generated pipeline cases for decoding, output spaces, crop/rescale,
  exposure ordering, taps, filters, adaptation, gamut, debug/preview and grain.
- Calibrated deterministic LUT baking, transport curves, Python bundle bake
  comparisons for SDR/log/PQ/HLG, and tap-chain reconstruction.
- All named RGB CCTFs against colour-science samples.
- Scientific QA color differences (ΔE2000, ΔITP, OkLab), tri/tetrahedral
  interpolation, singular values, Jacobian conditioning, FFT/variation,
  gamut hull/surface, hue rotation, dynamic range and noise covariance,
  compared with independent Python metric fixtures.
- Native project round-trip and Python GUI state import, including saving
  color, manual preview and display preferences; child-process output draining
  and export cancellation.
- Independent saving-space conversion, floating-point output, ICC recognition
  and metadata round-trip; opt-in LibRaw/rawpy and
  Lensfun integration tests.

Reference fixtures and registries live under `scripts/reference/`. Regenerate
with `PYTHONPATH=../src` and the reference's pinned Python dependencies. The
Python repository is a development oracle, never a Rust runtime dependency.
The standalone LUT smoke harness additionally reads every native format through
Python's format plugins, validates and executes all OCIO topologies, and checks
multi-paper ZIP packaging and QA reports.

The current QA uses 50,000 off-grid samples from the same fixed NumPy PCG64 stream, and the source's 8,000-point hull sampling. All 16 diagnostic probes are implemented. The full input-gamut, spectral envelope and output-rim measurements on a native 17³ V-Log bundle match the original Python QA: exact populations, hull-ratio difference 2.3e-8 and rim-displacement differences below 3.2e-9. These are separately recorded in `docs/validation/2026-10-17-lut-gamut-reference.json`. Low-resolution LUT sampling can still fail the source's fidelity thresholds; reports and README quality tables retain those failures. LUT interpolation error is a different measurement from GPU arithmetic error and does not share the GPU's 0.001 gate.

An earlier packaged build decoded the real Sony file and rendered a 2048 × 1462
resident-GPU preview; cold-start timings include GPU pipeline compilation. The current packaged build completes full-resolution
7168 × 5120 JPEG exports with default P3, highlight boost and explicit spectral
LUTs. The latter exercises wrapped 2D PCHIP dispatch above 16.7 MP. ICC and
camera metadata survive; f64 CPU rotation and the app signature are checked. The Mac was locked for the final UI pass,
so click-through verification of the latest export controls remains unverified;
headless tests exercise their worker's full-size/working-scale/saving-color path.

## Limits and remaining differences

Numerical parity is tolerance-based, not a claim of bit-identical output.
Small deterministic f64 fixtures use a 2e-6 output tolerance. Stochastic grain
can change per pixel after upstream rounding differences; microstructure uses
a deterministic local random field rather than Python's global numba state.
The model and its controls are shared in meaning, not the exact noise texture.
Native film-scanning glare is an extension: Python v0.3.4 omits glare while
scanning film. The native debug/LUT switches suppress both glare paths.

GPU shaders use f32. The wgpu spatial filters now share the CPU FIR / compensated IIR model; optical diffusion uses its exact discrete radial PSF via reflected overlap-save FFTs. The strict deterministic gate is maximum absolute channel error 0.001, or 0.1% of normalized full scale; it is not a relative-error guarantee near black. The stress image includes HDR ramps, sharp edges, checkerboards and impulses.

Operations the resident chain cannot represent fall back to staged processing, whose halation, DIR and optical filters also use the accurate GPU kernels. Layered grain uses per-pixel PCG streams. Exact mode collapses Poisson-binomial sampling by the exact Poisson-thinning identity and uses transformed rejection, preserving the particle law without the earlier normal approximation. Approximate statistics remain explicitly selectable. Grain / glare textures differ from the CPU RNG: a pixelwise 0.1% claim for different random realizations would be invalid. Analytical moment, tone, variance, reversal and particle-scaling checks run independently; an independent Python replay with identical particle / clumping draws now verifies numerical processing within 1.4e-6, including blur borders. This is separate from the random-law checks. f64 CPU remains the closest numerical reference. CUDA is
compile-checked here, not executed on this Mac; it retains CPU layered grain.

The optional `qa.json`/`qa.html` reports include structural / off-grid fidelity, density-pinned characteristic curves, dynamic range anchored at native camera midgray, the Planckian/daylight sweep, full-cube chromaticity maps, output compression rims, input-gamut population / circumferential probes, hue rotation, noise ellipses / gain maps / rosettes, a seeded continuous noisy gradient, wide-gamut runtime stress and nine R–G slices. All samples contribute to the bounded raster plots. Native SVG/PNG layouts differ from Matplotlib's presentation; the source stimuli and measurements are preserved and independently checked. Bundle filenames and JSON metadata use the Rust layout. Metadata includes the resolved runtime configuration, gains, source versions, stock/color contract, license/citation and quality summary. CLI flags can override a TOML/JSON spec; the resolved spec is saved. OCIO uses the Python reference's verified builtin catalog and is omitted for transports outside that catalog.

The native editor provides source/output/paper-back/border visibility, comparison, all seven GPU sampling modes, paper development and crossfades. GPU animation frames match the Python nine-layer oracle, and the static paper-back cache matches its alpha, lattice and seeded texture. It uses egui/wgpu and a fixed named stack corresponding to the source controller's ordering. Generic Napari layer-editor / console facilities are not reproduced. The final native pointer/dialog pass remains unverified because the Mac is locked; worker and serialization paths pass automated tests.

The parametric density-curve model and measurement/calibration helpers are native. `measure-profile` writes gamma, local slopes and erf toe fits plus spectral/density/dye/toe figures. `calibration-strip` supports all five condition ramps, crop / quarter-turn rotation, horizontal/vertical stacking, labels, JSON and self-contained SVG output. Its independent per-condition settings fix the Python helper's shallow-copy aliasing bug. The source's `utils/plotting.py` is explicitly a nonworking draft with obsolete interfaces; its intended working profile plots are available through `measure-profile`.

Large images remain bounded by the selected GPU's storage/dispatch limits; the tested full-resolution case is 7168 × 5120. CUDA is experimental and compile-only here. These are not claims of validation on every adapter or unlimited image sizes.

Embedded ICC profiles currently cover sRGB, ProPhoto, Adobe RGB, Rec.709, Rec.2020,
ACES2065-1, ACEScg, and encoded Display P3/DCI-P3. Other runtime output spaces retain their numeric encoding but do
not yet get a generated ICC profile. EXR carries float pixels; it does not copy
EXIF. rawler and LibRaw have different demosaic/crop/color pipelines, so their
RAW pixels are not interchangeable parity baselines.
