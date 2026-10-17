# Experimental parity

Reference: [andreavolpato/spektrafilm experimental](https://github.com/andreavolpato/spektrafilm/tree/28bf883e1672e884307edc75852549376e13644e),
commit `28bf883e1672e884307edc75852549376e13644e` (2026-07-08).
The Rust runtime and app do not require Python.

## Rendering and controls

- All 31 bundled profiles, fitted characteristic models, and grain/DIR stock
  presets come from this commit. Film chemistry regenerates the total curve
  and physical layers together. Print chemistry uses the same Gaussian/septic
  model, speed and channel gamma controls, exhaustion, and development selection.
- Film and print base tuning includes scale and CMY controls; film also has tilt.
- Negative DIR uses Langmuir donor release. Reversal uses a linear silver donor
  and a Langmuir receiver after Gaussian/exponential diffusion. The CPU computes
  the shared curve inversion, including NumPy's search order for folded axes.
- Grain uses RMS granularity at the 48 µm aperture, physical layer fractions,
  dye-cloud blur, clumping, post blur, and multiplicative sharpening with
  per-channel mass conservation. GPU reductions avoid a CPU image readback.
  Older particle-area controls no longer select a separate application model.
- Hanatos2025, Arctic2026beta04, Mallett2019, Jakob2019, Otsu2018 and
  gauss-lasers consume the bundled spectral registry. Reflectance methods use
  their cubic-sampled neutral for per-channel normalization. Hanatos window
  normalization uses its reconstructed white. Arctic alpha02 remains readable
  for older native projects.
- Input gamut compression supports the smooth inscribed hull and its detail
  control. Output defaults to OkLCh with the experimental knee/lightness values.
- Hoya X0/X1/Y2/YA3/R1 taking filters use the source's Akima-resampled spectra.
  Automatic exposure meters mean film-raw response on a bounded full-frame
  thumbnail before cropping, then applies the gain in linear light. Repeated
  previews reuse the scalar meter result while its image and settings match.
- All six workflow routes are available, including input passthrough and the
  three scanned-negative conversion routes. Conversion uses the bounded
  eight-step spectral inverse or its cached tetrahedral LUT. Device calibration
  and scan exposure stay outside the LUT so they can change without rebaking.
- Internal enlarger/scanner LUTs use tetrahedral interpolation. B&W emulsions
  use the source's 1024-point 1D acceleration path. The existing public PCHIP
  helper remains available but is not the experimental runtime default.

## Application and exports

The GUI exposes the new settings and preserves them in projects. Detect base,
blind device calibration and neutral print balance run on background workers;
results are discarded if their input snapshot changes. The preview cache
includes chemistry, base, workflow, filter, meter, illuminant and LUT dependencies.
Workflow selection takes precedence over legacy scan flags in imports and exports.
Blind calibration decodes at most 1500 sampled pixels instead of copying the RAW
into a full-resolution f64 fitting buffer.
Quarter-turn buttons, label-only help, RAW loading/error handling, Lensfun,
metadata and color-managed saving remain in place.

LUT baking keeps stock, chemistry, base and spectral settings and uses the
source's neutral bake contract: no spatial/stochastic processing, automatic
exposure, camera EV, print exposure, preflash or M/Y trims; no crop/rescale or
internal spectral acceleration LUTs. Native midgray is preserved by default.
`--exposure-ev` / bundle `exposure_ev` applies deliberate re-exposure. Legacy
explicit white-placement stops remain supported. Single-pair GUI exports pass
the current runtime parameters. Bundles record the pinned reference and gain.

## Verification

`tests/experimental.rs` compares 51 actual Python runtime cases on CPU and GPU:
all six spectral methods, both chemistries and bases, five taking filters,
meters/crop/encoding, nonlinear DIR, BW stocks and 1D LUTs, every workflow,
conversion LUTs, device matrices and scan EV. Separate oracles cover defaults,
septic chemistry and physical layers, LUT bakes and coupled grain processing.
The Python generators live in `scripts/reference/`; fixtures include the source
commit. Tests retain the existing strict deterministic GPU gate of 0.001
absolute normalized RGB, plus independent random-law/statistical checks.

Calibration checks fit synthetic clear film, preserve an ideal device scan,
and verify neutral printing after changing the base. GUI tests compare cached
previews with rebuilt pipelines after control changes. RAW, saving, metadata,
ICC, export-worker and interaction regressions remain part of the workspace suite.

Final runs passed: 169 tests in the f32 workspace run, 165 in the f64 run
(excluding the GUI), and all 17 GUI tests after the final workflow changes.
The existing network/cache-writing Lensfun download test remains ignored.

On an Apple M1 Max, the Sony ILCE-7M5 RAW gave these median warm resident GPU
render times. These include preparation and GPU upload/readback, but exclude
GUI texture conversion. The first frame can take longer for shader setup.

| Render mode | 640 × 426 | 2048 × 1365 |
| --- | ---: | ---: |
| Film/print with physical grain | 32.3 ms | 227.1 ms |
| Film/print, grain disabled | 20.4 ms | 134.8 ms |
| Exact scan conversion + print/scan | 19.2 ms | 47.1 ms |
| Conversion LUT + print/scan | 9.4 ms | 33.7 ms |

The GUI still uses a cached frame of at most 640 pixels while dragging and
refines to the selected preview size afterward. Bounded exposure-meter results
are shared across cached pipeline clones; images and calibration changes
invalidate their respective caches.

Commands, timings, fixture hashes and final suite results are recorded in
[the validation record](validation/2026-10-17-experimental.json).

## Limits

The renderer uses the bundled 81-band 380–780 nm spectral domain; Python's
research API for changing the spectral grid is not exposed by the native app.
CPU and GPU grain use different random streams, so random textures are compared
statistically or replayed with coupled draws. GPU rendering uses f32; f64 CPU
export remains available. CUDA compiles but has not been run on NVIDIA hardware;
unsupported conversion/saturating DIR stages fall back to shared processing.
ICC support remains as documented in `upstream-sync.md`; this migration does not
add profiles for spaces the Python reference itself does not cover.
