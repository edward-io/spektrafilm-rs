# Reference parity and application support

Reference: Python spektrafilm experimental at
`28bf883e1672e884307edc75852549376e13644e` (2026-07-08).
See [experimental parity](experimental-parity.md) for model, app, LUT contract
and verification details. This replaces the earlier `3bb2c2d` main baseline.
Historical measurements in `validation/` describe their recorded source version.

The runtime includes film/print chemistry and base tuning, RMS physical grain,
nonlinear DIR, six spectral methods, taking filters, film-raw exposure metering,
all six workflow routes and scan-conversion calibration. wgpu runs the normal
film chain and exact/LUT scan conversion on the GPU. Small previews remain
bounded and cached; full-resolution GPU export and f64 CPU export are available.

The native GUI and CLI retain project/default persistence, Python GUI-state
import, rotation, label tooltips, layer comparison, interpolation modes, paper
animation, LUT formats/bundles, OCIO and scientific diagnostics. RAW loading and
exports run on workers. Saving has an independent color space/encoding and
preserves supported image metadata.

## RAW and lenses

Rawler 0.8 includes Sony ILCE-7M5 support. The user's RAW was checked with rawler,
LibRaw and Auto and decoded to the active 7008 × 4672 image. LibRaw uses its
public C API via dynamic loading. Auto tries rawler and falls back to LibRaw.
`SPEKTRAFILM_LIBRAW_LIBRARY` optionally selects an installed LibRaw library.

Default builds include static Lensfun 0.3.4. An installed database is detected,
or the app downloads and caches it on first use. The GUI shows whether the
cache is available. `SPEKTRAFILM_LENSFUN_DATABASE` or an explicit directory can
override database selection. `SPEKTRAFILM_LENSFUN_LIBRARY` is an optional
library override, not required by the default build. Building the bundled
library requires GLib headers/static archives and pkg-config; on macOS use
`brew install glib pkg-config`. GLib/gettext/PCRE2 are linked statically.

## Precision and verification limits

Deterministic GPU comparisons retain the 0.001 maximum absolute normalized RGB
gate. This is a full-scale tolerance, not a relative guarantee near black.
Grain uses distinct CPU/GPU RNGs; tests check its law, mean and variance, plus
independent Python processing with identical supplied draws. CUDA is compile
checked; its unported resident stages use shared fallbacks. Adapter memory and
dispatch limits still constrain full-resolution exports.

The fixed bundled spectral grid and ICC limitations below are unchanged.

Embedded ICC profiles currently cover sRGB, ProPhoto, Adobe RGB, Rec.709, Rec.2020,
ACES2065-1, ACEScg, and encoded Display P3/DCI-P3. Other runtime output spaces retain their numeric encoding but do
not yet get a generated ICC profile. Other embedded RGB ICC profiles are
converted to linear sRGB on import; malformed or unsupported profiles fail
explicitly. EXR stores linear float pixels and standard RGB chromaticities;
untagged files default to linear sRGB, and unknown custom primaries fail
explicitly. EXR does not copy EXIF. rawler and LibRaw have different demosaic/crop/color pipelines, so their
RAW pixels are not interchangeable parity baselines.
