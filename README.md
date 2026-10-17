# spektrafilm-rs

![banner](docs/card.jpg)

A spectral simulator for analogue film and the print-and-scan chain, with a
native Rust GUI and CLI. This repository,
[edward-io/spektrafilm-rs](https://github.com/edward-io/spektrafilm-rs), is a fork
of [turbasvin/spektrafilm-rs](https://github.com/turbasvin/spektrafilm-rs), the Rust
port of [Andrea Volpato's spektrafilm](https://github.com/andreavolpato/spektrafilm).

The runtime implements RGB → spectral film exposure → dye density → enlarger
and print paper → scanner RGB. It provides a native GUI, a batch CLI, calibrated
LUT exports, full-resolution f32 GPU export and f64 CPU export. The implementation
is checked against the Python experimental branch at `28bf883` and runs without Python.
GPU previews use f32; numerical parity is measured with tolerances.
See the [changelog](#changelog) for changes from the original Rust fork,
[reference status and limits](docs/upstream-sync.md), and
[validation checklist](docs/parity-checklist.md) for the supporting evidence.

## Build and run

Requires Rust **1.89 or newer**, a C++ toolchain, `pkg-config`, and GLib
development headers and static archives. On macOS, install the build
dependencies with `brew install glib pkg-config`. Lensfun itself is built into
the executable; LibRaw is an optional runtime dependency.

```sh
git clone git@github.com:edward-io/spektrafilm-rs.git
cd spektrafilm-rs

cargo build --release -p spektrafilm-gui
cargo build --release -p spektrafilm-cli --bin spektrafilm
cargo build --release -p spektrafilm-cli --bin spektrafilm-f64 --features precision-f64

./target/release/spektrafilm-gui
# Or open an image on launch:
./target/release/spektrafilm-gui /path/to/image.ARW
```

These commands place the binaries in `target/release/` and run from the
repository root, using its `data/` directory. If you build with a custom
`--target-dir`, use its `release/` directory instead. When moving the GUI
elsewhere, set `SPEKTRAFILM_DATA_DIR` to the data directory; CLI commands accept
`--data-dir`.

## Changelog

### Unreleased

Changes from the original Rust fork's v0.1.2 baseline,
[turbasvin/spektrafilm-rs at `9dd59b0`](https://github.com/turbasvin/spektrafilm-rs/commit/9dd59b0380194b93686aaa230a8bb9680aa270a4):

- **Experimental parity:** tracks Python experimental at
  [`28bf883`](https://github.com/andreavolpato/spektrafilm/commit/28bf883e1672e884307edc75852549376e13644e),
  with all 31 fitted profiles and stock presets; film/print chemistry and base
  controls; saturating DIR; RMS physical grain and mass-preserving sharpening;
  six spectral methods, taking filters and film-raw exposure metering.
- **Scan conversion and calibration:** all six workflow routes, exact and
  tetrahedral-LUT negative conversion, scan exposure/device matrices, and background
  base detection, blind device calibration and neutral print balance. Explicit
  workflows take precedence over legacy scan flags when importing and exporting.
- **LUTs and tools:** calibrated CUBE, 3DL, Lumix and Hald export; multi-stage
  bundles, TOML/JSON specs, OCIO and QA reports; profile measurements and
  calibration strips. Experimental LUT bakes preserve native midgray, accept
  exposure EV and carry current chemistry, base and spectral settings.
- **Color and settings:** expanded named color spaces, transfer functions and
  gamut mapping; independent saving-space conversion; restored crop, spectral,
  filter and other previously ignored runtime controls.
- **Grain and GPU:** physical three-sublayer grain with dye-cloud blur and
  microstructure; wgpu implementations of layered grain, spatial filters,
  diffusion and color transforms; resident GPU scan conversion and mass-preserving
  grain sharpening; full-resolution GPU export.
- **RAW and lenses:** shared GUI/CLI I/O, rawler 0.8 with Sony ILCE-7M5 support,
  optional LibRaw with Auto fallback, built-in Lensfun, automatic database
  download/cache, corrected RGB vignetting handling and clearer load errors.
- **Lightroom catalogs:** read-only `.lrcat` browsing with background thumbnails,
  pick flags, virtual copies and keyboard navigation; crop, straighten, rotation
  and flips carried through previews, saved projects and GPU/CPU exports.
  Navigation preserves current workflow settings and caches recently developed
  images. Lightroom tonal edits and Adobe lens/perspective corrections are not imported.
- **GUI and previews:** project/default persistence, Python GUI state import,
  rotation, contextual hover help, layer/comparison controls and paper animation;
  cached pipelines, scaled inputs and exposure-meter results, with adjustable
  resolution, 640-pixel interactive frames and background workers.
- **Reference validation:** 51 experimental Python runtime cases on CPU/GPU,
  independent grain/LUT fixtures, f32/f64 suites and GUI regressions. The
  [parity report](docs/experimental-parity.md) records preview timings, the strict
  0.001 GPU error gate and remaining limits.

## Lightroom Classic catalogs

Choose **Catalog…**, drop a `.lrcat` file, or pass it to `spektrafilm-gui` on
launch. The left browser shows photo thumbnails and Lightroom's accepted,
rejected and unflagged states. **Accepted only** filters picks; click a photo,
use **Previous / Next**, or press the left/right arrow keys to browse.
Unavailable originals are skipped. Connect the photo drive and press
**Refresh** to update availability, flags and geometry.

Catalogs are opened read-only. Original files are developed by Spektrafilm,
with Lightroom's crop, straighten angle, rotation and flips applied. Lightroom
color, exposure, white balance and other tonal adjustments are not imported.
Imported geometry is retained in saved Spektrafilm projects and full-resolution
GPU/CPU exports. Camera previews are used only as browser thumbnails.

Adobe lens and perspective corrections are not reproduced; affected photos
show a warning because their framing can differ. The importer targets the
current Lightroom Classic catalog layout; unsupported layouts report an error.
The photo list is virtualized, thumbnails load in the background, and recently
developed images use a bounded memory cache for faster revisits.

Use an optimized build when browsing full-size RAWs; plain `cargo run` uses the
much slower debug build:

```sh
cargo run --release -p spektrafilm-gui -- "/path/to/catalog.lrcat"
```

To measure selection through the rendered preview on the first six available
RAWs (without preloading), run the local benchmark. Set
`SPEKTRAFILM_BENCH_RAW_DECODER=rawler` or `libraw` to compare decoders. Landscape
and portrait source previews are saved in `target/catalog-load-benchmark/`:

```sh
SPEKTRAFILM_BENCH_LIGHTROOM_CATALOG="/path/to/catalog.lrcat" \
  cargo test --release -p spektrafilm-gui benchmark_catalog_load -- --ignored --nocapture
```

## GUI and RAW development

Open or drop JPEG, PNG, TIFF, EXR or camera RAW files. RAW support includes the
Sony ILCE-7M5 through rawler 0.8. Auto tries rawler first, then falls back to an
installed LibRaw if decoding fails. LibRaw is optional; configuration and camera
regression instructions are in the [support notes](docs/upstream-sync.md). Lensfun is built in,
so lens correction needs no separate Lensfun installation or library-path setting.
Its database is found automatically; when none is installed, the loading worker
downloads the official compatible database once and caches it in your user data
directory. The folder picker is an optional override.

The GUI exposes film/paper, exposure, optical effects, layered grain, color,
crop/resize, RAW development and output controls. Project files preserve these
settings; Python GUI state JSON can be imported. Loading and rendering run in
background workers, with progress and errors in the bottom status bar.
Hover over control labels, section headings or dropdown entries for a short explanation.
Slider tracks and numeric editors do not show the control’s explanation.
Set **Preview long edge (px)** in the right sidebar to change preview size.
This is independent of the simplified-effect preview mode: the default is 2048
pixels on the long edge, dropping to at most 640 while dragging a control.
Scaled inputs are cached, and releasing the control refines the preview.
Auto preview can be disabled for manual rendering. RAW decoder, white-balance
and lens settings take effect with **Reload RAW with these settings**. Lens
correction reports whether the database is downloaded, installed locally or not
downloaded yet. Hover over an available database's status to see its location.
The first download requires internet access; subsequent loads use the cache offline.
Projects retain the RAW settings that produced the loaded image; pending RAW
edits become part of the project after Reload. Image paths remain valid when
the project is saved in another folder.

**Save preview** writes the bounded preview. **Export** renders the original at
full resolution on the GPU; disable GPU export under File export for the f64
CPU path. Both support cancellation (a GPU dispatch finishes before cancelling).
The LUT export panel supports individual files, stage bundles and diagnostic
reports; Build from bundle spec supports multi-paper/ZIP delivery. Place
`spektrafilm-f64` next to the GUI or set `SPEKTRAFILM_F64_CLI` for CPU/LUT export.
Project/default files use the platform's application configuration directory.
PNG/JPEG quantize and clip; float TIFF/EXR preserve
negative values and highlights above 1. Preview and saving color spaces are
independent; output files are converted to the saving space before encoding.
EXR always stores linear values with RGB chromaticities. Untagged EXR uses
linear sRGB; unsupported custom primaries produce an explicit error. Embedded
RGB ICC profiles are recognized or converted to linear sRGB during loading.
Failed image or LUT writes preserve the previous output file.

## CLI

```sh
# Reference precision, CPU
SPEKTRAFILM_BACKEND=cpu ./target/release/spektrafilm-f64 process input.ARW \
  --film kodak_portra_400 --paper kodak_portra_endura -o output.tif \
  --bit-depth 32 --data-dir data

# Runtime parameters are JSON; --scan-film bypasses paper printing
./target/release/spektrafilm process input.png --film kodak_gold_200 \
  --params settings.json -o output.png

./target/release/spektrafilm list-profiles --data-dir data
./target/release/spektrafilm list-color-spaces
```

RAW controls: `--raw-decoder auto|libraw|rawler`, `--white-balance
as_shot|daylight|tungsten|custom`, `--temperature`, `--tint`, `--lens-correction`,
and `--lens-database`. `--saving-color-space` and `--saving-cctf-encoding`
control the saved color transport independently of processing. JPEG quality
and TIFF/EXR bit depth are configurable.

## LUTs

```sh
# A calibrated, deterministic log-to-display CUBE
./target/release/spektrafilm-f64 export-lut --film kodak_portra_400 \
  --paper kodak_portra_endura --input-space vlog --output-space srgb \
  --size 33 -o portra.cube --data-dir data

# Expose film/print boundaries and optional collapsed combinations
./target/release/spektrafilm-f64 export-lut --film kodak_portra_400 \
  --paper kodak_portra_endura --input-space vlog --output-space srgb \
  --topology 4lut --include-combinations --size 33 -o portra-bundle

# Python-compatible BundleSpec fields in TOML or JSON
./target/release/spektrafilm-f64 build-luts --from bundle.toml --resolution 65 -o bundles
```

Supported formats: `--format cube|3dl|lumix|hald_png`. Lumix requires V-Log and
17/33/65 samples per axis and a single-stage bundle; Hald requires a square
resolution such as 16 or 64.
Bundles support 1/2/3/4-stage topologies, multiple papers, ZIP, wire metadata,
optional OCIO, and native scientific QA reports (`--qa` on bundle export).
Spatial effects, grain, glare and automatic exposure are disabled for baking.
Bundles are built in a temporary directory before replacing the previous
bundle, removing stale QA and LUT files. Use an empty folder or an existing
bundle folder; unrelated files cause an error and are preserved. ZIP delivery
includes only the newly built bundle.

Example `bundle.toml`:

```toml
film_profile = "kodak_portra_400"
print_profiles = ["kodak_portra_endura"]
input_color_space = "vlog"
output_color_space = "srgb"
name = "portra"
topology = "4lut"
resolution = 33
container = "zip"
qa = true
ocio_config = true
include_combinations = true
```

## Profile measurements and calibration

```sh
./target/release/spektrafilm-f64 measure-profile kodak_portra_400 \
  --output measurements.json --plot-dir profile-plots
./target/release/spektrafilm-f64 calibration-strip input.png \
  --film kodak_portra_400 --paper kodak_portra_endura \
  --kind negative_exposure --values=-2,-1,0,1,2 -o exposure-strip.png
```

Measurement output includes gamma, local exposure slopes and toe fits. Calibration
supports exposure, print exposure, RMS grain granularity, DIR amount and glare,
with crop/rotation, labeled SVG previews and JSON sidecars. Run either command
with `--help` for its controls.

## Backends and tests

`SPEKTRAFILM_BACKEND=cpu|wgpu` selects the compute backend. macOS uses Metal
through wgpu by default. An experimental CUDA backend is available with
`--features spektrafilm-gpu/cuda-backend` and `SPEKTRAFILM_BACKEND=cuda`.
The physical three-sublayer grain model runs inside the resident wgpu chain.
Unsupported resident-GPU operations use staged processing. The f64 CPU path is
the reference export path. Validation focuses on Apple Silicon / Metal; CUDA
has been compile-checked but has not been validated at runtime on this machine.
Grain textures can differ between backends, and rawler/LibRaw use different
demosaic, crop and color pipelines. See the linked validation notes for limits.

```sh
cargo test --workspace --features precision-f64
cargo test --workspace
```

The math/model/core crates implement the numerical pipeline;
`spektrafilm-io` provides shared application I/O; CLI and GUI are separate
frontends. Runtime profiles, spectral LUTs, filters and ICC assets live in
`data/`. macOS/Windows packaging helpers live in `scripts/`.

## Credits and license

Original Rust port by [turbasvin](https://github.com/turbasvin/spektrafilm-rs).
This fork extends that implementation.

Original Python implementation, spectral data and film/paper profiles by
[Andrea Volpato](https://github.com/andreavolpato/spektrafilm). The Hanatos2025
spectral-upsampling basis comes from work by
[Johannes Hanatos](https://github.com/hanatos). Numerical methods follow the
reference's NumPy, SciPy, scikit-image and colour-science implementations.

GPL-3.0; see [LICENSE](LICENSE) and bundled data licenses.
