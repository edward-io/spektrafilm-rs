# Experimental migration checks

Pinned reference: `28bf883e1672e884307edc75852549376e13644e`.
The [parity report](experimental-parity.md) defines the implemented scope and
explicit spectral-grid, precision, RNG and hardware limits.

| Requirement | Evidence |
| --- | --- |
| Profiles, chemistry, layers, bases, DIR | 31 source profiles; Python runtime and septic-model fixtures; curve inversion regression |
| Spectral methods, filters and exposure | All six methods, five filters and source meters/crop/encoded input fixtures |
| RMS physical grain | Source particle sizing, coupled-draw Python processing including mass-preserving USM; CPU/GPU law, tone, variance, repeatability and BW checks |
| Conversion workflows | Every route on CPU/GPU; exact and tetrahedral LUT conversion; matrix/EV cases |
| Calibration actions | Synthetic clear-film fit, ideal device scan and tuned-base printer neutrality; worker snapshot handling |
| GUI and project behavior | Defaults/Python import/roundtrip, tooltip interaction, cached versus rebuilt previews, loading/errors, RAW applied-state snapshots and export workers |
| GPU accuracy | Existing HDR/edge/optical/color stress cases, all bundled stock/paper cases and every named transport; max absolute normalized error ≤ 0.001 |
| LUT exports | Python bake fixtures, current-params baking, neutral bake contracts, midgray/EV gain, CLI/bundle overrides and format/interoperability checks |
| RAW/saving/metadata/ICC | Existing shared GUI/CLI fixtures and regressions; actual Sony RAW preview benchmark |
| Performance/builds | Release GUI and CLI plus f64 helper; cached 640-pixel GPU previews; exact/LUT conversion timings |

Final run counts, timings and source identifiers are in
[the validation record](validation/2026-10-17-experimental.json).
Historical main-branch audit results remain in `validation/` with their original
source versions and are not numerical golden outputs for experimental.
