# GUI tooltip audit

Reference: Python spektrafilm at `3bb2c2d2801ff68b92019cf1dbcbb133d60832bc`
(the v0.3.4 development reference). Reviewed the GUI parameter manifest alongside
the runtime formulas; descriptions follow the implementation when older GUI
text disagrees with it. Application-specific actions describe the Rust GUI.

| Controls | Verified meaning and corrections | Python source |
| --- | --- | --- |
| Halation | Renormalization preserves midgray by dividing out added halation energy; bounce weights are already normalized separately. Gaussian widths and exponential decay lengths are distinguished. | `model/diffusion.py:apply_halation_um` |
| DIR | Matrix rows are donor layers and columns are receivers; diffusion combines a Gaussian core and an exponential tail. | `model/couplers.py` |
| Diffusion filter | Strength is a commercial filter grade mapped nonlinearly to a scattering fraction. Warmth redistributes color between inner and outer halos. Core/halo/bloom weights are relative and normalized. The implementation conserves energy, including Black Pro-Mist, despite older manifest text describing absorption. | `model/diffusion.py:_strength_to_scatter`, `_halo_channel_weights`, `_resolve_family_cfg` |
| Grain | Physical grain uses three sublayers per RGB layer. Post-blur sigma is in pixels; particle scales multiply particle area. Dye-cloud blur is particle-dependent. | `model/grain.py`, `model/density_curves.py`, GUI parameter manifest |
| Print chemistry | Gamma slow affects the middle and slow sublayers. Developer exhaustion changes the shoulder while preserving midgray. | `utils/morph_curves.py` |
| Preflash | A uniform exposure through the film base is added linearly; its amount is not an EV value despite the older manifest tooltip. | `runtime/stages/printing.py:_compute_raw_preflash` |
| Exposure and geometry | Matrix metering uses a center-weighted 5×5 grid; multi-zone uses three concentric regions. Film format and crop dimensions refer to the image's long edge. | `utils/autoexposure.py`, GUI parameter manifest, `runtime/services/resize.py`, `utils/crop_resize.py` |
| Spectral reconstruction | Spectral blur smooths reconstructed spectra, rather than the film sensitivity curves. The adaptation surface is an illuminant-dependent polynomial correction. | `utils/spectral_upsampling.py`, `runtime/stages/filming.py` |
| Scanner and glare | Scan blur/unsharp sigma and glare blur are in pixels; glare percent is divided by 100. White and black levels are correction targets. | `runtime/stages/scanning.py`, `model/glare.py`, GUI parameter manifest |
| Simplified preview | Grain, selected blur operations and sharpening are disabled; halation and lens diffusion remain. Preview-size limits are separate from export input resolution. | `runtime/params_builder.py:digest_params`, GUI display manifest; Rust preview/export workers |
| Color, LUTs and display | Color-space/transfer descriptions follow the native registries and Python options. Native display conversion and LUT delivery descriptions reflect the Rust behavior. | GUI manifests/options, `spektrafilm_lut_creator`; Rust `lut_spaces.rs`, GUI display/export code |

Explanatory help is attached to separate text captions, not slider tracks or
numeric editors. Dropdown choices and action buttons keep their own help.
Headless egui interaction regressions exercise caption hover versus editable
areas and checkbox caption clicks.
