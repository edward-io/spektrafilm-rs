# Rust application cache audit

The October 18, 2026 audit covers cache keys, invalidation, eviction, memory
limits and failed publication in the Rust application. It follows the catalog
thumbnail flicker fix. The changes preserve image precision and keep Lightroom
catalogs read-only.

## Findings and fixes

| Cache | Problem | Result |
| --- | --- | --- |
| Decoded RAW images | An oversized background result could flush useful cached images before evicting itself. A closed photo could remain exempt from the budget. | Reject oversized background entries before eviction, and remove the current-photo exemption when clearing the image. Keep nearest-first eviction and LRU ordering for revisits. |
| Decoded RAW invalidation | Reframing old pixels recorded the current file timestamp, allowing those pixels to masquerade as a fresh decode after an external file change. | Preserve the fingerprint captured by the original decode through crop and rotation. File fingerprints include creation time and, on Unix, device, inode and change time, detecting replacements that preserve size and modification time. |
| RAW preload window | Saving or resetting startup defaults did not invalidate the previously resolved neighbor settings. | Recompute the window after these actions. Existing decoded entries remain reusable only under their complete RAW settings and rotation keys. |
| Catalog thumbnails | Refresh retained old paths/orientations and cached failures. Results from earlier requests could repopulate the refreshed cache. | Clear thumbnails and failed reads on successful refresh; tag requests/results with a generation and reject older results before changing pending state. |
| Paper-back textures | FIFO eviction ignored repeated access; simultaneous cold builds could occupy multiple slots for the same aspect. | Promote hits and remove duplicate keys before insertion. Retain eight normalized aspect sizes. |
| Input OKLCH boundary | A global `OnceLock` permanently reused the first polygon. Repeated boundary changes also smoothed the previous hull, and switching to the locus did not restore it. | Cache the latest table by the exact polygon, retain its immutable table per compressor, and clear the instance lookup when replacing the boundary. Derive each hull from the spectral locus. |
| Illuminant spectra | Both precision caches leaked allocations for every distinct temperature or spelling. | Remove the caches and return owned 81-sample arrays. Update all callers to borrow those arrays; preserve every spectral calculation. |
| GPU compute pipelines | Cache keys contained only the shader pointer, despite accepting different binding layouts. Compilation was duplicated in two helpers. | Share one compilation path keyed by source identity, source length and complete binding layout. |
| QA noise gradients | A display name and gain did not fully describe the actual primaries/encoding. Concurrent misses could insert duplicate keys. | Key by primaries, transfer function and resolved gain; replace duplicate keys on publication. Preserve the four-entry LRU. |
| GUI calibrated pipeline | Parameter keys omitted the data directory and on-disk calibration inputs. | Include selected profile fingerprints, spectral LUTs/descriptors, Arctic reflectance and neutral filters. Check files on the render worker. Preserve the existing calibration/runtime parameter distinction. |
| Lensfun snapshot installation | Publication failure occurred after the staging directory had been made permanent, leaving an unreachable snapshot. | Keep staging temporary until the snapshot pointer is published successfully. Failed publication removes the unpublished directory. |

## Capacity and invalidation policies

| Cache | Retention and invalidation |
| --- | --- |
| Catalog thumbnails | 256 entries, LRU on display; refresh generation prevents stale results. Failed reads occupy a bounded slot and retry on refresh. Source-preview fallbacks share the same limit. |
| Decoded originals | 2 GiB of additional pixel storage with prefetch enabled, 512 MiB for revisits otherwise. The displayed original is exempt while selected. Keys include absolute path, RAW options, Lightroom geometry and manual rotation; foreground reuse verifies the source fingerprint. |
| GUI pipeline and inverse conversion LUT | One calibrated pipeline, replaced when its calibration key changes. Its conversion LUT is owned by that pipeline. Exposure/calibration adjustments applied at runtime remain outside the inverse LUT. |
| Scaled preview inputs | Two sizes for the current image. Successful render publication updates the stored size; changing source pixels or framing clears the entries. |
| Source/output display textures | Current displayed images plus bounded comparison/transition state. Source display keys include color space, encoding and display-transform state; loading/reframing replaces the source. |
| Exposure metering | One content/settings fingerprint and scale shared by compatible pipeline clones. Meter pixels, dimensions, method, input encoding/space and adaptation settings enter the key. |
| Input XY mapping | One exact geometry key and coordinate array. Keys include grid size, knee, white and polygon. LUT values are sampled afresh. |
| Input OKLCH tables | One shared latest polygon/table, plus a table retained by each live compressor. No polygon comparison or global mutex in the per-sample lookup. |
| Output gamut tables | Finite compiled color-space registry × four perceptual algorithms. Knee/lightness controls do not change the boundary bake. No dynamic profile names or temperatures enter this cache. |
| QA noise gradient | Four LRU entries; each raster is fixed at 512 × 512 RGB f64 samples. |
| Paper back | Eight LRU entries; each image has a longest dimension of 1024 pixels. |
| GPU diffusion spectra | Eight LRU entries and a 1 GiB retained-buffer limit. Reject an entry larger than the budget. Keys include FFT shape, PSF support and every PSF coefficient bit. Publish only after submitting spectrum initialization; queue order protects other consumers. |
| GPU preview buffers | One input and one output buffer, grown when needed. Retain the largest capacities seen; reject requests above the device binding limit and use the CPU fallback. Overwrite current pixels and pass current dimensions on every operation. |
| Compiled GPU pipelines | Finite static shader sources and layouts supplied by the application. Backend/device ownership prevents reuse across devices. Scheduling changes do not create shader specializations. |
| Presets, taking filters, logo and inspector row metadata | Compiled immutable assets or the finite set of application controls. Process-lifetime retention is intentional. RNG spare-word state is fixed-size generator state. |
| Lensfun and embedded application data on disk | Versioned persistent snapshots, with no automatic generation eviction. A valid Lensfun snapshot is reused offline. Embedded data is namespaced by the bundle identifier. Retaining generations avoids deleting data still used by another process. |

Cache budgets govern retained cache entries, rather than all application memory.
Active images, outstanding render/export work and driver allocations can add to
the retained cache storage. File fingerprints avoid reading entire RAW/LUT files
on every cache lookup; they are metadata fingerprints, not content hashes.

## Regression coverage

New tests cover separate OKLCH boundaries and boundary replacement, preserved
RAW fingerprints through rotation, same-size replacements with preserved mtime,
RAW LRU/nearest eviction, oversized entries and unpinning, refreshed catalogs
with late thumbnail results, paper-back LRU and concurrent duplicate insertion,
owned illuminants across 2048 temperatures, QA encoding/primaries changes,
calibration file changes, GPU pipeline layout variants, GPU diffusion entry/byte
limits and oversized entries, preview buffer growth/shrink with fresh pixels,
and cleanup after forced Lensfun publication failure. Existing cached-versus-
rebuilt pipeline, metering, RAW setting, crop, decoder and numerical reference
tests remain in place.

The OKLCH boundary and RAW reframe regressions were run against their previous
implementations and failed before the fixes. Capacity tests use small budgets
and tiny buffers; they exercise eviction without allocating gigabytes or filling
the user's disk.

Validation includes full f32 and f64 workspace suites, focused tests added after
those runs, all-target compilation, and release GUI/export-helper builds. The
full workspace runs passed 263 tests in f32 and 274 in f64, with five intentionally
ignored tests per run. The final GUI run passed 73 tests with four ignored;
GPU/I/O library runs passed 43 tests in f32 and 48 in f64, with the live database
download ignored. The final replacement-fingerprint test also passed in f64.
All-target compilation and both release builds succeeded.

GPU execution uses this Mac's wgpu/Metal backend. CUDA, Vulkan, DX12 and the
Windows embedded-data extraction path were not exercised. The live app was not
restarted automatically; it must be reopened to use the rebuilt executable.
