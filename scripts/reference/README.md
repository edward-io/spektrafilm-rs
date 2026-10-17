# Python reference generators

Run from the Rust repository root using the adjacent Python repository and its
pinned dependencies (colour-science 0.4.6):

```sh
PYTHONPATH=../src MPLBACKEND=Agg python scripts/reference/generate_color_spaces.py
PYTHONPATH=../src MPLBACKEND=Agg python scripts/reference/generate_illuminants.py
PYTHONPATH=../src MPLBACKEND=Agg python scripts/reference/generate_lut_spaces.py
PYTHONPATH=../src MPLBACKEND=Agg python scripts/reference/generate_ocio.py
PYTHONPATH=../src MPLBACKEND=Agg python scripts/reference/generate_parity_fixtures.py
PYTHONPATH=../src MPLBACKEND=Agg python scripts/reference/generate_preview_fixtures.py
PYTHONPATH=../src MPLBACKEND=Agg python scripts/reference/generate_transport_fixtures.py
PYTHONPATH=../src MPLBACKEND=Agg python scripts/reference/generate_qa_fixtures.py
PYTHONPATH=../src MPLBACKEND=Agg python scripts/reference/generate_saving_fixtures.py
PYTHONPATH=../src MPLBACKEND=Agg python scripts/reference/generate_raw_fixture.py
PYTHONPATH=../src MPLBACKEND=Agg python scripts/reference/generate_gui_bindings.py
```

`generate_raw_fixture.py` also needs tifffile and the reference RAW libraries.
These are development tools; Rust runtime code does not invoke Python.

After building `spektrafilm-f64`, run the format/bundle interoperability smoke
check with Python's format plugins and PyOpenColorIO 2.5 or newer:

```sh
PYTHONPATH=../src MPLBACKEND=Agg python scripts/reference/verify_lut_outputs.py \
  --cli target/release/spektrafilm-f64
```

Additional numerical oracles cover Gaussian borders / compensated IIR, exact-law
grain with coupled Python draws, source exposure meters, parametric curves,
measurement / calibration helpers, density-pinned characteristic probes,
CCT/dynamic-range probes, noise polar/gradient patterns and wide-gamut stress.
`generate_qa_gamut_fixture.py` captures the actual Python input-gamut and full-cube
envelope functions. `generate_qa_normal_tables.py` / `generate_hull_sampling_fixture.py`
preserve the pinned NumPy streams; their third-party license is in `data/licenses`.

To compare a native one-LUT bundle's full gamut diagnostics with the actual
Python QA functions (the bundle must have `--qa` and default gamut settings):

```sh
PYTHONPATH=../src MPLBACKEND=Agg python scripts/reference/verify_qa_gamut_report.py /path/to/bundle
```

The normal runtime and packaged app never invoke these development scripts.
