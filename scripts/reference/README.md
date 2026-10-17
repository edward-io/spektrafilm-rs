# Python reference generators

The current model reference is experimental commit
`28bf883e1672e884307edc75852549376e13644e`. Set `PYTHONPATH` to that
checkout's `src` directory and use its pinned dependencies. For example:

```sh
PYTHONPATH=/path/to/experimental/src MPLBACKEND=Agg python scripts/reference/generate_experimental_fixture.py
PYTHONPATH=/path/to/experimental/src MPLBACKEND=Agg python scripts/reference/generate_grain_replay_fixture.py
```

The experimental generator captures real runtime results for 51 configurations,
septic chemistry/layers, defaults, converter values and stage references. Taking
filter samples are generated through the source's Akima interpolator. The grain
oracle supplies identical particle/clumping draws to the original processing
functions, independently checking blur, RMS sizing and multiplicative USM.
Other generators cover transports, LUT bakes, interpolation, image IO and QA.
These scripts are development tools; the Rust application does not invoke Python.

`generate_raw_fixture.py` also needs tifffile and the reference RAW libraries.
These are development tools; Rust runtime code does not invoke Python.

After building `spektrafilm-f64`, run the format/bundle interoperability smoke
check with Python's format plugins and PyOpenColorIO 2.5 or newer:

```sh
PYTHONPATH=/path/to/experimental/src MPLBACKEND=Agg python scripts/reference/verify_lut_outputs.py \
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
PYTHONPATH=/path/to/experimental/src MPLBACKEND=Agg python scripts/reference/verify_qa_gamut_report.py /path/to/bundle
```

The normal runtime and packaged app never invoke these development scripts.
