//! Statistical and physical-control checks: GPU noise need not share CPU pixels.
use spektrafilm_math::{image::ImageBuf, precision::from_f64};
use spektrafilm_model::grain::{LayeredGrainParams, apply_grain_to_density_layers};
fn stats(image: &ImageBuf, c: usize) -> (f64, f64) {
    let n = image.pixel_count() as f64;
    let mean = image.pixels().map(|p| p[c] as f64).sum::<f64>() / n;
    let variance = image
        .pixels()
        .map(|p| (p[c] as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    (mean, variance)
}
#[test]
fn gpu_layered_grain_matches_cpu_statistics_and_controls() {
    let Some(gpu) = spektrafilm_gpu::wgpu_backend::WgpuBackend::new() else {
        return;
    };
    let cpu = spektrafilm_gpu::cpu_backend::CpuBackend;
    let mut curves = Vec::new();
    let mut layers = Vec::new();
    for i in 0..65 {
        let x = i as f64 / 64.;
        let row: Vec<Vec<f64>> = [(0.2, 0.7), (0.3, 1.2), (0.5, 1.8)]
            .into_iter()
            .map(|(fraction, power)| {
                [2.4, 1.8, 1.2]
                    .map(|max| max * fraction * x.powf(power))
                    .to_vec()
            })
            .collect();
        curves.push(std::array::from_fn(|c| row.iter().map(|r| r[c]).sum()));
        layers.push(row);
    }
    let input = ImageBuf::from_data(
        256,
        256,
        (0..256 * 256)
            .flat_map(|_| [1., 0.6, 0.3].map(from_f64))
            .collect(),
    );
    let mut p = LayeredGrainParams {
        particle_area: 0.2,
        channel_scale: [1.6, 1.6, 3.2],
        layer_scale: [2., 1., 0.5],
        density_min: [0.03; 3],
        uniformity: [0.97, 0.99, 0.97],
        blur: 0.,
        dye_cloud_blur: 0.,
        micro_structure: [0., 0.],
        monochrome: false,
        fast_stats: false,
    };
    for case in 0..3 {
        if case == 1 {
            p.blur = 0.65;
            p.dye_cloud_blur = 5.;
        }
        if case == 2 {
            p.micro_structure = [2., 1000.];
            curves.reverse();
            layers.reverse();
        }
        let a = apply_grain_to_density_layers(&input, 5., &curves, &layers, case == 2, &p, &cpu);
        let b = apply_grain_to_density_layers(&input, 5., &curves, &layers, case == 2, &p, &gpu);
        let repeat =
            apply_grain_to_density_layers(&input, 5., &curves, &layers, case == 2, &p, &gpu);
        assert_eq!(b.data, repeat.data, "unstable GPU texture");
        assert!(b.data.iter().all(|v| v.is_finite()));
        for c in 0..3 {
            let (ma, va) = stats(&a, c);
            let (mb, vb) = stats(&b, c);
            eprintln!("case {case} channel {c}: CPU {ma} / {va}, GPU {mb} / {vb}");
            assert!((ma - mb).abs() < 0.004, "grain tone changed");
            assert!(
                (0.94..1.06).contains(&(vb / va)),
                "grain variance drift: {}",
                vb / va
            );
        }
    }
    // Larger particles raise variance without changing the mean exposure.
    curves.reverse();
    layers.reverse();
    p.blur = 0.;
    p.dye_cloud_blur = 0.;
    p.micro_structure = [0., 0.];
    let a = apply_grain_to_density_layers(&input, 5., &curves, &layers, false, &p, &gpu);
    p.particle_area *= 4.;
    let b = apply_grain_to_density_layers(&input, 5., &curves, &layers, false, &p, &gpu);
    for c in 0..3 {
        let (ma, va) = stats(&a, c);
        let (mb, vb) = stats(&b, c);
        assert!((ma - mb).abs() < 0.004);
        assert!(
            (3.2..4.8).contains(&(vb / va)),
            "particle area did not scale grain variance: {}",
            vb / va
        );
    }
    // The explicit approximate-statistics setting must select a different
    // sampling path; it was previously ignored by the GPU implementation.
    let exact = apply_grain_to_density_layers(&input, 5., &curves, &layers, false, &p, &gpu);
    p.fast_stats = true;
    let fast = apply_grain_to_density_layers(&input, 5., &curves, &layers, false, &p, &gpu);
    assert_ne!(exact.data, fast.data, "use_fast_stats is ignored");
    for c in 0..3 {
        let (a, va) = stats(&exact, c);
        let (b, vb) = stats(&fast, c);
        assert!((a - b).abs() < 0.004);
        assert!((vb / va - 1.).abs() < 0.06);
    }
}
