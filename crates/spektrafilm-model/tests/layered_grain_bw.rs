//! A B&W emulsion has one clumping field, broadcast to the RGB engine.
use spektrafilm_math::{image::ImageBuf, precision::from_f64};
use spektrafilm_model::grain::{LayeredGrainParams, apply_grain_to_density_layers};
#[test]
fn bw_microstructure_is_shared_across_rgb_storage_channels() {
    let curves: Vec<_> = (0..65).map(|i| [i as f64 / 32.; 3]).collect();
    let layers: Vec<Vec<Vec<f64>>> = curves
        .iter()
        .map(|r| [0.2, 0.3, 0.5].map(|w| vec![r[0] * w; 3]).to_vec())
        .collect();
    let p = LayeredGrainParams {
        rms_granularity: [8.; 3],
        usm_sigma: 0.,
        usm_amount: 0.,
        layer_scale: [2., 1., 0.5],
        density_min: [0.03; 3],
        uniformity: [0.97; 3],
        blur: 0.,
        dye_cloud_blur: 0.,
        micro_structure: [2., 1000.],
        monochrome: true,
        fast_stats: false,
    };
    let image = ImageBuf::from_data(64, 64, vec![from_f64(0.6); 64 * 64 * 3]);
    let cpu = apply_grain_to_density_layers(
        &image,
        5.,
        &curves,
        &layers,
        false,
        &p,
        &spektrafilm_gpu::cpu_backend::CpuBackend,
    );
    assert!(
        cpu.pixels().all(|p| p[0] == p[1] && p[1] == p[2]),
        "B&W microstructure created independent color grain"
    );
    if let Some(gpu) = spektrafilm_gpu::wgpu_backend::WgpuBackend::new() {
        let output = apply_grain_to_density_layers(&image, 5., &curves, &layers, false, &p, &gpu);
        assert!(
            output.pixels().all(|p| p[0] == p[1] && p[1] == p[2]),
            "GPU B&W microstructure created independent color grain"
        );
    }
}
