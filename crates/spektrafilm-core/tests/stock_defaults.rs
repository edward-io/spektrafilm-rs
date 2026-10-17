use spektrafilm_core::{
    params::RuntimeParams,
    pipeline::{Pipeline, apply_stock_defaults},
    profile,
};
#[test]
fn gui_stock_seeding_preserves_subsequent_low_level_edits() {
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let film = profile::load_profile_by_name(&data, "kodak_portra_400").unwrap();
    let paper = profile::load_profile_by_name(&data, "kodak_portra_endura").unwrap();
    let mut params = RuntimeParams::default();
    apply_stock_defaults(&film, &mut params);
    assert!(!params.settings.apply_stock_specifics);
    assert_eq!(
        params.film_render.dir_couplers.gamma_samelayer_rgb,
        [0.336, 0.319, 0.273]
    );
    params.film_render.dir_couplers.gamma_samelayer_rgb = [0.1, 0.2, 0.3];
    params.film_render.halation.halation_strength = [0.01, 0.02, 0.03];
    params.film_render.halation.halation_first_sigma_um = [2., 4., 6.];
    let pipeline = Pipeline::new_with_spectral(film, paper, params.clone(), &data).unwrap();
    assert_eq!(
        pipeline.params.film_render.dir_couplers.gamma_samelayer_rgb,
        [0.1, 0.2, 0.3]
    );
    assert_eq!(
        pipeline.params.film_render.halation.halation_strength,
        [0.01, 0.02, 0.03]
    );
    assert_eq!(
        pipeline.params.film_render.halation.halation_first_sigma_um,
        [2., 4., 6.]
    );
    let updated = pipeline.with_params(params);
    assert_eq!(
        updated.params.film_render.dir_couplers.gamma_samelayer_rgb,
        [0.1, 0.2, 0.3]
    );
}
