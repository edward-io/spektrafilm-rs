/// Runtime parameters for the film simulation pipeline.
///
/// Mirrors Python `params_schema.py`. Every field has a sensible default
/// matching the Python implementation.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionFilterParams {
    #[serde(default)]
    pub active: bool,
    #[serde(default = "default_bpm")]
    pub filter_family: String,
    #[serde(default = "default_half")]
    pub strength: f32,
    #[serde(default = "default_one")]
    pub spatial_scale: f32,
    #[serde(default)]
    pub halo_warmth: f32,
    #[serde(default = "default_one")]
    pub core_intensity: f32,
    #[serde(default = "default_one")]
    pub core_size: f32,
    #[serde(default = "default_one")]
    pub halo_intensity: f32,
    #[serde(default = "default_one")]
    pub halo_size: f32,
    #[serde(default = "default_one")]
    pub bloom_intensity: f32,
    #[serde(default = "default_one")]
    pub bloom_size: f32,
}

impl DiffusionFilterParams {
    /// Borrowed view as the model crate's `DiffusionFilter` (f64), for the
    /// CPU diffusion-filter apply. `family` borrows `self.filter_family`.
    pub fn to_model(&self) -> spektrafilm_model::diffusion::DiffusionFilter<'_> {
        spektrafilm_model::diffusion::DiffusionFilter {
            family: &self.filter_family,
            strength: self.strength as f64,
            spatial_scale: self.spatial_scale as f64,
            halo_warmth: self.halo_warmth as f64,
            core_intensity: self.core_intensity as f64,
            core_size: self.core_size as f64,
            halo_intensity: self.halo_intensity as f64,
            halo_size: self.halo_size as f64,
            bloom_intensity: self.bloom_intensity as f64,
            bloom_size: self.bloom_size as f64,
        }
    }
}

impl Default for DiffusionFilterParams {
    fn default() -> Self {
        Self {
            active: false,
            filter_family: "black_pro_mist".into(),
            strength: 0.5,
            spatial_scale: 1.0,
            halo_warmth: 0.0,
            core_intensity: 1.0,
            core_size: 1.0,
            halo_intensity: 1.0,
            halo_size: 1.0,
            bloom_intensity: 1.0,
            bloom_size: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraParams {
    #[serde(default = "default_none_filter")]
    pub color_filter: String,
    #[serde(default)]
    pub exposure_compensation_ev: f32,
    #[serde(default = "default_true")]
    pub auto_exposure: bool,
    #[serde(default = "default_center_weighted")]
    pub auto_exposure_method: String,
    #[serde(default)]
    pub lens_blur_um: f32,
    #[serde(default = "default_35")]
    pub film_format_mm: f32,
    #[serde(default = "default_filter_uv")]
    pub filter_uv: [f32; 3],
    #[serde(default = "default_filter_ir")]
    pub filter_ir: [f32; 3],
    #[serde(default)]
    pub diffusion_filter: DiffusionFilterParams,
}

impl Default for CameraParams {
    fn default() -> Self {
        Self {
            color_filter: default_none_filter(),
            exposure_compensation_ev: 0.0,
            auto_exposure: true,
            auto_exposure_method: "center_weighted".into(),
            lens_blur_um: 0.0,
            film_format_mm: 35.0,
            filter_uv: [0.0, 410.0, 8.0],
            filter_ir: [0.0, 675.0, 15.0],
            diffusion_filter: DiffusionFilterParams::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnlargerParams {
    #[serde(default = "default_th_kg3")]
    pub illuminant: String,
    #[serde(default = "default_one")]
    pub print_exposure: f32,
    #[serde(default = "default_true")]
    pub print_exposure_compensation: bool,
    #[serde(default = "default_true")]
    pub normalize_print_exposure: bool,
    #[serde(default)]
    pub y_filter_shift: f32,
    #[serde(default)]
    pub m_filter_shift: f32,
    #[serde(default = "default_55")]
    pub y_filter_neutral: f32,
    #[serde(default = "default_65")]
    pub m_filter_neutral: f32,
    #[serde(default)]
    pub c_filter_neutral: f32,
    #[serde(default)]
    pub lens_blur: f32,
    #[serde(default)]
    pub diffusion_filter: DiffusionFilterParams,
    #[serde(default)]
    pub preflash_exposure: f32,
    #[serde(default)]
    pub preflash_y_filter_shift: f32,
    #[serde(default)]
    pub preflash_m_filter_shift: f32,
}

impl Default for EnlargerParams {
    fn default() -> Self {
        Self {
            illuminant: "TH-KG3".into(),
            print_exposure: 1.0,
            print_exposure_compensation: true,
            normalize_print_exposure: true,
            y_filter_shift: 0.0,
            m_filter_shift: 0.0,
            y_filter_neutral: 55.0,
            m_filter_neutral: 65.0,
            c_filter_neutral: 0.0,
            lens_blur: 0.0,
            diffusion_filter: DiffusionFilterParams::default(),
            preflash_exposure: 0.0,
            preflash_y_filter_shift: 0.0,
            preflash_m_filter_shift: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScannerParams {
    #[serde(default)]
    pub lens_blur: f32,
    #[serde(default)]
    pub white_correction: bool,
    #[serde(default)]
    pub black_correction: bool,
    #[serde(default = "default_098")]
    pub white_level: f32,
    #[serde(default = "default_001")]
    pub black_level: f32,
    #[serde(default = "default_unsharp")]
    pub unsharp_mask: [f32; 2],
}

impl Default for ScannerParams {
    fn default() -> Self {
        Self {
            lens_blur: 0.0,
            white_correction: false,
            black_correction: false,
            white_level: 0.98,
            black_level: 0.01,
            unsharp_mask: [0.7, 0.7],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrainParams {
    #[serde(default = "default_rms", deserialize_with = "deserialize_channels")]
    pub rms_granularity: [f64; 3],
    #[serde(
        default = "default_rms_scales",
        deserialize_with = "deserialize_channels"
    )]
    pub particle_scale_sublayers: [f64; 3],
    #[serde(default = "default_usm_sigma")]
    pub mult_usm_sigma: f32,
    #[serde(default = "default_usm_amount")]
    pub mult_usm_amount: f32,
    #[serde(default = "default_1i")]
    pub micro_sublayers: u32,
    #[serde(default = "default_true")]
    pub active: bool,
    #[serde(default = "default_true")]
    pub sublayers_active: bool,
    // f64 to preserve Python JSON precision through the Poisson/Binomial
    // RNG pipeline — the f32 truncation of these values shifts the
    // Poisson lambda by ~5e-8 and produces a different RNG stream.
    #[serde(default = "default_02_f64", alias = "particle_area_um2")]
    pub agx_particle_area_um2: f64,
    #[serde(default = "default_particle_scale_f64", alias = "particle_scale")]
    pub agx_particle_scale: [f64; 3],
    #[serde(
        default = "default_particle_scale_layers_f64",
        alias = "particle_scale_layers"
    )]
    pub agx_particle_scale_layers: [f64; 3],
    #[serde(
        default = "default_density_min_f64",
        deserialize_with = "deserialize_channels"
    )]
    pub density_min: [f64; 3],
    #[serde(
        default = "default_uniformity_f64",
        deserialize_with = "deserialize_channels"
    )]
    pub uniformity: [f64; 3],
    #[serde(default = "default_065")]
    pub blur: f32,
    #[serde(default = "default_cloud_blur")]
    pub blur_dye_clouds_um: f32,
    #[serde(default = "default_micro_structure")]
    pub micro_structure: [f32; 2],
    #[serde(default = "default_1i")]
    pub n_sub_layers: u32,
    /// One shared noise field across all channels instead of independent
    /// per-channel RNG streams. Set by the pipeline for B&W films
    /// (upstream n_channels==1 has a single emulsion); not user-facing.
    #[serde(default)]
    pub monochrome: bool,
}

fn default_02_f64() -> f64 {
    0.2
}
fn default_particle_scale_f64() -> [f64; 3] {
    [1.6, 1.6, 3.2]
}
fn default_particle_scale_layers_f64() -> [f64; 3] {
    [2.0, 1.0, 0.5]
}
fn default_density_min_f64() -> [f64; 3] {
    [0.03, 0.03, 0.03]
}
fn default_uniformity_f64() -> [f64; 3] {
    [0.97; 3]
}

impl Default for GrainParams {
    fn default() -> Self {
        Self {
            rms_granularity: default_rms(),
            particle_scale_sublayers: default_rms_scales(),
            mult_usm_sigma: default_usm_sigma(),
            mult_usm_amount: default_usm_amount(),
            micro_sublayers: 1,
            active: true,
            sublayers_active: true,
            agx_particle_area_um2: 0.2,
            agx_particle_scale: [1.6, 1.6, 3.2],
            agx_particle_scale_layers: [2.0, 1.0, 0.5],
            density_min: [0.03, 0.03, 0.03],
            uniformity: [0.97; 3],
            blur: 0.89,
            blur_dye_clouds_um: 2.0,
            micro_structure: [0.2, 30.0],
            n_sub_layers: 1,
            monochrome: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HalationParams {
    #[serde(default = "default_true")]
    pub active: bool,
    // f64 to match Python's `np.asarray(..., dtype=np.float64)` —
    // f32 storage truncates ~7 decimals which shifts every sigma/lambda
    // by ~3e-8, accumulating through Gaussian/exponential kernels.
    #[serde(default = "default_one_f64")]
    pub scatter_amount: f64,
    #[serde(default = "default_one_f64")]
    pub scatter_spatial_scale: f64,
    #[serde(default = "default_one_f64")]
    pub halation_amount: f64,
    #[serde(default = "default_one_f64")]
    pub halation_spatial_scale: f64,
    #[serde(default = "default_scatter_core_f64")]
    pub scatter_core_um: [f64; 3],
    #[serde(default = "default_scatter_tail_f64")]
    pub scatter_tail_um: [f64; 3],
    #[serde(default = "default_scatter_tail_weight_f64")]
    pub scatter_tail_weight: [f64; 3],
    #[serde(default)]
    pub boost_ev: f32,
    #[serde(default = "default_03")]
    pub boost_range: f32,
    #[serde(default = "default_4")]
    pub protect_ev: f32,
    #[serde(default = "default_halation_strength_f64")]
    pub halation_strength: [f64; 3],
    #[serde(default = "default_halation_sigma_f64")]
    pub halation_first_sigma_um: [f64; 3],
    #[serde(default = "default_3i")]
    pub halation_n_bounces: u32,
    #[serde(default = "default_half_f64")]
    pub halation_bounce_decay: f64,
    #[serde(default = "default_true")]
    pub halation_renormalize: bool,
}

fn default_one_f64() -> f64 {
    1.0
}
fn default_half_f64() -> f64 {
    0.5
}
fn default_scatter_core_f64() -> [f64; 3] {
    [2.2, 2.0, 1.6]
}
fn default_scatter_tail_f64() -> [f64; 3] {
    [9.3, 9.7, 9.1]
}
fn default_scatter_tail_weight_f64() -> [f64; 3] {
    [0.78, 0.65, 0.67]
}
fn default_halation_strength_f64() -> [f64; 3] {
    [0.05, 0.015, 0.0]
}
fn default_halation_sigma_f64() -> [f64; 3] {
    [65.0, 65.0, 65.0]
}

impl Default for HalationParams {
    fn default() -> Self {
        Self {
            active: true,
            scatter_amount: 1.0,
            scatter_spatial_scale: 1.0,
            halation_amount: 1.0,
            halation_spatial_scale: 1.0,
            scatter_core_um: [2.2, 2.0, 1.6],
            scatter_tail_um: [9.3, 9.7, 9.1],
            scatter_tail_weight: [0.78, 0.65, 0.67],
            boost_ev: 0.0,
            boost_range: 0.3,
            protect_ev: 4.0,
            halation_strength: [0.05, 0.015, 0.0],
            halation_first_sigma_um: [65.0, 65.0, 65.0],
            halation_n_bounces: 3,
            halation_bounce_decay: 0.5,
            halation_renormalize: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirCouplersParams {
    #[serde(default = "default_langmuir")]
    pub langmuir_donor_k_rgb: [f64; 3],
    #[serde(default = "default_langmuir")]
    pub langmuir_receiver_k_rgb: [f64; 3],
    #[serde(default = "default_true")]
    pub active: bool,
    // f64 throughout — Python reads these as JSON floats (f64). The
    // f32 truncation of values like 0.341 (→ 0.3409999907... in f32 vs
    // 0.341 = 0.34100000000000003 in f64) shifts every coupler weight
    // and diffusion sigma by ~3e-8 and amplifies through the per-channel
    // density correction.
    #[serde(default = "default_one_f64")]
    pub amount: f64,
    #[serde(default = "default_one_f64")]
    pub inhibition_samelayer: f64,
    #[serde(default = "default_one_f64")]
    pub inhibition_interlayer: f64,
    #[serde(default = "default_gamma_same_f64")]
    pub gamma_samelayer_rgb: [f64; 3],
    #[serde(default = "default_gamma_r_gb_f64")]
    pub gamma_interlayer_r_to_gb: [f64; 2],
    #[serde(default = "default_gamma_g_rb_f64")]
    pub gamma_interlayer_g_to_rb: [f64; 2],
    #[serde(default = "default_gamma_b_rg_f64")]
    pub gamma_interlayer_b_to_rg: [f64; 2],
    #[serde(default = "default_20_f64")]
    pub diffusion_size_um: f64,
    #[serde(default = "default_200_f64")]
    pub diffusion_tail_um: f64,
    #[serde(default = "default_006_f64")]
    pub diffusion_tail_weight: f64,
}

fn default_gamma_same_f64() -> [f64; 3] {
    [0.341, 0.324, 0.273]
}
fn default_gamma_r_gb_f64() -> [f64; 2] {
    [0.355, 0.305]
}
fn default_gamma_g_rb_f64() -> [f64; 2] {
    [0.154, 0.358]
}
fn default_gamma_b_rg_f64() -> [f64; 2] {
    [0.171, 0.225]
}
fn default_20_f64() -> f64 {
    20.0
}
fn default_200_f64() -> f64 {
    200.0
}
fn default_006_f64() -> f64 {
    0.03
}

impl Default for DirCouplersParams {
    fn default() -> Self {
        Self {
            langmuir_donor_k_rgb: default_langmuir(),
            langmuir_receiver_k_rgb: default_langmuir(),
            active: true,
            amount: 1.0,
            inhibition_samelayer: 1.0,
            inhibition_interlayer: 1.0,
            gamma_samelayer_rgb: [0.341, 0.324, 0.273],
            gamma_interlayer_r_to_gb: [0.355, 0.305],
            gamma_interlayer_g_to_rb: [0.154, 0.358],
            gamma_interlayer_b_to_rg: [0.171, 0.225],
            diffusion_size_um: 20.0,
            diffusion_tail_um: 200.0,
            diffusion_tail_weight: 0.03,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlareParams {
    #[serde(default = "default_true")]
    pub active: bool,
    #[serde(default = "default_003")]
    pub percent: f32,
    #[serde(default = "default_07")]
    pub roughness: f32,
    #[serde(default = "default_half")]
    pub blur: f32,
}

impl Default for GlareParams {
    fn default() -> Self {
        Self {
            active: true,
            percent: 0.03,
            roughness: 0.7,
            blur: 0.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilmRenderingParams {
    #[serde(default)]
    pub chemistry: PrintCurvesMorphParams,
    #[serde(default)]
    pub base: BaseDensityParams,
    #[serde(default)]
    pub convert: ConvertFilmParams,
    #[serde(default = "default_channel_order")]
    pub channel_order: [usize; 3],
    #[serde(default = "default_one")]
    pub density_curve_gamma: f32,
    /// Development time (minutes) for B&W development-time families —
    /// forward-port of upstream dev's `chemistry.development_time`. Selects
    /// the nearest entry of the profile's family; `None` picks the
    /// floor-middle entry. Ignored by colour profiles.
    #[serde(default)]
    pub development_time: Option<f64>,
    #[serde(default)]
    pub grain: GrainParams,
    #[serde(default)]
    pub halation: HalationParams,
    #[serde(default)]
    pub dir_couplers: DirCouplersParams,
    #[serde(default)]
    pub glare: GlareParams,
}

impl Default for FilmRenderingParams {
    fn default() -> Self {
        Self {
            chemistry: PrintCurvesMorphParams::default(),
            base: BaseDensityParams::default(),
            convert: ConvertFilmParams::default(),
            channel_order: [0, 1, 2],
            density_curve_gamma: 1.0,
            development_time: None,
            grain: GrainParams::default(),
            halation: HalationParams::default(),
            dir_couplers: DirCouplersParams::default(),
            glare: GlareParams::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrintRenderingParams {
    #[serde(default)]
    pub base: BaseDensityParams,
    #[serde(default = "default_channel_order")]
    pub channel_order: [usize; 3],
    #[serde(default = "default_one")]
    pub density_curve_gamma: f32,
    /// Development time (minutes) for B&W print stocks (e.g. kodak_2302) —
    /// same semantics as `FilmRenderingParams::development_time`.
    #[serde(default)]
    pub development_time: Option<f64>,
    #[serde(default)]
    pub glare: GlareParams,
    #[serde(default, alias = "density_curves_morph")]
    pub chemistry: PrintCurvesMorphParams,
}

impl Default for PrintRenderingParams {
    fn default() -> Self {
        Self {
            base: BaseDensityParams::default(),
            channel_order: [0, 1, 2],
            density_curve_gamma: 1.0,
            development_time: None,
            glare: GlareParams::default(),
            chemistry: PrintCurvesMorphParams::default(),
        }
    }
}

/// Shared film/print chemistry, evaluated in f64 against the fitted density model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrintCurvesMorphParams {
    #[serde(default)]
    pub development_time: Option<f64>,
    #[serde(default = "default_true")]
    pub active: bool,
    #[serde(default = "default_one_f64")]
    pub gamma_factor: f64,
    #[serde(default = "default_one_f64")]
    pub gamma_factor_fast: f64,
    #[serde(default = "default_one_f64")]
    pub gamma_factor_slow: f64,
    #[serde(default = "default_one_f64")]
    pub gamma_factor_red: f64,
    #[serde(default = "default_one_f64")]
    pub gamma_factor_green: f64,
    #[serde(default = "default_one_f64")]
    pub gamma_factor_blue: f64,
    #[serde(default)]
    pub developer_exhaustion: f64,
}

impl Default for PrintCurvesMorphParams {
    fn default() -> Self {
        Self {
            development_time: None,
            active: true,
            gamma_factor: 1.0,
            gamma_factor_fast: 1.0,
            gamma_factor_slow: 1.0,
            gamma_factor_red: 1.0,
            gamma_factor_green: 1.0,
            gamma_factor_blue: 1.0,
            developer_exhaustion: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoParams {
    #[serde(default = "default_prophoto")]
    pub input_color_space: String,
    #[serde(default)]
    pub input_cctf_decoding: bool,
    #[serde(default = "default_srgb")]
    pub output_color_space: String,
    #[serde(default = "default_true")]
    pub output_cctf_encoding: bool,
    #[serde(default)]
    pub crop: bool,
    #[serde(default = "default_crop_center")]
    pub crop_center: [f32; 2],
    #[serde(default = "default_crop_size")]
    pub crop_size: [f32; 2],
    #[serde(default = "default_one")]
    pub upscale_factor: f32,
    #[serde(default)]
    pub scan_film: bool,
    #[serde(default)]
    pub output_gamut_compress: OutputGamutCompressParams,
    #[serde(default)]
    pub input_gamut_compress: InputGamutCompressParams,
}

impl Default for IoParams {
    fn default() -> Self {
        Self {
            input_color_space: "ProPhoto RGB".into(),
            input_cctf_decoding: false,
            output_color_space: "sRGB".into(),
            output_cctf_encoding: true,
            crop: false,
            crop_center: [0.5, 0.5],
            crop_size: [0.1, 0.1],
            upscale_factor: 1.0,
            scan_film: false,
            output_gamut_compress: OutputGamutCompressParams::default(),
            input_gamut_compress: InputGamutCompressParams::default(),
        }
    }
}

/// Input gamut compression config — mirrors upstream `InputGamutCompressSpec`.
/// Baked into the tc_lut at build time, so the per-pixel path stays
/// compression-agnostic. `"xy"` (the default, matching upstream 0.3.4) applies
/// the ACES-RGC-style radial compression toward the spectral locus;
/// `algorithm = "off"` passes input chromaticities through unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputGamutCompressParams {
    #[serde(default = "default_true")]
    pub active: bool,
    #[serde(default = "default_xy")]
    pub algorithm: String,
    #[serde(default = "default_input_knee")]
    pub knee: [f32; 3],
    #[serde(default = "default_input_boundary")]
    pub boundary: String,
    #[serde(default = "default_hull_detail")]
    pub hull_detail: f32,
}

impl Default for InputGamutCompressParams {
    fn default() -> Self {
        Self {
            active: true,
            algorithm: "xy".into(),
            knee: default_input_knee(),
            boundary: default_input_boundary(),
            hull_detail: default_hull_detail(),
        }
    }
}

/// Output gamut compression config — mirrors upstream `OutputGamutCompressSpec`.
/// `"oklch"` is the experimental default; `"cam16ucs"` applies CAM16-UCS
/// chroma knee + one-sided lightness roll-off; `"off"` passes RGB through
/// unchanged.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputGamutCompressParams {
    #[serde(default = "default_oklch")]
    pub algorithm: String,
    #[serde(default = "default_gamut_knee")]
    pub knee: [f32; 3],
    #[serde(default = "default_gamut_lightness")]
    pub lightness_compression: Option<[f32; 3]>,
}

impl Default for OutputGamutCompressParams {
    fn default() -> Self {
        Self {
            algorithm: "oklch".into(),
            knee: default_gamut_knee(),
            lightness_compression: default_gamut_lightness(),
        }
    }
}

fn default_xy() -> String {
    "xy".into()
}
fn default_oklch() -> String {
    "oklch".into()
}
fn default_gamut_knee() -> [f32; 3] {
    [0.95, 1.0, 1.6]
}
fn default_gamut_lightness() -> Option<[f32; 3]> {
    Some([0.95, 1.0, 1.6])
}

fn default_input_knee() -> [f32; 3] {
    [0.815, 1., 1.2]
}
fn default_input_boundary() -> String {
    "inscribed_hull".into()
}
fn default_hull_detail() -> f32 {
    5.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SettingsParams {
    #[serde(default)]
    pub use_convert_lut: bool,
    #[serde(default = "default_33")]
    pub convert_lut_resolution: u32,
    /// Apply stock presets when digesting runtime params. Native GUI applies
    /// them on selection, then disables this so subsequent edits survive.
    #[serde(default = "default_true")]
    pub apply_stock_specifics: bool,
    #[serde(default = "default_hanatos")]
    pub rgb_to_raw_method: String,
    #[serde(default = "default_true")]
    pub apply_hanatos2025_adaptation_window: bool,
    #[serde(default)]
    pub apply_hanatos2025_adaptation_surface: bool,
    #[serde(default)]
    pub spectral_gaussian_blur: f32,
    #[serde(default)]
    pub use_enlarger_lut: bool,
    #[serde(default)]
    pub use_scanner_lut: bool,
    #[serde(default = "default_17")]
    pub lut_resolution: u32,
    #[serde(default = "default_true")]
    pub use_fast_stats: bool,
    #[serde(default = "default_640")]
    pub preview_max_size: u32,
    #[serde(default)]
    pub preview_mode: bool,
    #[serde(default = "default_true")]
    pub neutral_print_filters_from_database: bool,
    /// Use CAT16 (instead of CAT02) for the input RGB→tc chromatic adaptation
    /// that feeds the Hanatos spectral upsampling. Default true matches
    /// upstream 0.3.4 (`_rgb_to_tc_b` hard-codes CAT16); false restores the
    /// CAT02 0.3.2 behavior.
    #[serde(default = "default_true")]
    pub use_cat16: bool,
}

impl Default for SettingsParams {
    fn default() -> Self {
        Self {
            use_convert_lut: false,
            convert_lut_resolution: 33,
            apply_stock_specifics: true,
            rgb_to_raw_method: "hanatos2025".into(),
            apply_hanatos2025_adaptation_window: true,
            apply_hanatos2025_adaptation_surface: false,
            spectral_gaussian_blur: 0.0,
            use_enlarger_lut: false,
            use_scanner_lut: false,
            lut_resolution: 17,
            use_fast_stats: true,
            preview_max_size: 640,
            preview_mode: false,
            neutral_print_filters_from_database: true,
            use_cat16: true,
        }
    }
}

/// Top-level runtime parameters. Combines all sub-parameter groups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeParams {
    #[serde(default)]
    pub workflow: WorkflowParams,
    #[serde(default)]
    pub debug: DebugParams,
    #[serde(default)]
    pub taps: TapsParams,
    #[serde(default)]
    pub camera: CameraParams,
    #[serde(default)]
    pub enlarger: EnlargerParams,
    #[serde(default)]
    pub scanner: ScannerParams,
    #[serde(default)]
    pub film_render: FilmRenderingParams,
    #[serde(default)]
    pub print_render: PrintRenderingParams,
    #[serde(default)]
    pub io: IoParams,
    #[serde(default)]
    pub settings: SettingsParams,
}

impl Default for RuntimeParams {
    fn default() -> Self {
        Self {
            workflow: WorkflowParams::default(),
            debug: DebugParams::default(),
            taps: TapsParams::default(),
            camera: CameraParams::default(),
            enlarger: EnlargerParams::default(),
            scanner: ScannerParams::default(),
            film_render: FilmRenderingParams::default(),
            print_render: PrintRenderingParams::default(),
            io: IoParams::default(),
            settings: SettingsParams::default(),
        }
    }
}

// Default value helpers
fn default_bpm() -> String {
    "black_pro_mist".into()
}
fn default_half() -> f32 {
    0.5
}
fn default_one() -> f32 {
    1.0
}
fn default_true() -> bool {
    true
}
fn default_center_weighted() -> String {
    "center_weighted".into()
}
fn default_35() -> f32 {
    35.0
}
fn default_filter_uv() -> [f32; 3] {
    [0.0, 410.0, 8.0]
}
fn default_filter_ir() -> [f32; 3] {
    [0.0, 675.0, 15.0]
}
fn default_th_kg3() -> String {
    "TH-KG3".into()
}
fn default_55() -> f32 {
    55.0
}
fn default_65() -> f32 {
    65.0
}
fn default_098() -> f32 {
    0.98
}
fn default_001() -> f32 {
    0.01
}
fn default_unsharp() -> [f32; 2] {
    [0.7, 0.7]
}

fn default_065() -> f32 {
    0.89
}

fn default_none_filter() -> String {
    "none".into()
}
fn default_rms() -> [f64; 3] {
    [6., 8., 10.]
}
fn default_rms_scales() -> [f64; 3] {
    [1., 0.5, 0.25]
}
fn default_usm_sigma() -> f32 {
    0.7
}
fn default_usm_amount() -> f32 {
    1.5
}
fn default_cloud_blur() -> f32 {
    2.
}
fn default_langmuir() -> [f64; 3] {
    [1.; 3]
}
fn default_33() -> u32 {
    33
}

fn deserialize_channels<'de, D: serde::Deserializer<'de>>(d: D) -> Result<[f64; 3], D::Error> {
    let values = Vec::<Option<f64>>::deserialize(d)?;
    match values.as_slice() {
        [Some(v)] => Ok([*v; 3]),
        [Some(a), b, c] => Ok([*a, b.unwrap_or(*a), c.unwrap_or(*a)]),
        _ => Err(serde::de::Error::custom(
            "expected one or three channels with a finite first channel",
        )),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BaseDensityParams {
    pub active: bool,
    pub scale: f64,
    pub tilt: f64,
    pub cyan: f64,
    pub magenta: f64,
    pub yellow: f64,
}
impl Default for BaseDensityParams {
    fn default() -> Self {
        Self {
            active: true,
            scale: 1.,
            tilt: 0.,
            cyan: 1.,
            magenta: 1.,
            yellow: 1.,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ConvertFilmParams {
    pub scan_illuminant: String,
    pub exposure_compensation_ev: f64,
    pub base_percentile: f64,
    pub calibration: String,
}
impl Default for ConvertFilmParams {
    fn default() -> Self {
        Self {
            scan_illuminant: "D55".into(),
            exposure_compensation_ev: 0.,
            base_percentile: 99.,
            calibration: "1 0 0  0 1 0  0 0 1".into(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkflowParams {
    pub route: String,
}
impl RuntimeParams {
    pub fn route(&self) -> &str {
        if !self.workflow.route.is_empty() {
            &self.workflow.route
        } else if self.io.scan_film {
            "input > film > scan"
        } else {
            "input > film > print > scan"
        }
    }
    pub fn scans_film(&self) -> bool {
        matches!(
            self.route(),
            "input > film > scan"
                | "input > convert-film > scan"
                | "input > convert-film > scan-minus-base"
        )
    }
    pub fn converts_film(&self) -> bool {
        self.route().contains("convert-film")
    }
}
fn default_micro_structure() -> [f32; 2] {
    [0.2, 30.0]
}
fn default_1i() -> u32 {
    1
}

fn default_03() -> f32 {
    0.3
}
fn default_4() -> f32 {
    4.0
}

fn default_3i() -> u32 {
    3
}

fn default_003() -> f32 {
    0.03
}
fn default_07() -> f32 {
    0.7
}
fn default_prophoto() -> String {
    "ProPhoto RGB".into()
}
fn default_srgb() -> String {
    "sRGB".into()
}
fn default_crop_center() -> [f32; 2] {
    [0.5, 0.5]
}
fn default_crop_size() -> [f32; 2] {
    [0.1, 0.1]
}
fn default_hanatos() -> String {
    "hanatos2025".into()
}
fn default_17() -> u32 {
    17
}
fn default_640() -> u32 {
    640
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DebugParams {
    #[serde(default)]
    pub deactivate_spatial_effects: bool,
    #[serde(default)]
    pub deactivate_stochastic_effects: bool,
    #[serde(default)]
    pub print_timings: bool,
    #[serde(default)]
    pub lut_mode: bool,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TapsParams {
    #[serde(default)]
    pub inject: Option<String>,
    #[serde(default)]
    pub collect: Option<String>,
}

fn default_channel_order() -> [usize; 3] {
    [0, 1, 2]
}
