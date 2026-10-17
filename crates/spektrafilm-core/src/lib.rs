pub mod bundle_output;
pub mod calibration;
pub mod color_reference;
pub mod data;
pub mod enlarger;
pub mod gamut_compression;
pub mod input_gamut;
pub mod measure;
pub mod neutral_filters;
mod param_digest;
pub mod params;
pub mod pipeline;
pub mod print_morph;
pub mod profile;
pub mod resize;
pub mod spectral_service;
pub mod stages;
pub mod stock;

pub mod lut_bundle;
pub mod lut_export;
pub mod lut_quality;
mod lut_spaces;

mod ocio;

mod qa_gamut;
pub mod qa_metrics;
mod qa_noise;
pub mod qa_patterns;
mod qa_rng;
pub mod qa_style;
mod qa_visual;

pub mod qa_report;

pub mod conversion;

#[cfg(test)]
pub(crate) fn experimental_reference() -> serde_json::Value {
    serde_json::from_str(include_str!("../tests/fixtures/experimental.json"))
        .expect("Python experimental fixture")
}
