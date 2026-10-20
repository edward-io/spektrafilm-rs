//! A portable edit recipe, stored in EXIF UserComment or an OpenEXR attribute.
use crate::{Metadata, RawOptions, SaveOptions, crop::Crop};
use anyhow::Result;
use little_exif::exif_tag::ExifTag;
use serde::Serialize;
use serde_json::{Value, json};
use std::fmt::Write;

/// Editor values, before crop-dependent grain scaling is applied to the pipeline.
#[derive(Serialize)]
pub struct EditSettings<'a, P: Serialize> {
    pub film: &'a str,
    pub paper: &'a str,
    pub params: &'a P,
    pub raw: &'a RawOptions,
    pub rotation: u8,
    pub crop: Crop,
}

impl Metadata {
    pub fn with_edit_settings(&self, settings: &impl Serialize) -> Result<Self> {
        let mut metadata = self.clone();
        metadata.edit_settings = Some(serde_json::to_value(settings)?);
        Ok(metadata)
    }
}

pub(crate) fn recipe(
    metadata: &Metadata,
    options: &SaveOptions,
    format: &str,
    dimensions: [u32; 2],
) -> Result<Option<String>> {
    let Some(settings) = &metadata.edit_settings else {
        return Ok(None);
    };
    let mut document = json!({
        "application": "spektrafilm",
        "application_version": env!("CARGO_PKG_VERSION"),
        "schema_version": 1,
        "settings": settings,
        "output": options,
        "format": format,
        "dimensions": dimensions,
    });
    if let Some(comment) = metadata.native.as_ref().and_then(|tags| {
        tags.into_iter().find_map(|tag| match tag {
            ExifTag::UserComment(value) => Some(value),
            _ => None,
        })
    }) {
        let previous = comment
            .strip_prefix(b"ASCII\0\0\0")
            .and_then(|bytes| serde_json::from_slice::<Value>(bytes).ok())
            .filter(|value| value["application"] == "spektrafilm" && value["schema_version"] == 1);
        // Re-exporting replaces our recipe without nesting earlier recipes.
        let original = match previous {
            Some(previous) => previous.get("original_user_comment").cloned(),
            None => Some(json!(comment)),
        };
        if let Some(original) = original {
            document["original_user_comment"] = original;
        }
    }
    // EXIF's ASCII identifier and OpenEXR text both need single-byte characters.
    let json = serde_json::to_string(&document)?;
    let mut ascii = String::with_capacity(json.len());
    for character in json.chars() {
        if character.is_ascii() {
            ascii.push(character);
        } else {
            for unit in character.encode_utf16(&mut [0; 2]) {
                write!(&mut ascii, "\\u{unit:04x}")?;
            }
        }
    }
    Ok(Some(ascii))
}
