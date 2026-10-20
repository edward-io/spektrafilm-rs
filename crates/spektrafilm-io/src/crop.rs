//! A free crop with straightening, in the coordinates of the displayed frame
//! (after the file's orientation and any quarter turns).
use crate::lightroom::Geometry;
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use spektrafilm_math::{image::ImageBuf, precision::from_f64};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Crop {
    /// Centre of the crop, as fractions of the frame's width and height.
    pub center: [f64; 2],
    /// Width and height of the crop, as fractions of the frame's width and height.
    pub size: [f64; 2],
    /// Clockwise rotation of the image under the crop, in degrees.
    pub angle: f64,
}

/// Largest straightening angle, as in Lightroom.
pub const MAX_ANGLE: f64 = 45.;

impl Default for Crop {
    fn default() -> Self {
        Self::FULL
    }
}

impl Crop {
    pub const FULL: Self = Self {
        center: [0.5, 0.5],
        size: [1., 1.],
        angle: 0.,
    };

    pub fn is_full(&self) -> bool {
        *self == Self::FULL
    }

    pub fn validate(&self) -> Result<()> {
        // A crop that fits the frame spans at most sqrt(2) of either axis at ±45°.
        ensure!(
            self.center.iter().all(|v| (0.0..=1.0).contains(v))
                && self
                    .size
                    .iter()
                    .all(|v| *v > 0. && *v <= std::f64::consts::SQRT_2 + 1e-6),
            "invalid crop"
        );
        ensure!(
            self.angle.is_finite() && self.angle.abs() <= MAX_ANGLE,
            "invalid crop angle"
        );
        Ok(())
    }

    /// Crop width and height in pixels of a `width` × `height` frame.
    pub fn pixels(&self, width: f64, height: f64) -> [f64; 2] {
        [self.size[0] * width, self.size[1] * height]
    }

    /// How much of the frame's long side the crop's long side covers; keeps
    /// grain and other spatial effects the same size when cropping.
    pub fn long_side_fraction(&self, width: f64, height: f64) -> f64 {
        let [w, h] = self.pixels(width, height);
        w.max(h) / width.max(height)
    }

    /// Map an offset from the crop centre (crop pixels, crop axes) to a frame
    /// position in pixels.
    pub fn to_frame(&self, width: f64, height: f64, offset: [f64; 2]) -> [f64; 2] {
        let (sin, cos) = self.angle.to_radians().sin_cos();
        let [u, v] = offset;
        [
            self.center[0] * width + u * cos + v * sin,
            self.center[1] * height - u * sin + v * cos,
        ]
    }

    /// The same crop after rotating the frame clockwise by quarter turns.
    pub fn rotated_quarter_turns(self, turns: u8) -> Self {
        let [x, y] = self.center;
        let [w, h] = self.size;
        let (center, size) = match turns % 4 {
            1 => ([1. - y, x], [h, w]),
            2 => ([1. - x, 1. - y], [w, h]),
            3 => ([y, 1. - x], [h, w]),
            _ => return self,
        };
        Self {
            center,
            size,
            ..self
        }
    }

    /// Resample the cropped, straightened region (bilinear, edges clamped).
    pub fn apply(&self, image: &ImageBuf) -> Result<ImageBuf> {
        self.validate()?;
        ensure!(
            image.width > 0 && image.height > 0,
            "cannot crop an empty image"
        );
        if self.is_full() {
            return Ok(image.clone());
        }
        let (fw, fh) = (image.width as f64, image.height as f64);
        let [cw, ch] = self.pixels(fw, fh);
        let (width, height) = (cw.round().max(1.), ch.round().max(1.));
        ensure!(
            width <= u32::MAX as f64 && height <= u32::MAX as f64,
            "crop dimensions are too large"
        );
        let (width, height) = (width as u32, height as u32);
        let samples = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(3))
            .context("crop dimensions are too large")?;
        let mut output = ImageBuf {
            width,
            height,
            data: Vec::new(),
        };
        output
            .data
            .try_reserve_exact(samples)
            .context("allocating crop")?;
        output.data.resize(samples, from_f64(0.));
        output
            .data
            .par_chunks_mut(width as usize * 3)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, pixel) in row.chunks_exact_mut(3).enumerate() {
                    let offset = [
                        ((x as f64 + 0.5) / width as f64 - 0.5) * cw,
                        ((y as f64 + 0.5) / height as f64 - 0.5) * ch,
                    ];
                    let [px, py] = self.to_frame(fw, fh, offset);
                    let x = (px - 0.5).clamp(0., fw - 1.);
                    let y = (py - 0.5).clamp(0., fh - 1.);
                    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
                    let (x1, y1) = (
                        (x0 + 1).min(image.width - 1),
                        (y0 + 1).min(image.height - 1),
                    );
                    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
                    for c in 0..3 {
                        let at = |x: u32, y: u32| image.data[image.idx(x, y, c)] as f64;
                        pixel[c] = from_f64(
                            (at(x0, y0) * (1. - fx) + at(x1, y0) * fx) * (1. - fy)
                                + (at(x0, y1) * (1. - fx) + at(x1, y1) * fx) * fy,
                        );
                    }
                }
            });
        Ok(output)
    }

    /// Lightroom's crop of a catalog photo, for the frame that its orientation
    /// alone produces (`width` × `height`, see [`Geometry::orientation_only`]).
    pub fn from_lightroom(geometry: &Geometry, width: f64, height: f64) -> Self {
        let transposed = geometry.orientation >= 5;
        // Lightroom's develop coordinates, before its orientation.
        let (w, h) = if transposed {
            (height, width)
        } else {
            (width, height)
        };
        let [left, top, right, bottom] = geometry.crop;
        let centre = [(left + right) * w / 2., (top + bottom) * h / 2.];
        let (sin, cos) = geometry.angle.to_radians().sin_cos();
        let (dx, dy) = ((right - left) * w, (bottom - top) * h);
        // Lightroom stores the rotated diagonal, not the level rectangle.
        let (cw, ch) = (dx * cos + dy * sin, -dx * sin + dy * cos);
        let size = if transposed { [ch, cw] } else { [cw, ch] };
        let [x, y] = crate::lightroom::orient(centre, w, h, geometry.orientation);
        // Mirroring orientations reverse the direction of rotation.
        let mirrored = matches!(geometry.orientation, 2 | 4 | 5 | 7);
        let angle = if mirrored {
            geometry.angle
        } else {
            -geometry.angle
        };
        let crop = Self {
            center: [x / width, y / height],
            size: [size[0] / width, size[1] / height],
            angle,
        };
        // An uncropped photo stays exactly uncropped despite rounding.
        let near = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-9);
        if near(&crop.center, &Self::FULL.center) && near(&crop.size, &[1., 1.]) && angle == 0. {
            Self::FULL
        } else {
            crop
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(width: u32, height: u32) -> ImageBuf {
        let mut image = ImageBuf::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let (u, v) = (x as f64 / width as f64, y as f64 / height as f64);
                image.set(
                    x,
                    y,
                    [
                        from_f64(u),
                        from_f64(v),
                        from_f64((u * 7.).sin() * (v * 5.).cos()),
                    ],
                );
            }
        }
        image
    }

    #[test]
    fn full_crop_is_the_identity() {
        let image = gradient(9, 5);
        assert_eq!(Crop::FULL.apply(&image).unwrap().data, image.data);
        assert_eq!(Crop::FULL.long_side_fraction(9., 5.), 1.);
    }

    #[test]
    fn invalid_crop_json_is_rejected_before_allocating() {
        let image = gradient(10, 10);
        for json in [
            r#"{"center":[0.5,0.5],"size":[1e300,1e300],"angle":0}"#,
            r#"{"center":[0.5,0.5],"size":[2,1],"angle":0}"#,
            r#"{"center":[0.5,0.5],"size":[0,1],"angle":0}"#,
            r#"{"center":[0.5,0.5],"size":[-1,1],"angle":0}"#,
            r#"{"center":[1.5,0.5],"size":[0.5,0.5],"angle":0}"#,
            r#"{"center":[0.5,0.5],"size":[0.5,0.5],"angle":46}"#,
        ] {
            let crop: Crop = serde_json::from_str(json).unwrap();
            assert!(crop.apply(&image).is_err(), "accepted {json}");
        }
    }

    #[test]
    fn straightened_crop_can_be_wider_than_the_frame() {
        let image = gradient(10, 10);
        let crop = Crop {
            center: [0.5, 0.5],
            size: [1.2, 0.1],
            angle: 45.,
        };
        let output = crop.apply(&image).unwrap();
        assert_eq!((output.width, output.height), (12, 1));
    }

    #[test]
    fn small_crops_and_clamped_edges_are_allowed() {
        let image = gradient(10, 10);
        let small = Crop {
            size: [1e-10, 1e-10],
            ..Crop::FULL
        }
        .apply(&image)
        .unwrap();
        assert_eq!((small.width, small.height), (1, 1));
        let rounded = Crop {
            center: [0.5 + 1e-7, 0.5],
            ..Crop::FULL
        }
        .apply(&image)
        .unwrap();
        assert_eq!((rounded.width, rounded.height), (10, 10));
        let tilted = Crop {
            angle: 10.,
            ..Crop::FULL
        }
        .apply(&image)
        .unwrap();
        assert_eq!((tilted.width, tilted.height), (10, 10));
    }

    #[test]
    fn empty_images_are_rejected() {
        for (width, height) in [(0, 0), (1, 0), (0, 1)] {
            let image = ImageBuf::new(width, height);
            assert!(Crop::FULL.apply(&image).is_err());
            assert!(
                Crop {
                    size: [0.5, 0.5],
                    ..Crop::FULL
                }
                .apply(&image)
                .is_err()
            );
        }
    }

    #[test]
    fn level_crop_copies_the_region() {
        let image = gradient(40, 20);
        let crop = Crop {
            center: [0.25, 0.5],
            size: [0.5, 0.5],
            angle: 0.,
        };
        let out = crop.apply(&image).unwrap();
        assert_eq!((out.width, out.height), (20, 10));
        assert_eq!(out.get(0, 0), image.get(0, 5));
        assert_eq!(out.get(19, 9), image.get(19, 14));
        assert_eq!(crop.long_side_fraction(40., 20.), 0.5);
    }

    #[test]
    fn quarter_turns_keep_the_same_region() {
        let image = gradient(30, 20);
        let crop = Crop {
            center: [0.4, 0.6],
            size: [0.5, 0.3],
            angle: 7.,
        };
        let cropped = crop.apply(&image).unwrap();
        for turns in 0..4 {
            let rotated = crop
                .rotated_quarter_turns(turns)
                .apply(&image.rotated_quarter_turns(turns))
                .unwrap();
            let expected = cropped.rotated_quarter_turns(turns);
            assert_eq!(
                (rotated.width, rotated.height),
                (expected.width, expected.height)
            );
            let error = rotated
                .data
                .iter()
                .zip(&expected.data)
                .map(|(a, b)| (*a as f64 - *b as f64).abs())
                .fold(0., f64::max);
            assert!(error < 1e-5, "turns {turns}: {error}");
        }
    }

    #[test]
    fn lightroom_crops_match_the_catalog_geometry() {
        let source = gradient(48, 32);
        for orientation in 1..=8 {
            for angle in [0., 6., -11.] {
                let geometry = Geometry {
                    orientation,
                    crop: [0.2, 0.15, 0.85, 0.8],
                    angle,
                };
                let expected = geometry.apply(source.clone(), 1).unwrap();
                let frame = geometry
                    .orientation_only()
                    .apply(source.clone(), 1)
                    .unwrap();
                let crop = Crop::from_lightroom(&geometry, frame.width as f64, frame.height as f64);
                let actual = crop.apply(&frame).unwrap();
                assert_eq!(
                    (actual.width, actual.height),
                    (expected.width, expected.height),
                    "orientation {orientation}, angle {angle}"
                );
                let error = actual
                    .data
                    .iter()
                    .zip(&expected.data)
                    .map(|(a, b)| (*a as f64 - *b as f64).abs())
                    .fold(0., f64::max);
                assert!(
                    error < 1e-5,
                    "orientation {orientation}, angle {angle}: {error}"
                );
            }
        }
    }
}
