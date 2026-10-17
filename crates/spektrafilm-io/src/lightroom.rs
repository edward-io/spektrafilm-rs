//! Read-only Lightroom Classic catalog import. Develop settings are data, never Lua code.
use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use spektrafilm_math::{image::ImageBuf, precision::from_f64};
use std::{
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Clone, Debug)]
pub struct Photo {
    pub id: i64,
    pub path: PathBuf,
    pub available: bool,
    pub copy_name: String,
    /// Lightroom's pick flag: 1 accepted, 0 unflagged, -1 rejected.
    pub flag: i8,
    pub geometry: Result<Geometry, String>,
    pub framing_warning: bool,
}

pub struct Catalog {
    pub path: PathBuf,
    pub photos: Vec<Photo>,
}

impl Catalog {
    pub fn open(path: &Path) -> Result<Self> {
        let mut connection = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .context("opening Lightroom catalog read-only")?;
        connection.busy_timeout(Duration::from_millis(500))?;
        connection.execute_batch("PRAGMA query_only = ON; PRAGMA trusted_schema = OFF;")?;
        // A read transaction keeps paths, flags and Develop settings in one consistent snapshot.
        let transaction = connection.transaction()?;
        // shortcut: Lightroom's internal schema is version dependent; validate new layouts when Adobe changes these tables.
        let mut statement = transaction
            .prepare(
                "SELECT i.id_local, r.absolutePath, fo.pathFromRoot, f.idx_filename,
                    coalesce(i.copyName, ''), CAST(i.pick AS INTEGER), i.orientation,
                    coalesce(d.text, '')
             FROM Adobe_images i
             JOIN AgLibraryFile f ON f.id_local = i.rootFile
             JOIN AgLibraryFolder fo ON fo.id_local = f.folder
             JOIN AgLibraryRootFolder r ON r.id_local = fo.rootFolder
             LEFT JOIN Adobe_imageDevelopSettings d ON d.image = i.id_local
             WHERE i.fileFormat != 'VIDEO'
             ORDER BY i.captureTime, f.idx_filename, i.id_local",
            )
            .context("unsupported Lightroom Classic catalog layout")?;
        let mut photos = statement
            .query_map([], |row| {
                let root: String = row.get(1)?;
                let folder: String = row.get(2)?;
                let filename: String = row.get(3)?;
                let orientation: Option<String> = row.get(6)?;
                let text: String = row.get(7)?;
                let fields = develop_fields(&text);
                let framing_warning = [
                    "LensProfileEnable",
                    "LensManualDistortionAmount",
                    "PerspectiveUpright",
                    "PerspectiveHorizontal",
                    "PerspectiveVertical",
                    "PerspectiveRotate",
                    "PerspectiveAspect",
                    "PerspectiveX",
                    "PerspectiveY",
                ]
                .iter()
                .any(|name| number(&fields, name, 0.).unwrap_or(1.) != 0.)
                    || number(&fields, "PerspectiveScale", 100.).unwrap_or(0.) != 100.;
                let path = PathBuf::from(root).join(folder).join(filename);
                Ok(Photo {
                    id: row.get(0)?,
                    path,
                    available: false,
                    copy_name: row.get(4)?,
                    flag: row.get::<_, i64>(5)?.signum() as i8,
                    geometry: Geometry::from_fields(orientation.as_deref(), &fields)
                        .map_err(|e| e.to_string()),
                    framing_warning,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        drop(statement);
        transaction.commit()?;
        for photo in &mut photos {
            photo.available = photo.path.is_file();
        }
        Ok(Self {
            path: path.to_path_buf(),
            photos,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Geometry {
    /// EXIF orientation of the catalog image, including Lightroom's rotate/flip operations.
    pub orientation: u16,
    /// Opposite crop corners in Lightroom's pre-orientation develop coordinates.
    pub crop: [f64; 4],
    pub angle: f64,
}

impl Geometry {
    fn from_fields(orientation: Option<&str>, fields: &[(&str, &str)]) -> Result<Self> {
        let orientation = match orientation.unwrap_or("AB") {
            "AB" => 1,
            "BA" => 2,
            "CD" => 3,
            "DC" => 4,
            "CB" => 5,
            "BC" => 6,
            "AD" => 7,
            "DA" => 8,
            value => bail!("unknown Lightroom orientation: {value}"),
        };
        let geometry = Self {
            orientation,
            crop: [
                number(fields, "CropLeft", 0.)?,
                number(fields, "CropTop", 0.)?,
                number(fields, "CropRight", 1.)?,
                number(fields, "CropBottom", 1.)?,
            ],
            angle: number(fields, "CropAngle", 0.)?,
        };
        geometry.validate()?;
        Ok(geometry)
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            (1..=8).contains(&self.orientation),
            "invalid catalog orientation"
        );
        ensure!(
            self.angle.is_finite() && self.angle.abs() <= 45.,
            "invalid crop angle"
        );
        ensure!(
            self.crop
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
            "crop boundaries must be finite and between 0 and 1"
        );
        Ok(())
    }

    pub fn apply(&self, image: ImageBuf, source_orientation: u16) -> Result<ImageBuf> {
        self.validate()?;
        ensure!(image.width > 0 && image.height > 0, "empty source image");
        let source_orientation = if (1..=8).contains(&source_orientation) {
            source_orientation
        } else {
            1
        };
        if self.crop == [0., 0., 1., 1.]
            && self.angle == 0.
            && self.orientation == source_orientation
        {
            return Ok(image);
        }
        let (w, h) = if source_orientation >= 5 {
            (image.height as f64, image.width as f64)
        } else {
            (image.width as f64, image.height as f64)
        };
        let [left, top, right, bottom] = self.crop;
        let (cx, cy) = ((left + right) * w / 2., (top + bottom) * h / 2.);
        let (sin, cos) = self.angle.to_radians().sin_cos();
        let (dx, dy) = ((right - left) * w, (bottom - top) * h);
        // Lightroom stores the rotated diagonal, not the width/height of the level rectangle.
        let (cw, ch) = (dx * cos + dy * sin, -dx * sin + dy * cos);
        ensure!(
            cw >= 0.5 && ch >= 0.5 && cw <= w * 1.5 && ch <= h * 1.5,
            "invalid crop dimensions"
        );
        let (width, height) = (cw.round() as u32, ch.round() as u32);
        let (ow, oh) = if self.orientation >= 5 {
            (height, width)
        } else {
            (width, height)
        };
        let mut output = ImageBuf::new(ow, oh);
        let inverse = match self.orientation {
            6 => 8,
            8 => 6,
            value => value,
        };
        output
            .data
            .par_chunks_mut(ow as usize * 3)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, pixel) in row.chunks_exact_mut(3).enumerate() {
                    let [u, v] = orient(
                        [x as f64 + 0.5, y as f64 + 0.5],
                        ow as f64,
                        oh as f64,
                        inverse,
                    );
                    let (u, v) = (
                        (u / width as f64 - 0.5) * cw,
                        (v / height as f64 - 0.5) * ch,
                    );
                    let p = orient(
                        [cx + u * cos - v * sin, cy + u * sin + v * cos],
                        w,
                        h,
                        source_orientation,
                    );
                    let x = (p[0] - 0.5).clamp(0., image.width as f64 - 1.);
                    let y = (p[1] - 0.5).clamp(0., image.height as f64 - 1.);
                    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
                    let (x1, y1) = (
                        (x0 + 1).min(image.width - 1),
                        (y0 + 1).min(image.height - 1),
                    );
                    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
                    for c in 0..3 {
                        let at = |x: u32, y: u32| {
                            image.data[((y * image.width + x) * 3) as usize + c] as f64
                        };
                        pixel[c] = from_f64(
                            (at(x0, y0) * (1. - fx) + at(x1, y0) * fx) * (1. - fy)
                                + (at(x0, y1) * (1. - fx) + at(x1, y1) * fx) * fy,
                        );
                    }
                }
            });
        Ok(output)
    }
}

fn orient([x, y]: [f64; 2], w: f64, h: f64, orientation: u16) -> [f64; 2] {
    match orientation {
        2 => [w - x, y],
        3 => [w - x, h - y],
        4 => [x, h - y],
        5 => [y, x],
        6 => [h - y, x],
        7 => [h - y, w - x],
        8 => [y, w - x],
        _ => [x, y],
    }
}

fn number(fields: &[(&str, &str)], name: &str, default: f64) -> Result<f64> {
    fields
        .iter()
        .find(|(key, _)| *key == name)
        .map(|(_, value)| {
            value
                .parse::<f64>()
                .with_context(|| format!("invalid {name}"))
        })
        .unwrap_or(Ok(default))
}

/// Only top-level scalar assignments count; nested masks and quoted strings cannot supply geometry.
fn develop_fields(text: &str) -> Vec<(&str, &str)> {
    let mut fields = Vec::new();
    let (mut depth, mut quote, mut escaped, mut start) = (0, None, false, 0);
    for (i, byte) in text.bytes().enumerate() {
        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == q {
                quote = None;
            }
            continue;
        }
        match byte {
            b'\'' | b'"' => quote = Some(byte),
            b'{' => {
                depth += 1;
                if depth == 1 {
                    start = i + 1;
                }
            }
            b',' | b'}' if depth == 1 => {
                if let Some((key, value)) = text[start..i].split_once('=') {
                    let value = value.trim();
                    if !value.starts_with(['{', '"', '\'']) {
                        fields.push((key.trim(), value));
                    }
                }
                start = i + 1;
                if byte == b'}' {
                    depth -= 1;
                }
            }
            b'}' => depth -= 1,
            _ => {}
        }
    }
    fields
}

/// Camera previews are used only for catalog thumbnails, never for film development.
/// Longest edge of browser thumbnails; sharp at roughly 80 pt on a 2× display.
pub const THUMBNAIL_PX: u32 = 160;

pub fn thumbnail(path: &Path, orientation: u16) -> Result<image::RgbaImage> {
    let mut image = if super::is_raw(path) {
        let source = rawler::rawsource::RawSource::new(path)?;
        let decoder = rawler::get_decoder(&source)?;
        match decoder.thumbnail_image(&source, &Default::default())? {
            Some(image) => image,
            None => decoder
                .preview_image(&source, &Default::default())?
                .context("no embedded RAW preview")?,
        }
    } else {
        image::ImageReader::open(path)?
            .with_guessed_format()?
            .decode()?
    };
    image = image.thumbnail(THUMBNAIL_PX, THUMBNAIL_PX);
    if let Some(orientation) = image::metadata::Orientation::from_exif(orientation as u8) {
        image.apply_orientation(orientation);
    }
    Ok(image.to_rgba8())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp() -> ImageBuf {
        ImageBuf::from_data(4, 2, (0..8).flat_map(|v| [from_f64(v as f64); 3]).collect())
    }
    fn values(image: &ImageBuf) -> Vec<f64> {
        image.data.chunks_exact(3).map(|p| p[0] as f64).collect()
    }
    #[test]
    fn crop_uses_catalog_defaults_and_ignores_color_and_nested_settings() {
        let fields = develop_fields(
            "s = { Exposure2012 = 4, FilterList = { { CropLeft = 0.9, text = \"},CropTop=1\" } }, CropLeft = 0.25, CropBottom = 0.5 }",
        );
        let geometry = Geometry::from_fields(Some("AB"), &fields).unwrap();
        let cropped = geometry.apply(ramp(), 1).unwrap();
        assert_eq!((cropped.width, cropped.height), (3, 1));
        assert_eq!(values(&cropped), [1., 2., 3.]);
        assert!(Geometry::from_fields(Some("AB"), &develop_fields("s={CropLeft=NaN}")).is_err());
        assert!(Geometry::from_fields(Some("??"), &[]).is_err());
    }
    #[test]
    fn all_catalog_orientations_match_exif_and_source_orientation_is_not_applied_twice() {
        let expected = [
            vec![0., 1., 2., 3., 4., 5., 6., 7.],
            vec![3., 2., 1., 0., 7., 6., 5., 4.],
            vec![7., 6., 5., 4., 3., 2., 1., 0.],
            vec![4., 5., 6., 7., 0., 1., 2., 3.],
            vec![0., 4., 1., 5., 2., 6., 3., 7.],
            vec![4., 0., 5., 1., 6., 2., 7., 3.],
            vec![7., 3., 6., 2., 5., 1., 4., 0.],
            vec![3., 7., 2., 6., 1., 5., 0., 4.],
        ];
        for source in 1..=8 {
            let original = Geometry {
                orientation: source,
                crop: [0., 0., 1., 1.],
                angle: 0.,
            }
            .apply(ramp(), 1)
            .unwrap();
            for catalog in 1..=8 {
                let g = Geometry {
                    orientation: catalog,
                    crop: [0., 0., 1., 1.],
                    angle: 0.,
                };
                let output = g.apply(original.clone(), source).unwrap();
                assert_eq!(
                    values(&output),
                    expected[catalog as usize - 1],
                    "source={source}, catalog={catalog}"
                );
            }
        }
    }
    #[test]
    fn straightened_crop_resolves_the_rotated_diagonal() {
        let image = ImageBuf::from_data(
            10,
            10,
            (0..100).flat_map(|v| [from_f64(v as f64); 3]).collect(),
        );
        let (sin, cos) = 20f64.to_radians().sin_cos();
        let (dx, dy) = (4. * cos - 2. * sin, 4. * sin + 2. * cos);
        let g = Geometry {
            orientation: 6,
            crop: [
                0.5 - dx / 20.,
                0.5 - dy / 20.,
                0.5 + dx / 20.,
                0.5 + dy / 20.,
            ],
            angle: 20.,
        };
        let output = g.apply(image, 1).unwrap();
        assert_eq!((output.width, output.height), (2, 4));
        let first_x = 5. - 1.5 * cos - 0.5 * sin - 0.5;
        let first_y = 5. - 1.5 * sin + 0.5 * cos - 0.5;
        assert!((values(&output)[0] - (first_x + 10. * first_y)).abs() < 1e-4);
    }
    #[test]
    fn catalog_is_read_only_preserves_virtual_copies_and_flags() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("test.lrcat");
        let db = Connection::open(&path).unwrap();
        db.execute_batch("CREATE TABLE Adobe_images(id_local INTEGER, rootFile INTEGER, copyName TEXT, pick REAL, orientation TEXT, fileFormat TEXT, captureTime TEXT);
            CREATE TABLE AgLibraryFile(id_local INTEGER, folder INTEGER, idx_filename TEXT);
            CREATE TABLE AgLibraryFolder(id_local INTEGER, rootFolder INTEGER, pathFromRoot TEXT);
            CREATE TABLE AgLibraryRootFolder(id_local INTEGER, absolutePath TEXT);
            CREATE TABLE Adobe_imageDevelopSettings(image INTEGER, text TEXT);
            INSERT INTO AgLibraryRootFolder VALUES(1,'/photos/');
            INSERT INTO AgLibraryFolder VALUES(2,1,'trip/');
            INSERT INTO AgLibraryFile VALUES(3,2,'frame.ARW');
            INSERT INTO Adobe_images VALUES(4,3,NULL,1.0,'DA','RAW','2026-01-01');
            INSERT INTO Adobe_images VALUES(5,3,'Copy 1',-1.0,'AB','RAW','2026-01-01');
            INSERT INTO Adobe_images VALUES(6,3,NULL,0,'AB','RAW','2026-01-02');
            INSERT INTO Adobe_images VALUES(7,3,NULL,1,NULL,'VIDEO','2026-01-03');
            INSERT INTO Adobe_imageDevelopSettings VALUES(4,'s = { CropLeft=0.25, Exposure2012=5 }');").unwrap();
        drop(db);
        let before = std::fs::read(&path).unwrap();
        let catalog = Catalog::open(&path).unwrap();
        assert_eq!(catalog.photos.len(), 3);
        assert_eq!(
            catalog.photos.iter().map(|p| p.flag).collect::<Vec<_>>(),
            [1, -1, 0]
        );
        assert_eq!(catalog.photos[0].path, Path::new("/photos/trip/frame.ARW"));
        assert_eq!(catalog.photos[1].copy_name, "Copy 1");
        assert_eq!(catalog.photos[0].geometry.as_ref().unwrap().orientation, 8);
        assert_eq!(std::fs::read(&path).unwrap(), before);
        let missing = temp.path().join("missing.lrcat");
        assert!(Catalog::open(&missing).is_err());
        assert!(!missing.exists());
    }
    #[test]
    fn inspect_catalog_if_requested() {
        let Some(path) = std::env::var_os("SPEKTRAFILM_TEST_LIGHTROOM_CATALOG") else {
            return;
        };
        let catalog = Catalog::open(Path::new(&path)).unwrap();
        assert!(!catalog.photos.is_empty());
        println!(
            "{} photos; {} accepted; {} rejected; {} invalid geometry",
            catalog.photos.len(),
            catalog.photos.iter().filter(|p| p.flag == 1).count(),
            catalog.photos.iter().filter(|p| p.flag == -1).count(),
            catalog
                .photos
                .iter()
                .filter(|p| p.geometry.is_err())
                .count()
        );
    }
}
