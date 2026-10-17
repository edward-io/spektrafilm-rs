//! Pinned Lensfun 0.3 C ABI. Normal builds link the upstream implementation in.
use anyhow::{Context, Result};
use std::ffi::{c_char, c_void};
type Ptr = *mut c_void;
// lfComponentRole in the vendored header: UNKNOWN=2, INTENSITY=3,
// RED=4, GREEN=5, BLUE=6. This is an RGB triplet of f32 samples.
pub const RGB_COMPONENTS: i32 = 4 | (5 << 4) | (6 << 8);
#[repr(C)]
pub struct Camera {
    pub maker: *const c_char,
    pub model: *const c_char,
    pub variant: *const c_char,
    pub mount: *const c_char,
    pub crop: f32,
    pub score: i32,
}
type New = unsafe extern "C" fn() -> Ptr;
type Destroy = unsafe extern "C" fn(Ptr);
type Load = unsafe extern "C" fn(Ptr) -> i32;
type LoadDir = unsafe extern "C" fn(Ptr, *const c_char) -> i32;
type Cameras = unsafe extern "C" fn(Ptr, *const c_char, *const c_char, i32) -> *mut *const Camera;
type Lenses =
    unsafe extern "C" fn(Ptr, *const Camera, *const c_char, *const c_char, i32) -> *mut Ptr;
type ModifierNew = unsafe extern "C" fn(Ptr, f32, i32, i32) -> Ptr;
type Initialize = unsafe extern "C" fn(Ptr, Ptr, i32, f32, f32, f32, f32, i32, i32, i32) -> i32;
type Color = unsafe extern "C" fn(Ptr, Ptr, f32, f32, i32, i32, i32, i32) -> i32;
type Geometry = unsafe extern "C" fn(Ptr, f32, f32, i32, i32, *mut f32) -> i32;
pub struct Api {
    pub new: New,
    pub destroy: Destroy,
    pub load: Load,
    pub load_dir: LoadDir,
    pub cameras: Cameras,
    pub lenses: Lenses,
    pub free: Destroy,
    pub modifier_new: ModifierNew,
    pub modifier_destroy: Destroy,
    pub initialize: Initialize,
    pub color: Color,
    pub geometry: Geometry,
    _library: Option<libloading::Library>,
}
#[cfg(feature = "bundled-lensfun")]
unsafe extern "C" {
    fn lf_db_new() -> Ptr;
    fn lf_db_destroy(db: Ptr);
    fn lf_db_load(db: Ptr) -> i32;
    fn lf_db_load_directory(db: Ptr, path: *const c_char) -> i32;
    fn lf_db_find_cameras_ext(
        db: Ptr,
        make: *const c_char,
        model: *const c_char,
        flags: i32,
    ) -> *mut *const Camera;
    fn lf_db_find_lenses_hd(
        db: Ptr,
        camera: *const Camera,
        make: *const c_char,
        model: *const c_char,
        flags: i32,
    ) -> *mut Ptr;
    fn lf_free(memory: Ptr);
    fn lf_modifier_new(lens: Ptr, crop: f32, width: i32, height: i32) -> Ptr;
    fn lf_modifier_destroy(modifier: Ptr);
    fn lf_modifier_initialize(
        modifier: Ptr,
        lens: Ptr,
        format: i32,
        focal: f32,
        aperture: f32,
        distance: f32,
        scale: f32,
        target: i32,
        flags: i32,
        reverse: i32,
    ) -> i32;
    fn lf_modifier_apply_color_modification(
        modifier: Ptr,
        pixels: Ptr,
        x: f32,
        y: f32,
        width: i32,
        height: i32,
        desc: i32,
        stride: i32,
    ) -> i32;
    fn lf_modifier_apply_subpixel_geometry_distortion(
        modifier: Ptr,
        x: f32,
        y: f32,
        width: i32,
        height: i32,
        coords: *mut f32,
    ) -> i32;
}
impl Api {
    pub fn new() -> Result<Self> {
        if let Some(path) = std::env::var_os("SPEKTRAFILM_LENSFUN_LIBRARY") {
            let library=unsafe{libloading::Library::new(&path)}.with_context(||format!("Could not load the explicitly selected Lensfun library {}. Unset SPEKTRAFILM_LENSFUN_LIBRARY to use built-in Lensfun.",path.to_string_lossy()))?;
            return unsafe { Self::dynamic(library) };
        }
        #[cfg(feature = "bundled-lensfun")]
        {
            Ok(Self {
                new: lf_db_new,
                destroy: lf_db_destroy,
                load: lf_db_load,
                load_dir: lf_db_load_directory,
                cameras: lf_db_find_cameras_ext,
                lenses: lf_db_find_lenses_hd,
                free: lf_free,
                modifier_new: lf_modifier_new,
                modifier_destroy: lf_modifier_destroy,
                initialize: lf_modifier_initialize,
                color: lf_modifier_apply_color_modification,
                geometry: lf_modifier_apply_subpixel_geometry_distortion,
                _library: None,
            })
        }
        #[cfg(not(feature = "bundled-lensfun"))]
        {
            let library=["liblensfun.1.dylib","/opt/homebrew/lib/liblensfun.1.dylib","/usr/local/lib/liblensfun.1.dylib","liblensfun.0.3.4.dylib","liblensfun.so.1","lensfun.dll"].iter().find_map(|p|unsafe{libloading::Library::new(p).ok()}).context("This build disabled built-in Lensfun; install Lensfun or rebuild with the default features")?;
            unsafe { Self::dynamic(library) }
        }
    }
    unsafe fn dynamic(library: libloading::Library) -> Result<Self> {
        // SAFETY: each pointer is resolved with the pinned 0.3 ABI; the retained
        // library outlives every database/modifier guard in lens::correct.
        unsafe {
            Ok(Self {
                new: *library.get(b"lf_db_new\0")?,
                destroy: *library.get(b"lf_db_destroy\0")?,
                load: *library.get(b"lf_db_load\0")?,
                load_dir: *library.get(b"lf_db_load_directory\0")?,
                cameras: *library.get(b"lf_db_find_cameras_ext\0")?,
                lenses: *library.get(b"lf_db_find_lenses_hd\0")?,
                free: *library.get(b"lf_free\0")?,
                modifier_new: *library.get(b"lf_modifier_new\0")?,
                modifier_destroy: *library.get(b"lf_modifier_destroy\0")?,
                initialize: *library.get(b"lf_modifier_initialize\0")?,
                color: *library.get(b"lf_modifier_apply_color_modification\0")?,
                geometry: *library.get(b"lf_modifier_apply_subpixel_geometry_distortion\0")?,
                _library: Some(library),
            })
        }
    }
}
