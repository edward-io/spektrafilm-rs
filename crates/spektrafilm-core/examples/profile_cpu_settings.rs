use std::{hash::Hasher, hint::black_box, path::Path, time::Instant};
use spektrafilm_core::{params::RuntimeParams, pipeline::Pipeline, profile, spectral_service::{self, SpectraLut}};
use spektrafilm_math::{image::ImageBuf, precision::from_f64};

fn main() {
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let spectra = spectral_service::load_spectra_lut(&data).unwrap();
    for sigma in [0.25, 1., 2.5] {
        for run in 0..4 {
            let mut table = SpectraLut {size: spectra.size, n_wavelengths: spectra.n_wavelengths, data: spectra.data.clone()};
            let start = Instant::now();
            table.blur_wavelengths(black_box(sigma));
            let ms = start.elapsed().as_secs_f64()*1000.;
            let mut hash = std::collections::hash_map::DefaultHasher::new();
            for v in &table.data {hash.write_u32(v.to_bits());}
            println!("BLUR sigma={sigma} run={run} ms={ms:.3} hash={:016x}",hash.finish());
        }
    }
    for (w,h) in [(2048,1365),(7008,4672)] {
        let input = ImageBuf::from_data(w,h,(0..w as usize*h as usize*3).map(|i|from_f64((i%4093)as f64/2048.-0.02)).collect());
        for encoded in [false,true] {
            let mut p=RuntimeParams::default();
            p.workflow.route="input".into();
            p.io.input_color_space="sRGB".into();
            p.io.input_cctf_decoding=encoded;
            p.io.output_color_space=if encoded {"Display P3"} else {"sRGB"}.into();
            p.io.output_cctf_encoding=true;
            let pipeline=Pipeline::new(profile::load_profile_by_name(&data,"kodak_gold_200").unwrap(),profile::load_profile_by_name(&data,"kodak_portra_endura").unwrap(),p);
            for run in 0..4 {
                let image=input.clone();
                let start=Instant::now();
                let output=pipeline.process(black_box(image),&spektrafilm_gpu::cpu_backend::CpuBackend);
                let ms=start.elapsed().as_secs_f64()*1000.;
                let mut hash=std::collections::hash_map::DefaultHasher::new();
                for v in &output.data {hash.write_u64((*v as f64).to_bits());}
                println!("INPUT w={w} h={h} encoded={encoded} run={run} ms={ms:.3} hash={:016x}",hash.finish());
            }
        }
    }
}
