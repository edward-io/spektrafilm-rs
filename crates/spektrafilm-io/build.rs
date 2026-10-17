use std::{env, fs, path::PathBuf};
fn main() {
    println!("cargo:rerun-if-changed=vendor/lensfun-0.3.4");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_BUNDLED_LENSFUN");
    if env::var_os("CARGO_FEATURE_BUNDLED_LENSFUN").is_none() {
        return;
    }
    let source = PathBuf::from("vendor/lensfun-0.3.4");
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let target = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let windows = target == "windows";
    let compiler = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let mut config = String::from(
        "#ifndef __CONFIG_H__\n#define __CONFIG_H__\n#define CONF_PACKAGE \"lensfun\"\n#define CONF_VERSION \"0.3.4.0\"\n#define CONF_COMPILER_GCC 1\n#define CONF_LENSFUN_INTERNAL\n#define CONF_DATADIR \"/usr/share/lensfun\"\n#define SYSTEM_DB_UPDATE_PATH \"/var/lib/lensfun-updates\"\n#define DATABASE_SUBDIR \"version_1\"\n#define GLIB_VERSION_MIN_REQUIRED GLIB_VERSION_2_26\n#define GLIB_VERSION_MAX_ALLOWED GLIB_VERSION_MIN_REQUIRED\n",
    );
    if windows {
        config.push_str("#define PLATFORM_WINDOWS 1\n");
    }
    if target == "linux" {
        config.push_str("#define HAVE_ENDIAN_H 1\n");
    }
    if compiler == "msvc" {
        config.push_str("#define _USE_MATH_DEFINES 1\n");
    }
    config.push_str("#endif\n");
    fs::write(output.join("config.h"), config).unwrap();
    let header = fs::read_to_string(source.join("include/lensfun/lensfun.h.in"))
        .unwrap()
        .replace("@VERSION_MAJOR@", "0")
        .replace("@VERSION_MINOR@", "3")
        .replace("@VERSION_MICRO@", "4")
        .replace("@VERSION_BUGFIX@", "0");
    assert!(!header.contains("@VERSION_"));
    fs::write(output.join("lensfun.h"), header).unwrap();
    // Link static GLib and its available static dependencies, avoiding a runtime
    // Lensfun/GLib installation. Only platform libraries remain dynamic on macOS.
    let mut glib_config = pkg_config::Config::new();
    let glib=glib_config.statik(true).cargo_metadata(false).probe("glib-2.0")
        .expect("Building built-in Lensfun requires GLib development headers and pkg-config (macOS: brew install glib pkg-config). Lensfun itself need not be installed.");
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++11")
        .warnings(false)
        .define("CONF_LENSFUN_STATIC", None)
        .include(&output)
        .include(source.join("libs/lensfun"))
        .includes(&glib.include_paths);
    if compiler == "msvc" {
        build
            .include(source.join("include/regex"))
            .include(source.join("libs/regex"));
    }
    if compiler != "msvc" {
        build.define("HAVE_REGEX_H", None);
    }
    for file in [
        "camera.cpp",
        "database.cpp",
        "lens.cpp",
        "mount.cpp",
        "cpuid.cpp",
        "mod-color-sse.cpp",
        "mod-color-sse2.cpp",
        "mod-color.cpp",
        "mod-coord-sse.cpp",
        "mod-coord.cpp",
        "mod-subpix.cpp",
        "modifier.cpp",
        "auxfun.cpp",
    ] {
        build.file(source.join("libs/lensfun").join(file));
    }
    if windows {
        build.file(source.join("libs/lensfun/windows/auxfun.cpp"));
    }
    build.compile("spektrafilm_lensfun");
    // Use private archive names. macOS's linker may otherwise choose an
    // adjacent .dylib for a common -l name even when Cargo requested static.
    // Private names also prevent another dependency from overriding the mode.
    for library in &glib.libs {
        let archive = glib
            .link_paths
            .iter()
            .map(|p| p.join(format!("lib{library}.a")))
            .find(|p| p.is_file());
        if let Some(archive) = archive {
            let private = format!("spektrafilm_{}", library.replace(['-', '.'], "_"));
            fs::copy(archive, output.join(format!("lib{private}.a"))).unwrap();
            println!("cargo:rustc-link-lib=static={private}");
        } else {
            assert!(
                library != "glib-2.0",
                "The bundled Lensfun build needs the GLib static archive"
            );
            println!("cargo:rustc-link-lib={library}");
        }
    }
    for framework in &glib.frameworks {
        println!("cargo:rustc-link-lib=framework={framework}");
    }
    for path in &glib.framework_paths {
        println!("cargo:rustc-link-search=framework={}", path.display());
    }
    for arguments in &glib.ld_args {
        println!("cargo:rustc-link-arg=-Wl,{}", arguments.join(","));
    }
    if compiler == "msvc" {
        let mut regex = cc::Build::new();
        regex
            .warnings(false)
            .define("HAVE_SYS_TYPES_H", None)
            .define("TRE_REGEX_T_FIELD", "value")
            .define("TRE_VERSION", "\"0.8.0\"")
            .define("inline", "_inline")
            .include(source.join("include/regex"))
            .include(source.join("libs/regex"));
        for entry in fs::read_dir(source.join("libs/regex")).unwrap().flatten() {
            if entry.path().extension().is_some_and(|e| e == "c") {
                regex.file(entry.path());
            }
        }
        regex.compile("spektrafilm_lensfun_regex");
    }
}
