use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=SHINRIN_CORELS_NO_GMP");
    // Setting SHINRIN_CORELS_NO_GMP=1 builds CORELS without -DGMP, using
    // its word-array bit-vector fallback (useful for benchmarking the
    // mini-gmp path against upstream's no-GMP configuration).
    let no_gmp = std::env::var("SHINRIN_CORELS_NO_GMP").is_ok_and(|v| v != "0");

    let cpp_dir = PathBuf::from("src/shinrin/_corels/cpp");

    // C++ CORELS engine + bridge. Compiled with -DGMP (unless disabled via
    // SHINRIN_CORELS_NO_GMP), with a shim include dir placed first so
    // `#include <gmp.h>` resolves to the vendored mini-gmp rather than a
    // system libgmp.
    let mut cxx = cc::Build::new();
    cxx.cpp(true)
        .std("c++11");
    if !no_gmp {
        cxx.define("GMP", None);
    }
    cxx.include(cpp_dir.join("gmpshim"))
        .include(cpp_dir.join("corels"))
        .include(cpp_dir.join("mining"))
        .include(&cpp_dir)
        .flag_if_supported("-O3")
        .warnings(false);

    for name in [
        "corels/cache.cpp",
        "corels/corels.cpp",
        "corels/pmap.cpp",
        "corels/rulelib.cpp",
        "corels/run.cpp",
        "corels/utils.cpp",
        "mining/utils.cpp",
        "bridge.cpp",
    ] {
        cxx.file(cpp_dir.join(name));
    }

    // Compiled before the mini-gmp static lib so linkers that resolve
    // archives strictly left-to-right see consumers before providers.
    cxx.compile("shinrin_corels");

    // mini-gmp: GMP's portable mpz_t implementation, vendored from the
    // official 6.3.0 tarball (see minigmp/README.md). Provides the GMP
    // API without any system dependency.
    let mut c = cc::Build::new();
    c.file(cpp_dir.join("minigmp/mini-gmp.c"))
        .flag_if_supported("-O3")
        .warnings(false);
    c.compile("shinrin_minigmp");

    let target = std::env::var("TARGET").unwrap();
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
    } else if target.contains("linux") {
        println!("cargo:rustc-link-lib=stdc++");
    }
}
