use std::path::PathBuf;

fn main() {
    // Track the vendored C++ trees (sources AND headers) so edits actually
    // trigger a rebuild. Note: emitting any rerun-if-* directive switches
    // cargo to "only rerun when these change" mode, so these two lines are
    // load-bearing — without them, stale C++ objects silently link against
    // fresh Rust bindings.
    println!("cargo:rerun-if-changed=src/shinrin/_corels/cpp");
    println!("cargo:rerun-if-changed=src/shinrin/_spot/cpp");
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

    // ------------------------------------------------------------------
    // Vendored GOSDT engine ("Fast Sparse Decision Tree Optimization via
    // Reference Ensembles") + serial TBB shim + C ABI bridge. GOSDT
    // unconditionally includes <gmp.h>, so the mini-gmp shim is always in
    // scope here (the mpn_logical.c additions live in the plain-C build).
    let corels_cpp_dir = PathBuf::from("src/shinrin/_corels/cpp");
    let gosdt_dir = PathBuf::from("src/shinrin/_spot/cpp");
    let libgosdt_include = gosdt_dir.join("libgosdt/include");
    let mut gosdt = cc::Build::new();
    gosdt.cpp(true)
        .std("c++20")
        .define("GMP", None)
        .include(corels_cpp_dir.join("gmpshim"))
        .include(gosdt_dir.join("tbbshim"))
        .include(&gosdt_dir)
        .include(&libgosdt_include)
        .flag_if_supported("-O3")
        .warnings(false);

    for name in [
        "libgosdt/src/bitmask.cpp",
        "libgosdt/src/bitset.cpp",
        "libgosdt/src/configuration.cpp",
        "libgosdt/src/dataset.cpp",
        "libgosdt/src/diagnosis/false_convergence.cpp",
        "libgosdt/src/diagnosis/non_convergence.cpp",
        "libgosdt/src/diagnosis/trace.cpp",
        "libgosdt/src/dispatch/dispatch.cpp",
        "libgosdt/src/extraction/models.cpp",
        "libgosdt/src/gosdt.cpp",
        "libgosdt/src/graph.cpp",
        "libgosdt/src/local_state.cpp",
        "libgosdt/src/message.cpp",
        "libgosdt/src/model.cpp",
        "libgosdt/src/optimizer.cpp",
        "libgosdt/src/queue.cpp",
        "libgosdt/src/task.cpp",
        "bridge_gosdt.cpp",
    ] {
        gosdt.file(gosdt_dir.join(name));
    }

    gosdt.compile("shinrin_gosdt");

    // mini-gmp: GMP's portable mpz_t implementation, vendored from the
    // official 6.3.0 tarball (see minigmp/README.md). Provides the GMP
    // API (plus the mpn logical ops in gmpshim/mpn_logical.c) without any
    // system dependency.
    let mut c = cc::Build::new();
    c.file(cpp_dir.join("minigmp/mini-gmp.c"))
        .file(corels_cpp_dir.join("gmpshim/mpn_logical.c"))
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
