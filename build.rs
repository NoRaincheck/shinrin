use std::path::PathBuf;

fn main() {
    let cpp_dir = PathBuf::from("src/shinrin/_corels/cpp");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++11")
        // Vendored CORELS is compiled without -DGMP: bit vectors use the
        // plain word-array fallback, so there is no libgmp dependency.
        .include(cpp_dir.join("corels"))
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
        build.file(cpp_dir.join(name));
    }

    build.compile("shinrin_corels");

    let target = std::env::var("TARGET").unwrap();
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
    } else if target.contains("linux") {
        println!("cargo:rustc-link-lib=stdc++");
    }
}
