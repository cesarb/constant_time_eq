use std::env;

fn main() {
    if rustversion::cfg!(since(1.80)) {
        println!("cargo:rustc-check-cfg=cfg(count_instructions_test)");
        println!("cargo:rustc-check-cfg=cfg(inline_asm_is_stable)");
    }

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").expect("target_arch not set");
    let inline_asm_is_stable = match target_arch.as_str() {
        "x86" | "x86_64" | "arm" | "aarch64" | "riscv32" | "riscv64" => true,
        "loongarch64" if rustversion::cfg!(since(1.72)) => true,
        "arm64ec" | "s390x" if rustversion::cfg!(since(1.84)) => true,
        "loongarch32" if rustversion::cfg!(since(1.91)) => true,
        "powerpc" | "powerpc64" if rustversion::cfg!(since(1.95)) => true,
        _ => false,
    };
    if inline_asm_is_stable {
        println!("cargo:rustc-cfg=inline_asm_is_stable");
    }

    if option_env!("COUNT_INSTRUCTIONS_TEST").map_or(false, |v| !v.is_empty()) {
        println!("cargo:rustc-cfg=count_instructions_test");
    }
}
