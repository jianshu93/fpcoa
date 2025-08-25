#[cfg(feature = "macos")]
fn main() {
    println!("cargo:rustc-link-lib=framework=Accelerate");
}

#[cfg(not(any(feature = "macos")))]
fn main() {}
