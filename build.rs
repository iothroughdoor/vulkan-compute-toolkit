use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/shader/add_one.comp");

    glsl_compile("src/shader/add_one.comp", "data/shader/add_one.spv");
}

fn glsl_compile(input_path: &str, output_path: &str) {
    let mut glslc = Command::new("glslc");
    let status = glslc
        .arg(input_path)
        .arg("--target-spv=spv1.3")
        .arg("-o")
        .arg(output_path)
        .status()
        .expect("compilation went wrong");
    assert!(status.success(), "glsl compilation failed");
}
