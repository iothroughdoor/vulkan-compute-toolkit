use std::process::Command;
use std::fs;

fn main() {
    //println!("cargo:rerun-if-changed=src/shader/add_one.glsl");
    //println!("cargo:rerun-if-changed=src/shader/sum_array.glsl");

    match fs::create_dir_all("data/shader") {
        Err(e) => {
            if e.kind() != std::io::ErrorKind::AlreadyExists {
                panic!("Could not create data/shader path");
            }
        },
        Ok(_) => {}
    };

    glsl_compile("src/shader/add_one.glsl", "data/shader/add_one.spv");
    glsl_compile("src/shader/sum_array.glsl", "data/shader/sum_array.spv");
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
