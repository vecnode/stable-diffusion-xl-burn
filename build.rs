fn main() {
    // Check if TORCH_CUDA_VERSION is set
    if std::env::var("TORCH_CUDA_VERSION").is_err() {
        // Try to detect CUDA version from system
        if let Ok(output) = std::process::Command::new("nvcc")
            .arg("--version")
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            if let Some(version_line) = output_str.lines().find(|l| l.contains("release")) {
                if let Some(version) = version_line.split_whitespace().find_map(|s| {
                    let v = s.trim_end_matches(',');
                    if v.starts_with(|c: char| c.is_ascii_digit()) {
                        Some(v)
                    } else {
                        None
                    }
                }) {
                    // Map CUDA version to TORCH_CUDA_VERSION format
                    // PyTorch provides builds for: cu118 (11.8), cu121 (12.1), cu124 (12.4)
                    // CUDA 11.4-11.7 are compatible with cu118 due to backward compatibility
                    let version_parts: Vec<&str> = version.split('.').collect();
                    if version_parts.len() >= 2 {
                        let major: u32 = version_parts[0].parse().unwrap_or(0);
                        let minor: u32 = version_parts[1].parse().unwrap_or(0);
                        
                        let torch_cuda_version = if major == 11 && minor >= 4 && minor <= 8 {
                            "cu118".to_string()  // CUDA 11.4-11.8 use cu118 build
                        } else if major == 12 && minor <= 1 {
                            "cu121".to_string()  // CUDA 12.0-12.1 use cu121 build
                        } else if major == 12 && minor >= 2 && minor <= 4 {
                            "cu124".to_string()  // CUDA 12.2-12.4 use cu124 build
                        } else {
                            format!("cu{}{}", major, minor)  // Fallback
                        };
                        
                        println!("cargo:warning=TORCH_CUDA_VERSION not set. Detected CUDA {} -> use TORCH_CUDA_VERSION={}", version, torch_cuda_version);
                        println!("cargo:warning=To enable CUDA support, run: ./setup-cuda.sh");
                        println!("cargo:warning=Or manually: export TORCH_CUDA_VERSION={} && cargo clean && cargo build --release", torch_cuda_version);
                        
                        // Note: We can't set env vars for dependencies in build.rs
                        // The user needs to set it before running cargo
                        return;
                    }
                }
            }
        }
        
        // If no CUDA detected, provide helpful message
        println!("cargo:warning=TORCH_CUDA_VERSION not set. LibTorch will be built without CUDA support.");
        println!("cargo:warning=To enable CUDA support:");
        println!("cargo:warning=  1. Set TORCH_CUDA_VERSION environment variable (e.g., cu118, cu121, cu128)");
        println!("cargo:warning=  2. Example: export TORCH_CUDA_VERSION=cu118");
        println!("cargo:warning=  3. Then rebuild: cargo clean && cargo build --release");
        println!("cargo:warning=");
        println!("cargo:warning=Common CUDA versions:");
        println!("cargo:warning=  CUDA 11.8 -> TORCH_CUDA_VERSION=cu118");
        println!("cargo:warning=  CUDA 12.1 -> TORCH_CUDA_VERSION=cu121");
        println!("cargo:warning=  CUDA 12.4 -> TORCH_CUDA_VERSION=cu124");
    } else {
        let cuda_version = std::env::var("TORCH_CUDA_VERSION").unwrap();
        println!("cargo:warning=TORCH_CUDA_VERSION={} is set. LibTorch will be built with CUDA support.", cuda_version);
    }
}
