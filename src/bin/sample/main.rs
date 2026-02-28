use std::error::Error;
use std::process;

use stablediffusion::model::stablediffusion::{
    Diffuser, DiffuserConfig, Embedder, EmbedderConfig,
    LatentDecoder, LatentDecoderConfig, RESOLUTIONS,
    RawImages, 
};

use stablediffusion::backend::Backend;

use burn::{
    config::Config,
    module::Module,
    tensor::{self, ElementConversion, Tensor},
};

use burn::record::{NamedMpkFileRecorder, HalfPrecisionSettings, Recorder};

// Use LibTorch backend with CUDA if available, otherwise use WGPU for GPU acceleration
use burn_tch::{LibTorch, LibTorchDevice};
use burn_wgpu::{Wgpu, WgpuDevice};
use tch;
use std::io::{self, Write};

fn load_embedder_model<B: Backend>(model_dir: &str, device: &B::Device) -> Result<Embedder<B>, Box<dyn Error>> {
    print!("  Loading config... ");
    io::stdout().flush().unwrap();
    let config = EmbedderConfig::load(&format!("{}.cfg", model_dir))?;
    println!("✓");
    
    print!("  Loading model weights from disk (this may take a while)... ");
    io::stdout().flush().unwrap();
    let record = NamedMpkFileRecorder::<HalfPrecisionSettings>::new().load(model_dir.into(), device)?;
    println!("✓");
    
    print!("  Initializing model structure... ");
    io::stdout().flush().unwrap();
    let model = config.init(device);
    println!("✓");
    
    print!("  Loading weights into model... ");
    io::stdout().flush().unwrap();
    let model = model.load_record(record);
    println!("✓");

    Ok(model)
}

fn load_diffuser_model<B: Backend>(model_dir: &str, device: &B::Device) -> Result<Diffuser<B>, Box<dyn Error>> {
    print!("  Loading config... ");
    io::stdout().flush().unwrap();
    let config = DiffuserConfig::load(&format!("{}.cfg", model_dir))?;
    println!("✓");
    
    print!("  Loading model weights from disk (this may take a while - UNet is large)... ");
    io::stdout().flush().unwrap();
    let record = NamedMpkFileRecorder::<HalfPrecisionSettings>::new().load(model_dir.into(), device)?;
    println!("✓");
    
    print!("  Initializing UNet structure... ");
    io::stdout().flush().unwrap();
    let model = config.init(device);
    println!("✓");
    
    print!("  Loading weights into UNet (transferring to GPU if using CUDA)... ");
    io::stdout().flush().unwrap();
    let model = model.load_record(record);
    println!("✓");

    Ok(model)
}

fn load_latent_decoder_model<B: Backend>(
    model_dir: &str,
    device: &B::Device
) -> Result<LatentDecoder<B>, Box<dyn Error>> {
    print!("  Loading config... ");
    io::stdout().flush().unwrap();
    let config = LatentDecoderConfig::load(&format!("{}.cfg", model_dir))?;
    println!("✓");
    
    print!("  Loading model weights from disk... ");
    io::stdout().flush().unwrap();
    let record = NamedMpkFileRecorder::<HalfPrecisionSettings>::new().load(model_dir.into(), device)?;
    println!("✓");
    
    print!("  Initializing autoencoder structure... ");
    io::stdout().flush().unwrap();
    let model = config.init(device);
    println!("✓");
    
    print!("  Loading weights into autoencoder... ");
    io::stdout().flush().unwrap();
    let model = model.load_record(record);
    println!("✓");

    Ok(model)
}

#[allow(dead_code)]
fn arb_tensor<B: Backend, const D: usize>(dims: [usize; D], device: &B::Device) -> Tensor<B, D> {
    let prod: usize = dims.iter().cloned().product();
    Tensor::arange(0..prod as i64, device).float().sin().reshape(dims)
}

use stablediffusion::model::stablediffusion::Conditioning;

use stablediffusion::backend_converter::*;

use burn::tensor::Bool;

use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
struct Opts {
    /// Directory of the model weights
    #[structopt(parse(from_os_str), short = "md", long)]
    model_dir: PathBuf, 

    /// Use the refiner model?
    #[structopt(short = "ref", long)]
    use_refiner: bool, 

    /// Path of the reference image for inpainting
    #[structopt(parse(from_os_str), short = "rd", long)]
    reference_img: Option<PathBuf>, 

    /// Left-most pixel of the crop window
    #[structopt(long)]
    crop_left: Option<usize>,
    
    /// Right-most pixel of the crop window
    #[structopt(long)]
    crop_right: Option<usize>,

    /// Top-most pixel of the crop window
    #[structopt(long)]
    crop_top: Option<usize>,

    /// Bottom-most pixel of the crop window
    #[structopt(long)]
    crop_bottom: Option<usize>,

    /// Crop outside or inside the specified crop window?
    #[structopt(long)]
    crop_out: bool, 

    /// Controls the strength of the adherence to the prompt
    #[structopt(short = "gs", long, default_value = "7.5")]
    unconditional_guidance_scale: f64,

    /// Number of diffusion iterations used for generating the image
    #[structopt(short = "steps", long, default_value = "30")]
    n_diffusion_steps: usize,

    #[structopt(short = "pr", long)]
    prompt: String,

    /// Directory of the image outputs
    #[structopt(parse(from_os_str), short = "od", long)]
    output_dir: PathBuf,

    /// Image width (default: 1024)
    #[structopt(short = "w", long, default_value = "1024")]
    width: i32,

    /// Image height (default: 1024)
    #[structopt(short = "h", long, default_value = "1024")]
    height: i32,
}

// Type aliases for backends
// LibTorch with CUDA for best performance, WGPU as fallback for GPU via Vulkan
type TorchBackend = LibTorch<f32>;
type TorchBackendF16 = LibTorch<tensor::f16>;
type WgpuBackend = Wgpu;

struct InpaintingTensors<B: Backend> {
    orig_dims: (usize, usize), 
    reference_latent: Tensor<B, 4>, 
    mask: Tensor<B, 4, Bool>, 
}

fn prompt_backend_choice() -> u8 {
    loop {
        print!("Select backend:\n  1: torch+burn (LibTorch with CUDA)\n  2: burn (WGPU - pure Burn backend)\nChoice [1/2]: ");
        io::stdout().flush().unwrap();
        
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();
        
        match input {
            "1" => {
                println!("Selected: torch+burn (LibTorch backend)");
                return 1;
            }
            "2" => {
                println!("Selected: burn (WGPU backend)");
                return 2;
            }
            _ => {
                println!("Invalid choice. Please enter 1 or 2.");
            }
        }
    }
}

fn main() {
    let backend_choice = prompt_backend_choice();
    
    match backend_choice {
        1 => {
            // Path 1: LibTorch backend (current implementation)
            println!("\n=== Using LibTorch Backend ===");
            if tch::Cuda::is_available() && tch::Cuda::device_count() > 0 {
                println!("CUDA detected! Using LibTorch with CUDA GPU");
                println!("Device: CUDA(0)");
                let device = LibTorchDevice::Cuda(0);
                if let Err(e) = run_with_torch_device(device) {
                    eprintln!("CUDA failed: {}. Falling back to CPU...", e);
                    let cpu_device = LibTorchDevice::Cpu;
                    if let Err(e) = run_with_torch_device(cpu_device) {
                        eprintln!("Error: {}", e);
                        std::process::exit(1);
                    }
                }
            } else {
                println!("CUDA not available. Using CPU (will be slow).");
                let device = LibTorchDevice::Cpu;
                if let Err(e) = run_with_torch_device(device) {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        2 => {
            // Path 2: WGPU backend (pure Burn)
            println!("\n=== Using WGPU Backend (Pure Burn) ===");
            run_with_wgpu_backend();
        }
        _ => {
            eprintln!("Invalid backend choice");
            std::process::exit(1);
        }
    }
}

fn run_with_wgpu_backend() {
    let device = WgpuDevice::default();
    println!("Device: {:?}", device);
    println!("WGPU will use GPU acceleration via Vulkan/DirectX12/Metal");
    
    let opts = Opts::from_args();
    
    // WGPU uses f32, but models are stored in f16
    // We'll load them and convert - this should work via backend converter
    println!("\nNote: WGPU backend uses f32 precision.");
    println!("Models will be loaded and converted from f16 to f32 automatically.");
    
    if let Err(e) = run_with_wgpu_device(device, opts) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run_with_wgpu_device(device: WgpuDevice, opts: Opts) -> Result<(), Box<dyn std::error::Error>> {
    // WGPU implementation - similar structure to LibTorch but using WGPU backend
    // Note: This is experimental and may have performance differences
    
    // Precompute model paths to avoid repeated format!() calls
    let model_dir_str = opts.model_dir.to_str().unwrap();
    let embedder_path = format!("{}/embedder", model_dir_str);
    let diffuser_path = format!("{}/diffuser", model_dir_str);
    let refiner_path = format!("{}/refiner", model_dir_str);
    let latent_decoder_path = format!("{}/latent_decoder", model_dir_str);
    
    // For inpainting (if needed)
    let _inpainting_info: Option<()> = opts.reference_img.map(|_ref_dir| {
        println!("Inpainting not yet implemented for WGPU backend.");
        println!("Please use option 1 (torch+burn) for inpainting support.");
        std::process::exit(0);
    });

    let conditioning = {
        println!("Loading embedder...");
        // Try loading with f16 first, then convert if needed
        // WGPU should handle the conversion automatically
        let embedder: Embedder<WgpuBackend> =
            load_embedder_model_wgpu(&embedder_path, &device)?;

        let resolution = [opts.height, opts.width];  // [height, width] format
        
        // Warn if resolution is not in training set
        if !RESOLUTIONS.iter().any(|&[h, w]| h == resolution[0] && w == resolution[1]) {
            println!("Warning: Resolution {}x{} is not in the training set, but will be attempted.", resolution[1], resolution[0]);
        }

        let size = Tensor::from_ints(resolution, &device).unsqueeze();
        let crop = Tensor::from_ints([0, 0], &device).unsqueeze();
        let ar = Tensor::from_ints(resolution, &device).unsqueeze();

        println!("Running embedder...");
        embedder.text_to_conditioning(&opts.prompt, size, crop, ar)
    };

    // WGPU uses f32, so we don't need f16 conversion
    // But we need to ensure the conditioning matches the diffuser backend
    let conditioning_wgpu: Conditioning<WgpuBackend> = conditioning;

    let latent = {
        println!("Loading diffuser...");
        let diffuser: Diffuser<WgpuBackend> =
            load_diffuser_model_wgpu(&diffuser_path, &device)?;

        println!("Running diffuser...");
        // Only clone conditioning once for the diffusion process
        diffuser.sample_latent(conditioning_wgpu.clone(), opts.unconditional_guidance_scale, opts.n_diffusion_steps)
    };

    let latent = if opts.use_refiner {
        println!("Loading refiner...");
        let diffuser: Diffuser<WgpuBackend> =
            load_diffuser_model_wgpu(&refiner_path, &device)?;

        println!("Running refiner...");
        diffuser.refine_latent(
            latent,
            conditioning_wgpu,
            opts.unconditional_guidance_scale,
            800,
            opts.n_diffusion_steps,
        )
    } else {
        latent
    };

    let images = {
        println!("Loading latent decoder...");
        let latent_decoder: LatentDecoder<WgpuBackend> =
            load_latent_decoder_model_wgpu(&latent_decoder_path, &device)?;

        println!("Running decoder...");
        latent_decoder.latent_to_image(latent)
    };

    println!("Saving images...");
    save_images(
        &images.buffer,
        opts.output_dir.to_str().unwrap(),
        images.width as u32,
        images.height as u32,
    )?;
    println!("Done.");

    Ok(())
}

// WGPU-specific model loading functions
// These try to load f16 models and convert to f32 for WGPU
fn load_embedder_model_wgpu<B: Backend>(model_dir: &str, device: &B::Device) -> Result<Embedder<B>, Box<dyn Error>> {
    print!("  Loading config... ");
    io::stdout().flush().unwrap();
    let config = EmbedderConfig::load(&format!("{}.cfg", model_dir))?;
    println!("✓");
    
    print!("  Loading model weights from disk... ");
    io::stdout().flush().unwrap();
    let record = NamedMpkFileRecorder::<HalfPrecisionSettings>::new().load(model_dir.into(), device)?;
    println!("✓");
    
    print!("  Initializing model structure... ");
    io::stdout().flush().unwrap();
    let model = config.init(device);
    println!("✓");
    
    print!("  Loading weights into model (converting f16 to f32)... ");
    io::stdout().flush().unwrap();
    let model = model.load_record(record);
    println!("✓");

    Ok(model)
}

fn load_diffuser_model_wgpu<B: Backend>(model_dir: &str, device: &B::Device) -> Result<Diffuser<B>, Box<dyn Error>> {
    print!("  Loading config... ");
    io::stdout().flush().unwrap();
    let config = DiffuserConfig::load(&format!("{}.cfg", model_dir))?;
    println!("✓");
    
    print!("  Loading model weights from disk (this may take a while - UNet is large)... ");
    io::stdout().flush().unwrap();
    let record = NamedMpkFileRecorder::<HalfPrecisionSettings>::new().load(model_dir.into(), device)?;
    println!("✓");
    
    print!("  Initializing UNet structure... ");
    io::stdout().flush().unwrap();
    let model = config.init(device);
    println!("✓");
    
    print!("  Loading weights into UNet (converting f16 to f32, transferring to GPU)... ");
    io::stdout().flush().unwrap();
    let model = model.load_record(record);
    println!("✓");

    Ok(model)
}

fn load_latent_decoder_model_wgpu<B: Backend>(
    model_dir: &str,
    device: &B::Device
) -> Result<LatentDecoder<B>, Box<dyn Error>> {
    print!("  Loading config... ");
    io::stdout().flush().unwrap();
    let config = LatentDecoderConfig::load(&format!("{}.cfg", model_dir))?;
    println!("✓");
    
    print!("  Loading model weights from disk... ");
    io::stdout().flush().unwrap();
    let record = NamedMpkFileRecorder::<HalfPrecisionSettings>::new().load(model_dir.into(), device)?;
    println!("✓");
    
    print!("  Initializing autoencoder structure... ");
    io::stdout().flush().unwrap();
    let model = config.init(device);
    println!("✓");
    
    print!("  Loading weights into autoencoder (converting f16 to f32)... ");
    io::stdout().flush().unwrap();
    let model = model.load_record(record);
    println!("✓");

    Ok(model)
}

fn run_with_torch_device(device: LibTorchDevice) -> Result<(), Box<dyn std::error::Error>> {
    let opts = Opts::from_args();
    
    // Precompute model paths to avoid repeated format!() calls
    let model_dir_str = opts.model_dir.to_str().unwrap();
    let embedder_path = format!("{}/embedder", model_dir_str);
    let diffuser_path = format!("{}/diffuser", model_dir_str);
    let refiner_path = format!("{}/refiner", model_dir_str);
    let latent_decoder_path = format!("{}/latent_decoder", model_dir_str);

    let inpainting_info = opts.reference_img.map(|ref_dir| {
        let imgs = load_images(&[ref_dir.to_str().unwrap().into()]).unwrap();

        if !RESOLUTIONS.iter().any(|&[h, w]| h as usize == imgs.height && w as usize == imgs.width) {
            println!("Reference image dimensions are incompatible.\nThe compatible dimensions are:");
            for [h, w] in RESOLUTIONS {
                println!("Width: {}, Height: {}", w, h);
            }
            process::exit(1);
        }

        let crop_left = opts.crop_left.unwrap_or(0);
        let crop_right = opts.crop_right.unwrap_or(imgs.width);
        let crop_top = opts.crop_top.unwrap_or(0);
        let crop_bottom = opts.crop_bottom.unwrap_or(imgs.height);

        assert!(crop_right <= imgs.width && crop_bottom <= imgs.height && crop_left < crop_right || crop_top < crop_bottom, "Invalid crop parameters.");

        // compute latent
        println!("Loading latent encoder...");
        let latent_decoder: LatentDecoder<TorchBackend> =
            load_latent_decoder_model(&latent_decoder_path, &device).unwrap();

        println!("Running encoder...");

        let latent = latent_decoder.image_to_latent(&imgs, &device);
        let latent: Tensor<TorchBackendF16, 4> = DefaultBackendConverter::new().convert(latent, &device);

        // get converted pixels idxs
        let [_, _, height, width] = latent.dims();
        let scale = imgs.height / height;
        let crop_left = crop_left / scale;
        let crop_right = crop_right / scale;
        let crop_top = crop_top / scale;
        let crop_bottom = crop_bottom / scale;

        // compute mask
        let crop_width = crop_right - crop_left;
        let crop_height = crop_bottom - crop_top;

        let pad_left = crop_left;
        let pad_right = width - crop_right;

        let pad_top = crop_top;
        let pad_bottom = height - crop_bottom;

        let mask = Tensor::<TorchBackendF16, 2>::ones([crop_height, crop_width], &device)
            .pad( (pad_left, pad_right, pad_top, pad_bottom), 0.0_f32.elem() )
            .bool()
            .unsqueeze::<4>()
            .expand([1, 4, height, width]);
        let mask = if opts.crop_out {
            mask.bool_not()
        } else {
            mask
        };

        InpaintingTensors::<TorchBackendF16> {
            orig_dims: (imgs.width, imgs.height), 
            reference_latent: latent, 
            mask: mask.unsqueeze::<4>(), 
        }
    });

    let conditioning = {
        println!("Loading embedder...");
        let embedder: Embedder<TorchBackend> =
            load_embedder_model(&embedder_path, &device).unwrap();
        

        let resolution = if let Some(inpainting_info) = inpainting_info.as_ref() {
            [inpainting_info.orig_dims.1 as i32, inpainting_info.orig_dims.0 as i32]
        } else {
            [opts.height, opts.width]  // [height, width] format
        };
        
        // Warn if resolution is not in training set (but still allow it)
        if !RESOLUTIONS.iter().any(|&[h, w]| h == resolution[0] && w == resolution[1]) {
            println!("Warning: Resolution {}x{} is not in the training set, but will be attempted.", resolution[1], resolution[0]);
            println!("Supported resolutions include: 512x512, 768x768, 1024x1024, and various aspect ratios.");
        }

        // Use Burn's generic tensor creation APIs
        let size = Tensor::from_ints(resolution, &device).unsqueeze();
        let crop = Tensor::from_ints([0, 0], &device).unsqueeze();
        let ar = Tensor::from_ints(resolution, &device).unsqueeze();

        println!("Running embedder...");
        embedder.text_to_conditioning(&opts.prompt, size, crop, ar)
    };

    let conditioning: Conditioning<TorchBackendF16> =
        conditioning.convert(DefaultBackendConverter::new(), &device);

    let latent = {
        println!("Loading diffuser...");
        let diffuser: Diffuser<TorchBackendF16> =
            load_diffuser_model(&diffuser_path, &device).unwrap();

        if let Some(inpainting_info) = inpainting_info {
            // Reduce clones: pass conditioning by reference where possible
            diffuser.sample_latent_with_inpainting(conditioning.clone(), opts.unconditional_guidance_scale, opts.n_diffusion_steps, inpainting_info.reference_latent, inpainting_info.mask)
        } else {
            println!("Running diffuser...");
            // Only clone conditioning once for the diffusion process
            diffuser.sample_latent(conditioning.clone(), opts.unconditional_guidance_scale, opts.n_diffusion_steps)
        }
    };

    let latent = if opts.use_refiner {
        println!("Loading refiner...");
        let diffuser: Diffuser<TorchBackendF16> =
            load_diffuser_model(&refiner_path, &device).unwrap();

        println!("Running refiner...");
        diffuser.refine_latent(
            latent,
            conditioning, // Move conditioning here (no longer needed after)
            opts.unconditional_guidance_scale,
            800,
            opts.n_diffusion_steps,
        )
    } else {
        latent
    };

    let latent: Tensor<TorchBackend, 4> = DefaultBackendConverter::new().convert(latent, &device);

    let images = {
        println!("Loading latent decoder...");
        let latent_decoder: LatentDecoder<TorchBackend> =
            load_latent_decoder_model(&latent_decoder_path, &device).unwrap();

        println!("Running decoder...");
        latent_decoder.latent_to_image(latent)
    };

    println!("Saving images...");
    save_images(
        &images.buffer,
        opts.output_dir.to_str().unwrap(),
        images.width as u32,
        images.height as u32,
    )?;
    println!("Done.");

    Ok(())
}

use image::{self, ColorType::Rgb8, RgbImage, ImageError, ImageResult};
use image::io::Reader as ImageReader;

fn load_images(filenames: &[String]) -> Result<RawImages, ImgLoadError> {
    let images = filenames
        .into_iter()
        .map(|filename| load_image(&filename))
        .collect::<ImageResult<Vec<RgbImage>>>()?;

    let (width, height) = images.first().map(|img| img.dimensions()).ok_or(ImgLoadError::NoImages)?;

    if !images.iter().map(|img| img.dimensions()).all(|d| d == (width, height) ) {
        return Err(ImgLoadError::DifferentDimensions);
    }

    let image_buffers: Vec<Vec<u8>> = images
        .into_iter()
        .map(|image| image.into_vec())
        .collect();

    Ok(
        RawImages {
            buffer: image_buffers, 
            width: width as usize, 
            height: height as usize, 
        }
    )
}

#[derive(Debug)]
enum ImgLoadError {
    DifferentDimensions, 
    NoImages, 
    #[allow(dead_code)]
    ImageError(ImageError), 
}

impl From<ImageError> for ImgLoadError {
    fn from(err: ImageError) -> Self {
        ImgLoadError::ImageError(err)
    }
}

fn load_image(filename: &str) -> ImageResult<RgbImage> {
    Ok(
        ImageReader::open(filename)?.decode()?.to_rgb8()
    )
}

fn save_images(images: &Vec<Vec<u8>>, basepath: &str, width: u32, height: u32) -> ImageResult<()> {
    for (index, img_data) in images.iter().enumerate() {
        let path = format!("{}{}.png", basepath, index);
        image::save_buffer(path, &img_data[..], width, height, Rgb8)?;
    }

    Ok(())
}

// save red test image
#[allow(dead_code)]
fn save_test_image() -> ImageResult<()> {
    let width = 256;
    let height = 256;
    let raw: Vec<_> = (0..width * height)
        .into_iter()
        .flat_map(|i| {
            let row = i / width;
            let red = (255.0 * row as f64 / height as f64) as u8;

            [red, 0, 0]
        })
        .collect();

    image::save_buffer("red.png", &raw[..], width, height, Rgb8)
}
