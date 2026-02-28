#!/bin/bash
# Setup script to enable CUDA support for tch crate
# This script detects your CUDA version and sets TORCH_CUDA_VERSION accordingly

set -e

echo "=== CUDA Setup for Stable Diffusion XL ==="
echo ""

# Detect CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    echo "Detected CUDA version: $CUDA_VERSION"
else
    echo "Warning: nvcc not found. CUDA toolkit may not be installed."
    echo "Please install CUDA toolkit or set TORCH_CUDA_VERSION manually."
    exit 1
fi

# Map CUDA version to TORCH_CUDA_VERSION
# PyTorch/LibTorch provides prebuilt binaries for specific CUDA versions
# Note: CUDA 11.4 is compatible with cu118 (CUDA 11.8 build) due to backward compatibility
# Common versions: cu118 (11.8), cu121 (12.1), cu124 (12.4)
case "$CUDA_VERSION" in
    11.4|11.5|11.6|11.7|11.8)
        TORCH_CUDA="cu118"
        echo "Using TORCH_CUDA_VERSION=cu118 (CUDA 11.8 build, compatible with CUDA 11.4+)"
        echo "Note: CUDA 11.4 is backward compatible with CUDA 11.8 LibTorch build"
        ;;
    12.0|12.1)
        TORCH_CUDA="cu121"
        echo "Using TORCH_CUDA_VERSION=cu121 (compatible with CUDA 12.0-12.1)"
        ;;
    12.2|12.3|12.4)
        TORCH_CUDA="cu124"
        echo "Using TORCH_CUDA_VERSION=cu124 (compatible with CUDA 12.2-12.4)"
        ;;
    *)
        echo "Warning: CUDA version $CUDA_VERSION may not have a direct LibTorch build."
        echo "Trying cu118 (most common, backward compatible with CUDA 11.x)..."
        TORCH_CUDA="cu118"
        ;;
esac

echo ""
echo "Setting TORCH_CUDA_VERSION=$TORCH_CUDA"
export TORCH_CUDA_VERSION=$TORCH_CUDA

echo ""
echo "To make this permanent, add to your ~/.bashrc or ~/.zshrc:"
echo "  export TORCH_CUDA_VERSION=$TORCH_CUDA"
echo ""
echo "Now cleaning and rebuilding with CUDA support..."
echo ""

# Clean previous builds to force re-download of LibTorch
cargo clean

# Build with CUDA support
echo "Building with CUDA support (this will download CUDA-enabled LibTorch)..."
cargo build --release

echo ""
echo "=== Setup complete! ==="
echo "CUDA support should now be enabled. Run your application to verify."
