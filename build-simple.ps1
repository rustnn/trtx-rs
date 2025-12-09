#!/usr/bin/env pwsh
# Simple build script with GPU support

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Building with GPU Support" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Set environment variables
$env:CUDA_ROOT = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:TENSORRT_RTX_DIR = "D:\TensorRT-10.7.0.23"
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
$env:PATH = "C:\Program Files\LLVM\bin;$env:PATH"

Write-Host "Environment configured:" -ForegroundColor Green
Write-Host "  CUDA_ROOT = $env:CUDA_ROOT"
Write-Host "  TENSORRT_RTX_DIR = $env:TENSORRT_RTX_DIR"
Write-Host "  LIBCLANG_PATH = $env:LIBCLANG_PATH"
Write-Host ""

# Clean and build
Write-Host "Cleaning..." -ForegroundColor Cyan
cargo clean

Write-Host ""
Write-Host "Building (this may take several minutes)..." -ForegroundColor Cyan
Write-Host ""

cargo build --verbose

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[OK] Build successful!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Running tests..." -ForegroundColor Cyan
    Write-Host ""

    cargo test --verbose

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "================================================" -ForegroundColor Green
        Write-Host "  All Tests Passed!" -ForegroundColor Green
        Write-Host "================================================" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "[!] Some tests failed" -ForegroundColor Yellow
    }
} else {
    Write-Host ""
    Write-Host "[!] Build failed" -ForegroundColor Red
}
