#!/usr/bin/env pwsh
# Run GPU tests with TensorRT

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Running GPU Tests with TensorRT" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Set environment variables
$env:CUDA_ROOT = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:TENSORRT_RTX_DIR = "D:\TensorRT-10.7.0.23"
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
# Add TensorRT and CUDA DLL paths to PATH for runtime
$env:PATH = "D:\TensorRT-10.7.0.23\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;C:\Program Files\LLVM\bin;$env:PATH"

Write-Host "Environment configured:" -ForegroundColor Green
Write-Host "  CUDA_ROOT = $env:CUDA_ROOT"
Write-Host "  TENSORRT_RTX_DIR = $env:TENSORRT_RTX_DIR"
Write-Host "  LIBCLANG_PATH = $env:LIBCLANG_PATH"
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
    Write-Host "[!] Some tests failed (exit code: $LASTEXITCODE)" -ForegroundColor Yellow
}
