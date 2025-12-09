#!/usr/bin/env pwsh
# Download a minimal ONNX model for testing

Write-Host "Downloading minimal ONNX model..." -ForegroundColor Cyan

$url = "https://github.com/onnx/models/raw/main/validated/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx"
$output = "trtx\tests\data\super-resolution-10.onnx"

# Create directory if it doesn't exist
$dir = Split-Path -Parent $output
if (!(Test-Path $dir)) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
}

try {
    Invoke-WebRequest -Uri $url -OutFile $output -UseBasicParsing

    if (Test-Path $output) {
        $size = (Get-Item $output).Length
        $sizeKB = [math]::Round($size / 1KB, 2)
        Write-Host "Downloaded successfully: $sizeKB KB" -ForegroundColor Green
        Write-Host "Location: $output" -ForegroundColor Green
    } else {
        Write-Host "Download failed - file not found" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Download failed: $_" -ForegroundColor Red
    exit 1
}
