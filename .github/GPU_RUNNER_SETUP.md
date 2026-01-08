# GPU Runner Setup Guide

This guide explains how to set up a self-hosted Windows runner with NVIDIA T4 GPU for testing trtx-rs with real TensorRT-RTX.

## Prerequisites

### Hardware
- Windows machine with NVIDIA T4 GPU (or compatible GPU)
- Sufficient disk space for dependencies (~10GB)

### Software
- Windows 10/11 or Windows Server
- NVIDIA GPU drivers (latest recommended)
- CUDA Toolkit (compatible with TensorRT-RTX version)
- TensorRT-RTX installation

## Step 1: Install Dependencies

### 1.1 NVIDIA Driver
1. Download from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
2. Install and reboot
3. Verify: `nvidia-smi` should show your T4 GPU

### 1.2 CUDA Toolkit
1. Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Install to default location (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`)
3. Verify: `nvcc --version`

### 1.3 TensorRT-RTX
1. Download from [NVIDIA Developer](https://developer.nvidia.com/tensorrt)
2. Extract to a permanent location (e.g., `C:\TensorRT-RTX`)
3. Set environment variable:
   ```powershell
   [System.Environment]::SetEnvironmentVariable('TENSORRT_RTX_DIR', 'C:\TensorRT-RTX', 'Machine')
   ```

### 1.4 Rust Toolchain
1. Download from [rustup.rs](https://rustup.rs/)
2. Install with default settings
3. Verify: `rustc --version`

## Step 2: Configure GitHub Runner

### 2.1 Add Runner to Repository
1. Go to your GitHub repository
2. Settings → Actions → Runners → New self-hosted runner
3. Choose Windows
4. Follow the installation instructions

### 2.2 Configure Runner Labels
When configuring the runner, add these labels:
- `self-hosted` (automatic)
- `windows` (automatic)
- `gpu` (add manually)
- `t4` (add manually)

To add labels:
```powershell
.\config.cmd --url https://github.com/rustnn/trtx-rs --token YOUR_TOKEN --labels gpu,t4
```

### 2.3 Install as Windows Service (Recommended)
```powershell
# Run as Administrator
.\svc.sh install
.\svc.sh start
```

## Step 3: Set Environment Variables

On the runner machine, set these system environment variables:

```powershell
# As Administrator
[System.Environment]::SetEnvironmentVariable('CUDA_PATH', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x', 'Machine')
[System.Environment]::SetEnvironmentVariable('TENSORRT_RTX_DIR', 'C:\TensorRT-RTX', 'Machine')

# Add to PATH
$oldPath = [System.Environment]::GetEnvironmentVariable('Path', 'Machine')
$newPath = "$oldPath;C:\TensorRT-RTX\lib;$env:CUDA_PATH\bin"
[System.Environment]::SetEnvironmentVariable('Path', $newPath, 'Machine')
```

Restart the runner service after setting variables:
```powershell
.\svc.sh stop
.\svc.sh start
```

## Step 4: Verify Setup

### 4.1 Test GPU Access
```powershell
nvidia-smi
```

Expected output: Should show T4 GPU info

### 4.2 Test CUDA
```powershell
nvcc --version
```

Expected output: CUDA version info

### 4.3 Test TensorRT-RTX
```powershell
dir $env:TENSORRT_RTX_DIR\include
dir $env:TENSORRT_RTX_DIR\lib
```

Expected output: Should list TensorRT headers and libraries

## Step 5: Trigger GPU Tests

### Automatic Triggers
GPU tests run automatically on:
- Push to `main` branch (if trtx code changed)
- Pull requests to `main` (if trtx code changed)

### Manual Trigger
1. Go to Actions tab
2. Select "GPU Tests (Windows T4)"
3. Click "Run workflow"

## Troubleshooting

### Runner Not Picking Up Jobs
- Check runner status in Settings → Actions → Runners
- Verify labels are set correctly (gpu, t4)
- Restart runner service

### CUDA/TensorRT Not Found
- Verify environment variables are set at Machine level (not User)
- Restart runner service after changing variables
- Check PATH includes CUDA and TensorRT lib directories

### Build Failures
- Check CUDA version compatibility with TensorRT-RTX
- Verify GPU drivers are up to date
- Check disk space

### GPU Not Available
- Run `nvidia-smi` on runner machine
- Check GPU is not being used by other processes
- Verify drivers are installed correctly

## Security Considerations

Self-hosted runners have access to repository secrets and can execute code. For security:

1. **Use a dedicated machine**: Don't use a shared or personal workstation
2. **Restrict network access**: Configure firewall rules
3. **Monitor usage**: Review runner logs regularly
4. **Keep updated**: Update OS, drivers, and dependencies
5. **Use runner groups**: Limit which workflows can use the runner

## Cost Considerations

Self-hosted runners are free from GitHub's perspective, but consider:
- Hardware costs (GPU machine)
- Power consumption
- Network bandwidth
- Maintenance time

For occasional testing, consider:
- Using the manual workflow trigger
- Limiting automatic triggers with path filters
- Setting up on-demand runner activation

## Alternative: Cloud GPU Runners

If maintaining a physical machine is not feasible, consider:
- AWS EC2 G4 instances (T4 GPUs)
- Azure NC T4 v3 instances
- Google Cloud Compute with T4 GPUs

Configure these as ephemeral self-hosted runners that spin up on demand.
