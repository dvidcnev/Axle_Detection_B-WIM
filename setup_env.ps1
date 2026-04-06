# setup_env.ps1
# -------------
# Creates the project .venv (Python 3.12) and installs all dependencies.
# Run once from the project root:
#   .\setup_env.ps1
#
# If Python 3.12 is not on PATH, download it from https://www.python.org/downloads/

param(
    [ValidateSet("cpu", "cu121", "cu118")]
    [string]$Cuda = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot   # directory containing this script
Set-Location $ProjectRoot

# ── 1. Find Python 3.12 ────────────────────────────────────────────────────
Write-Host "`n[1/4] Locating Python 3.12..." -ForegroundColor Cyan

$py = $null
foreach ($candidate in @("py", "python3.12", "python3", "python")) {
    try {
        $ver = & $candidate --version 2>&1
        if ($ver -match "3\.12") { $py = $candidate; break }
    } catch {}
}

if (-not $py) {
    Write-Host "ERROR: Python 3.12 not found on PATH." -ForegroundColor Red
    Write-Host "Download it from https://www.python.org/downloads/release/python-3120/" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during install." -ForegroundColor Yellow
    exit 1
}

$pyVer = (& $py --version 2>&1).ToString().Trim()
Write-Host "  Using: $py  ($pyVer)" -ForegroundColor Green

# ── 2. Create .venv ────────────────────────────────────────────────────────
Write-Host "`n[2/4] Creating .venv..." -ForegroundColor Cyan
$venvPath = Join-Path $ProjectRoot ".venv"

if (Test-Path $venvPath) {
    Write-Host "  .venv already exists — skipping creation." -ForegroundColor Yellow
} else {
    & $py -m venv .venv
    Write-Host "  Created: $venvPath" -ForegroundColor Green
}

$venvPy  = Join-Path $venvPath "Scripts\python.exe"
$venvPip = Join-Path $venvPath "Scripts\pip.exe"

# ── 3. Install PyTorch ─────────────────────────────────────────────────────
Write-Host "`n[3/4] Installing PyTorch..." -ForegroundColor Cyan

if ($Cuda -eq "") {
    Write-Host "  Which version of PyTorch do you want?" -ForegroundColor Yellow
    Write-Host "    [1] CPU only       (no GPU, smaller download)"
    Write-Host "    [2] CUDA 12.1      (NVIDIA GPU, recommended)"
    Write-Host "    [3] CUDA 11.8      (older NVIDIA GPU)"
    $choice = Read-Host "  Enter 1, 2, or 3"
    $Cuda = switch ($choice) {
        "2"     { "cu121" }
        "3"     { "cu118" }
        default { "cpu" }
    }
}

switch ($Cuda) {
    "cu121" {
        Write-Host "  Installing PyTorch with CUDA 12.1..." -ForegroundColor Green
        & $venvPip install torch --index-url https://download.pytorch.org/whl/cu121
    }
    "cu118" {
        Write-Host "  Installing PyTorch with CUDA 11.8..." -ForegroundColor Green
        & $venvPip install torch --index-url https://download.pytorch.org/whl/cu118
    }
    default {
        Write-Host "  Installing PyTorch (CPU only)..." -ForegroundColor Green
        & $venvPip install torch
    }
}

# ── 4. Install remaining requirements ──────────────────────────────────────
Write-Host "`n[4/4] Installing remaining requirements..." -ForegroundColor Cyan

# Install only the non-torch lines from requirements.txt
$reqLines = Get-Content (Join-Path $ProjectRoot "requirements.txt") |
    Where-Object { $_ -notmatch "^\s*#" -and $_ -notmatch "torch" -and $_.Trim() -ne "" }

$tmpReq = Join-Path $env:TEMP "axle_reqs_tmp.txt"
$reqLines | Set-Content $tmpReq
& $venvPip install -r $tmpReq
Remove-Item $tmpReq -Force

# ── Done ───────────────────────────────────────────────────────────────────
Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "  Python : $venvPy"
Write-Host ""
Write-Host "  To activate manually:" -ForegroundColor Yellow
Write-Host "    .venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "  To train (no activation needed):" -ForegroundColor Yellow
Write-Host "    python scripts\run_training.py --model tcn"
Write-Host "================================================`n" -ForegroundColor Cyan
