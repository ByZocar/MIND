# =====================================================================
# setup.ps1 - Windows-native equivalent of `make setup`.
#
# Creates the virtualenv, installs dependencies, installs the `acv`
# package in editable mode, and copies .env from template. Aborts on
# the first pip failure (no silent successes).
#
# Usage:
#   .\setup.ps1
# =====================================================================

$ErrorActionPreference = "Stop"

$VENV   = ".venv"
$PYTHON = "python"

function Invoke-Checked {
    param([string]$Description, [scriptblock]$Cmd)
    Write-Host "==> $Description" -ForegroundColor Cyan
    & $Cmd
    if ($LASTEXITCODE -ne 0) {
        throw ("FAILED: " + $Description + " (exit " + $LASTEXITCODE + "). Check the log above.")
    }
}

Write-Host "==> Checking Python..." -ForegroundColor Cyan
& $PYTHON --version
if ($LASTEXITCODE -ne 0) {
    throw "Python not found in PATH. Install Python 3.12 from python.org and retry."
}

if (-not (Test-Path $VENV)) {
    Invoke-Checked "Creating virtualenv at $VENV" { & $PYTHON -m venv $VENV }
} else {
    Write-Host "==> Virtualenv already exists at $VENV (reusing)." -ForegroundColor Yellow
}

$Pip = Join-Path $VENV "Scripts\pip.exe"
$Py  = Join-Path $VENV "Scripts\python.exe"

Invoke-Checked "Upgrading pip" { & $Py -m pip install --upgrade pip }

Invoke-Checked "Installing requirements.txt (2-5 min first time)" {
    & $Pip install -r requirements.txt
}

Invoke-Checked "Installing project package (pip install -e .)" {
    & $Pip install -e .
}

if (-not (Test-Path ".env")) {
    Write-Host "==> Copying .env.example -> .env" -ForegroundColor Cyan
    Copy-Item ".env.example" ".env"
} else {
    Write-Host "==> .env already exists (not overwritten)." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Setup OK." -ForegroundColor Green
Write-Host "Activate the venv and verify:" -ForegroundColor Green
Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "    python -m pytest -q" -ForegroundColor Green
