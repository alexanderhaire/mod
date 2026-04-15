# Ops Dashboard — Windows startup wrapper.
# Usage (manual):   powershell -ExecutionPolicy Bypass -File .\run.ps1
# Usage (service):  install with NSSM pointing at this script, or wrap as a
#                   Windows Scheduled Task set to "Run whether user is logged on or not"
#                   with trigger "At startup".

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# --- Load .env into this process so child streamlit inherits it ---
$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $kv = $line.Split("=", 2)
            $key = $kv[0].Trim()
            $val = $kv[1].Trim().Trim('"').Trim("'")
            [System.Environment]::SetEnvironmentVariable($key, $val, "Process")
        }
    }
    Write-Host "Loaded environment from .env"
} else {
    Write-Warning ".env not found; falling back to .streamlit/secrets.toml (not recommended for production)"
}

# --- Activate venv if present ---
$venvActivate = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    . $venvActivate
}

# --- Log directory ---
$logDir = Join-Path $PSScriptRoot "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logFile = Join-Path $logDir "opsdash-$stamp.log"

$address = if ($env:STREAMLIT_SERVER_ADDRESS) { $env:STREAMLIT_SERVER_ADDRESS } else { "127.0.0.1" }
$port    = if ($env:STREAMLIT_SERVER_PORT)    { $env:STREAMLIT_SERVER_PORT }    else { "8501" }

Write-Host "Starting opsdash on ${address}:${port} (logs: $logFile)"
& streamlit run app.py `
    --server.address $address `
    --server.port $port `
    --server.headless true `
    --browser.gatherUsageStats false `
    *>&1 | Tee-Object -FilePath $logFile
