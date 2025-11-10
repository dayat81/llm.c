$ErrorActionPreference = "Stop"
$version = "12.6.1"
$installerName = "cuda_12.6.1_560.94_windows.exe"
$downloadUrl = "https://developer.download.nvidia.com/compute/cuda/$version/local_installers/$installerName"
$downloadDir = Join-Path $env:USERPROFILE "Downloads"
if (!(Test-Path $downloadDir)) {
    New-Item -ItemType Directory -Path $downloadDir | Out-Null
}
$installerPath = Join-Path $downloadDir $installerName

function Write-Log($msg) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $msg"
}

if (-Not (Test-Path $installerPath)) {
    Write-Log "Downloading CUDA Toolkit $version from $downloadUrl"
    Invoke-WebRequest -Uri $downloadUrl -OutFile $installerPath -UseBasicParsing
    Write-Log "Download complete: $installerPath"
} else {
    Write-Log "Installer already exists at $installerPath; skipping download"
}

Write-Log "Launching silent installer (this may take several minutes)..."
$arguments = "-s"
$process = Start-Process -FilePath $installerPath -ArgumentList $arguments -PassThru
$process.WaitForExit()

if ($process.ExitCode -ne 0) {
    Write-Log "Installer exited with code $($process.ExitCode)."
    exit $process.ExitCode
}

Write-Log "CUDA Toolkit $version installation finished."
