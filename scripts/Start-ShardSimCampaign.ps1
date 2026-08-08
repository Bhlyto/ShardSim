param(
    [Parameter(Mandatory = $true)]
    [string]$Campaign,
    [int]$CasesPerBatch = 50,
    [int]$Epochs = 150,
    [int]$ModelBatchSize = 8,
    [int]$Width = 16,
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu",
    [switch]$TestAtEnd
)

$ErrorActionPreference = "Stop"
$repository = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repository ".venv\Scripts\python.exe"
$campaignPath = [System.IO.Path]::GetFullPath((Join-Path $repository $Campaign))
$logDirectory = Join-Path $campaignPath "background"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Environnement Python introuvable: $python"
}
if (-not (Test-Path -LiteralPath (Join-Path $campaignPath "campaign.lock.json"))) {
    throw "Campagne non verrouillée: $campaignPath"
}

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stdoutPath = Join-Path $logDirectory "$timestamp.stdout.log"
$stderrPath = Join-Path $logDirectory "$timestamp.stderr.log"
$jobPath = Join-Path $logDirectory "current-job.json"
$arguments = @(
    "-m", "shardsim.campaign", "full", $campaignPath,
    "--cases-per-batch", $CasesPerBatch,
    "--algorithm", "heat-residual-unet",
    "--epochs", $Epochs,
    "--batch-size", $ModelBatchSize,
    "--width", $Width,
    "--device", $Device
)
if ($TestAtEnd) {
    $arguments += "--test-at-end"
}

$env:PYTHONPATH = Join-Path $repository "src"
$process = Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $repository `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -WindowStyle Hidden `
    -PassThru

@{
    campaign = $campaignPath
    pid = $process.Id
    started_at = (Get-Date).ToUniversalTime().ToString("o")
    stdout = $stdoutPath
    stderr = $stderrPath
} | ConvertTo-Json | Set-Content -LiteralPath $jobPath -Encoding UTF8

Write-Output "Campagne lancée en arrière-plan (PID $($process.Id))."
Write-Output "Progression: Get-Content -LiteralPath '$stdoutPath' -Wait"
Write-Output "Erreurs:     Get-Content -LiteralPath '$stderrPath' -Wait"
Write-Output "Dashboard:   $campaignPath\outputs\<clé>\reports\dashboard.html"
