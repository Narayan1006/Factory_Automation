# ==============================================================================
# Bosch Digital Twin - Unified Demo Orchestrator
# AI-Driven Digital Twin for Smart Factory Operations
# ==============================================================================
param (
    [ValidateSet("A", "B", "C", "D")]
    [string]$Scenario = "A",
    [double]$Speed = 120.0,
    [int]$MaxEvents = 0,
    [switch]$NoReplay
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = (Get-Item "$ScriptDir\..").FullName
Set-Location $ProjectRoot

# Ensure logs dir exists
$LogDir = Join-Path $ProjectRoot "artifacts\logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
}

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  BOSCH LINE 3 DIGITAL TWIN - LIVE DEMONSTRATION" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  Active Scenario  : Scenario $Scenario" -ForegroundColor Yellow
Write-Host "  Simulation Speed : $Speed x (1 sim-hour in $([math]::Round(3600.0/$Speed, 1))s)" -ForegroundColor Yellow
Write-Host "  Project Root     : $ProjectRoot" -ForegroundColor DarkGray
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check & Start Docker Compose Infrastructure
Write-Host "[1/4] Checking Docker stack (Mosquitto, InfluxDB, Grafana)..." -ForegroundColor Green
$dockerPs = docker compose -f infra/docker-compose.yml ps --format json | ConvertFrom-Json 2>$null
if (-not $dockerPs -or ($dockerPs.Count -lt 3)) {
    Write-Host "      Starting containers..." -ForegroundColor Yellow
    docker compose -f infra/docker-compose.yml up -d
    Start-Sleep -Seconds 3
} else {
    Write-Host "      Containers are healthy and running." -ForegroundColor Green
}

# 2. Launch Ingestion Service in Background
Write-Host "[2/4] Starting MQTT -> InfluxDB Ingestion Daemon..." -ForegroundColor Green
$IngestLog = Join-Path $LogDir "ingest.log"
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

$ingestProc = Start-Process -FilePath $PythonExe `
    -ArgumentList "-m", "twin.ingest.mqtt_to_influx" `
    -RedirectStandardOutput $IngestLog `
    -RedirectStandardError $IngestLog `
    -PassThru

Write-Host "      Ingestion PID: $($ingestProc.Id) (Logging to artifacts/logs/ingest.log)" -ForegroundColor DarkGray

# 3. Launch Scoring Service with AI in Background
Write-Host "[3/4] Starting Scoring & AI Inference Service (Scenario: $Scenario)..." -ForegroundColor Green
$ScoringLog = Join-Path $LogDir "scoring.log"
$scoringProc = Start-Process -FilePath $PythonExe `
    -ArgumentList "-m", "twin.scoring.scoring_service", "--scenario", $Scenario, "--ai" `
    -RedirectStandardOutput $ScoringLog `
    -RedirectStandardError $ScoringLog `
    -PassThru

Write-Host "      Scoring PID: $($scoringProc.Id) (Logging to artifacts/logs/scoring.log)" -ForegroundColor DarkGray

# Allow services to subscribe to MQTT
Start-Sleep -Seconds 2

# Display Dashboards URLs
Write-Host ""
Write-Host "========================================================================" -ForegroundColor Magenta
Write-Host "  GRAFANA DASHBOARDS (Login: admin / admin_password_123):" -ForegroundColor Magenta
Write-Host "  - Line Overview   : http://localhost:3000/d/line3-overview" -ForegroundColor White
Write-Host "  - Station Detail  : http://localhost:3000/d/line3-station-detail" -ForegroundColor White
Write-Host "  - Parts & Risk    : http://localhost:3000/d/line3-parts-risk" -ForegroundColor White
Write-Host "  - AI Insights     : http://localhost:3000/d/line3-ai-insights" -ForegroundColor Yellow
Write-Host "========================================================================" -ForegroundColor Magenta
Write-Host ""

# 4. Run Replay Engine in Foreground
try {
    if (-not $NoReplay) {
        Write-Host "[4/4] Starting Replay for Scenario $Scenario..." -ForegroundColor Green
        $replayArgs = @("-m", "twin.replay.run", "--scenario", $Scenario, "--speed", $Speed)
        if ($MaxEvents -gt 0) {
            $replayArgs += @("--max-events", $MaxEvents)
        }
        & $PythonExe @replayArgs
    } else {
        Write-Host "[4/4] Background services running. Press Ctrl+C to terminate." -ForegroundColor Cyan
        while ($true) { Start-Sleep -Seconds 1 }
    }
}
finally {
    Write-Host ""
    Write-Host "Shutting down background services..." -ForegroundColor Yellow
    if ($ingestProc -and -not $ingestProc.HasExited) {
        Stop-Process -Id $ingestProc.Id -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped Ingestion Daemon." -ForegroundColor DarkGray
    }
    if ($scoringProc -and -not $scoringProc.HasExited) {
        Stop-Process -Id $scoringProc.Id -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped Scoring Daemon." -ForegroundColor DarkGray
    }
    Write-Host "Demo session ended cleanly." -ForegroundColor Green
}
