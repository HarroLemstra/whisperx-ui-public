$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

$pythonw = Join-Path $PSScriptRoot ".venv\Scripts\pythonw.exe"
$appHome = Join-Path $env:LOCALAPPDATA "WhisperXNightQueue"
$statusPath = Join-Path $appHome "data\tray_status.json"
$trayLogPath = Join-Path $appHome "logs\tray.log"
$trayTaskName = "WhisperXNightQueueTray"
$powershellExe = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"

if (-not (Test-Path $pythonw)) {
    throw "Missing venv pythonw at $pythonw. Ensure .venv is created."
}

New-Item -ItemType Directory -Path (Join-Path $appHome "data") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $appHome "logs") -Force | Out-Null

if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        $line = $_.Trim()
        if ($line -and (-not $line.StartsWith("#")) -and $line.Contains("=")) {
            $parts = $line.Split("=", 2)
            $key = $parts[0].Trim()
            $value = $parts[1].Trim()
            if ($value.StartsWith('"') -and $value.EndsWith('"')) {
                $value = $value.Substring(1, $value.Length - 2)
            } elseif ($value.StartsWith("'") -and $value.EndsWith("'")) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            if ($key) {
                Set-Item -Path "Env:$key" -Value $value
            }
        }
    }
}

if (-not $env:HF_TOKEN -and $env:HF_token) {
    $env:HF_TOKEN = $env:HF_token
}

function Ensure-TrayAutostart {
    $taskCommand = "`"$powershellExe`" -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$PSScriptRoot\start_tray.ps1`""
    $null = & schtasks /Create /F /SC ONLOGON /RL LIMITED /TN $trayTaskName /TR $taskCommand 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to register scheduled task '$trayTaskName'."
    }
}

function Ensure-StartupShortcut {
    $startupDir = [Environment]::GetFolderPath("Startup")
    $shortcutPath = Join-Path $startupDir "WhisperX Night Queue.cmd"
    $cmdContent = "@echo off`r`n`"$powershellExe`" -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$PSScriptRoot\start_tray.ps1`"`r`n"
    Set-Content -Path $shortcutPath -Value $cmdContent -Encoding ASCII
}

function Get-UrlFromTrayLog {
    if (-not (Test-Path $trayLogPath)) {
        return $null
    }
    try {
        $line = Get-Content $trayLogPath -Tail 200 | Select-String -Pattern "Tray running at (http://[^\s]+)" | Select-Object -Last 1
        if ($line -and $line.Matches.Count -gt 0) {
            return $line.Matches[0].Groups[1].Value
        }
    } catch {}
    return $null
}

function Get-TrayStatus {
    if (-not (Test-Path $statusPath)) {
        return $null
    }
    try {
        return (Get-Content $statusPath -Raw | ConvertFrom-Json)
    } catch {
        return $null
    }
}

function Test-TrayEndpoint {
    param(
        [Parameter(Mandatory = $true)][string]$Url
    )
    try {
        $null = Invoke-WebRequest -Uri $Url -UseBasicParsing -Method Get -TimeoutSec 3
        return $true
    } catch {
        return $false
    }
}

function Find-RunningUrl {
    for ($p = 7860; $p -le 7880; $p++) {
        $candidate = "http://127.0.0.1:$p"
        if (Test-TrayEndpoint -Url $candidate) {
            return $candidate
        }
    }
    return $null
}

try {
    Ensure-TrayAutostart
} catch {
    try {
        Ensure-StartupShortcut
        Write-Host "Autostart via Startup-map ingesteld (fallback)."
    } catch {
        Write-Host "Waarschuwing: autostart kon niet worden ingesteld."
    }
}

$existing = Get-TrayStatus
if ($existing -and $existing.state -eq "running" -and $existing.url) {
    $existingPid = 0
    try { $existingPid = [int]$existing.pid } catch {}
    $processAlive = $false
    if ($existingPid -gt 0) {
        $processAlive = [bool](Get-Process -Id $existingPid -ErrorAction SilentlyContinue)
    }
    if ($processAlive -and (Test-TrayEndpoint -Url $existing.url)) {
        Write-Host "Tray app draait al op $($existing.url)"
        exit 0
    } else {
        Write-Host "Stale tray status gevonden; probeer nieuwe start..."
    }
}

$process = Start-Process -FilePath $pythonw -ArgumentList ".\tray_app.py" -WorkingDirectory $PSScriptRoot -PassThru
$deadline = (Get-Date).AddSeconds(20)
$ready = $false
$readyUrl = $null

while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 500
    if ($process.HasExited) {
        break
    }

    $status = Get-TrayStatus
    if (-not $status) {
        continue
    }
    if ($status.state -ne "running" -or -not $status.url) {
        continue
    }

    $statusPid = 0
    try { $statusPid = [int]$status.pid } catch {}
    $pidMatch = ($statusPid -eq $process.Id)
    $statusProcAlive = $false
    if ($statusPid -gt 0) {
        $statusProcAlive = [bool](Get-Process -Id $statusPid -ErrorAction SilentlyContinue)
    }

    if (($pidMatch -or $statusProcAlive) -and (Test-TrayEndpoint -Url $status.url)) {
        $ready = $true
        $readyUrl = [string]$status.url
        break
    }
}

if ($ready) {
    Write-Host "Tray running at $readyUrl"
    exit 0
}

if ($process.HasExited -and $process.ExitCode -eq 0) {
    $fallbackUrl = Get-UrlFromTrayLog
    if (-not $fallbackUrl) {
        $fallbackUrl = Find-RunningUrl
    }
    if ($fallbackUrl -and (Test-TrayEndpoint -Url $fallbackUrl)) {
        Write-Host "Tray app draait al op $fallbackUrl"
        exit 0
    }
}

Write-Host "Tray startup failed."
if ($process.HasExited) {
    Write-Host "Tray process exited with code $($process.ExitCode)."
} else {
    Write-Host "Tray process leeft nog, maar healthcheck is niet geslaagd."
}
Write-Host "Bekijk log: $trayLogPath"
exit 1
