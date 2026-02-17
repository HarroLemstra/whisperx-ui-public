$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Missing venv python at $python. Create/activate .venv first."
}

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

Write-Host "Starting WhisperX Night Queue UI on http://127.0.0.1:7860"
& $python ".\app.py"
