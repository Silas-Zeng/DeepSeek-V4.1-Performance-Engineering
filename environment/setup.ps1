$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    python -m venv (Join-Path $repoRoot ".venv")
}

& $venvPython -m pip install -r (Join-Path $repoRoot "requirements-analysis.txt")
& $venvPython (Join-Path $repoRoot "scripts\collect_environment.py") `
    --output (Join-Path $repoRoot "reports\environment.json")
