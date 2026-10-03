$ErrorActionPreference = 'Stop'
$pythonRuntime = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonRuntime)) { throw 'Create .venv and install requirements.txt first; see README.md.' }
Push-Location -LiteralPath $PSScriptRoot
try { & $pythonRuntime (Join-Path $PSScriptRoot 'app.py') @args }
finally { Pop-Location }
