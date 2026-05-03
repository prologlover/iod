# Launch the Byzantine Detection Research Dashboard
$env:PYTHONIOENCODING = "utf-8"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
Write-Host "Starting dashboard at http://localhost:8501" -ForegroundColor Cyan
.\.venv\Scripts\streamlit.exe run dashboard/app.py --server.port 8501
