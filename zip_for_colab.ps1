# Zip the project for Google Colab upload
# Excludes: .venv, __pycache__, data/raw (dataset is downloaded on Colab), outputs

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ZipPath   = "$ScriptDir\..\iod_colab.zip"

Write-Host "Creating Colab-ready zip..." -ForegroundColor Cyan

# Use 7-Zip if available, otherwise fall back to Compress-Archive
$7z = Get-Command "7z" -ErrorAction SilentlyContinue

if ($7z) {
    & 7z a -tzip $ZipPath "$ScriptDir\*" `
        -xr!".venv" -xr!"__pycache__" -xr!"*.pyc" `
        -xr!"data\raw" -xr!"outputs\models\*.pt" `
        -xr!"dashboard\.streamlit" `
        -xr!".git"
} else {
    # PowerShell fallback
    $exclude = @('.venv','__pycache__','.git','data\raw')
    $files = Get-ChildItem -Path $ScriptDir -Recurse -File | Where-Object {
        $rel = $_.FullName.Replace("$ScriptDir\","")
        -not ($exclude | ForEach-Object { $rel.StartsWith($_) } | Where-Object { $_ })
    }
    Compress-Archive -Path $files.FullName -DestinationPath $ZipPath -Force
}

$size = [math]::Round((Get-Item $ZipPath).Length / 1MB, 1)
Write-Host "Done: $ZipPath  ($size MB)" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Upload iod_colab.zip to Google Colab (Method B) or Google Drive (Method A)"
Write-Host "  2. Open Byzantine_Detection_Colab.ipynb in Colab"
Write-Host "  3. Runtime -> Change runtime type -> GPU (T4)"
Write-Host "  4. Run cells top to bottom"
