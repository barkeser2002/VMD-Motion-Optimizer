# Requires: pyinstaller, PyQt6
# Kullanım: PowerShell'de bu dosyayla aynı klasörde:
#   powershell -ExecutionPolicy Bypass -File build_exe.ps1

$ErrorActionPreference = 'Stop'

# Sanal ortam oluştur (taşınabilir derleme için önerilir)
if (-not (Test-Path .venv)) {
  py -3 -m venv .venv
}

$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
.venv\Scripts\pip.exe install --upgrade pip
.venv\Scripts\pip.exe install -r requirements.txt pyinstaller

# Tek dosya, konsolsuz, portable dizinle (one dir daha uyumlu)
$icon = "icon.ico" # .ico yolu
$common = "--noconfirm --clean --name VMDOptimizer"
$opts = "--onedir --windowed"
if (Test-Path $icon) { $opts = "$opts --icon `"$icon`"" }

# Ek veriler
$extra = "--add-data `"scripts;scripts`" --add-data `"version.txt;.`""
if (Test-Path "logo.png") { $extra = "$extra --add-data `"logo.png;.`"" }
if (Test-Path "icon.ico") { $extra = "$extra --add-data `"icon.ico;.`"" }

# Giriş: GUI uygulaması
& $python -m PyInstaller $common $opts $extra scripts\app.py

Write-Host "Derleme tamamlandı. dist/VMDOptimizer klasörünü taşıyıp portable çalıştırabilirsiniz."
