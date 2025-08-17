@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Python venv oluştur ve bağımlılıkları yükle
if not exist .venv (
  py -3 -m venv .venv
)

set PYTHON=.venv\Scripts\python.exe
set PIP=.venv\Scripts\pip.exe

"%PIP%" install --upgrade pip
"%PIP%" install -r requirements.txt pyinstaller

REM PyInstaller ile GUI uygulamasını derle (onedir, windowed)
set NAME=VMDOptimizer
set OPTS=--noconfirm --clean --name %NAME% --onedir --windowed

REM İkon ayarla (gereken her yerde ./icon.ico kullanılacak)
set ICON=icon.ico
if not "%ICON%"=="" (
  if exist %ICON% (
    set OPTS=%OPTS% --icon "%ICON%"
  )
)

REM Ek veri dosyaları
set DATA_LOGO=
if exist logo.png (
  set DATA_LOGO=--add-data "logo.png;."
)
set DATA_VER=
if exist version.txt (
  set DATA_VER=--add-data "version.txt;."
)
set DATA_ICON=
if exist icon.ico (
  set DATA_ICON=--add-data "icon.ico;."
)

"%PYTHON%" -m PyInstaller %OPTS% --add-data "scripts;scripts" %DATA_LOGO% %DATA_VER% %DATA_ICON% scripts\app.py

echo.
echo Derleme tamamlandi. dist\%NAME% klasorunu portable olarak calistirabilirsiniz.
endlocal
