# File: scripts/build_publish_win.ps1
$ErrorActionPreference = "Stop"

# 1) версия (SemVer), должна совпадать с релизом Velopack
$VERSION = "1.0.0"

# 2) чистим
Remove-Item -Recurse -Force .\build,\dist -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .\publish -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force .\publish | Out-Null

# 3) pyinstaller -> publish\
# Важно: имя exe (SolarainProto.exe) потом укажем в vpk --mainExe
pyinstaller `
  --noconsole `
  --onedir app_entry.py `
  --name SolarainProto `
  --distpath .\publish `
  --clean
