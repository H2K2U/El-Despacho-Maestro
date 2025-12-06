# File: scripts/vpk_pack_win.ps1
$ErrorActionPreference = "Stop"

$PACK_ID = "Solarain.SolarainProto"
$VERSION = "1.0.0"

# publish\SolarainProto\SolarainProto.exe (после pyinstaller --onedir)
vpk pack --packId $PACK_ID --packVersion $VERSION --packDir .\publish\SolarainProto --mainExe SolarainProto.exe
