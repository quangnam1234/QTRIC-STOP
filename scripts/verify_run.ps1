param([string]$RunDir = "outputs/full")
$Root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = "$Root\src;$env:PYTHONPATH"
python -m qtrc_stop.cli verify --run-dir $RunDir
exit $LASTEXITCODE
