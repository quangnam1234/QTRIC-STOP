param(
  [Parameter(Mandatory=$true)][string]$Dataset,
  [string]$RunDir = "outputs/full"
)
$Root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = "$Root\src;$env:PYTHONPATH"
python -m qtrc_stop.cli run --config "$Root\configs\full.yaml" --dataset $Dataset --run-dir $RunDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
