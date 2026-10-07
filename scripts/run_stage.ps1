param(
  [Parameter(Mandatory=$true)][ValidateSet("cp0","cp1","cp2","cp3","cp4","cp5","cp6","cp7","cp8")][string]$Stage,
  [Parameter(Mandatory=$true)][string]$Dataset,
  [string]$RunDir = "outputs/full",
  [string]$Config = "configs/full.yaml"
)
$Root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = "$Root\src;$env:PYTHONPATH"
python -m qtrc_stop.cli run --config "$Root\$Config" --dataset $Dataset --run-dir $RunDir --start-stage $Stage --end-stage $Stage
exit $LASTEXITCODE
