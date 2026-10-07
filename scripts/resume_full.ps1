param(
  [Parameter(Mandatory=$true)][string]$Dataset,
  [string]$RunDir = "outputs/full"
)
& "$PSScriptRoot\run_full.ps1" -Dataset $Dataset -RunDir $RunDir
exit $LASTEXITCODE
