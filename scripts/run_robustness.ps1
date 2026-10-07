param(
  [Parameter(Mandatory=$true)][string]$Dataset,
  [string]$RunDir = "outputs/full",
  [string]$BaseConfig = "configs/full.yaml"
)
$Root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = "$Root\src;$env:PYTHONPATH"
$TmpConfig = Join-Path $RunDir "robustness_enabled.yaml"
python -c "import yaml,sys,pathlib; p=pathlib.Path(sys.argv[1]); q=pathlib.Path(sys.argv[2]); c=yaml.safe_load(p.read_text(encoding='utf-8')); c.setdefault('robustness',{})['enabled']=True; q.parent.mkdir(parents=True,exist_ok=True); q.write_text(yaml.safe_dump(c,allow_unicode=True,sort_keys=False),encoding='utf-8')" "$Root\$BaseConfig" $TmpConfig
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m qtrc_stop.cli run --config $TmpConfig --dataset $Dataset --run-dir $RunDir --start-stage cp7 --end-stage cp7
exit $LASTEXITCODE
