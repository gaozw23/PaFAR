$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
& '.\.venv\Scripts\python.exe' 'scripts\run_realdata_pipeline.py' `
  --config 'configs\realdata_primary.yaml' `
  --stage all `
  --n-jobs 4 `
  --resume `
  --confirm RUN_PAFAR_REALDATA_PRIMARY
exit $LASTEXITCODE

