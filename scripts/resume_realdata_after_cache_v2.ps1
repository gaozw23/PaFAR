$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
& '.\.venv\Scripts\python.exe' scripts\run_realdata_pipeline.py `
  --config configs\realdata_primary.yaml `
  --stage analysis-after-cache `
  --n-jobs 4 `
  --resume `
  --confirm RUN_PAFAR_REALDATA_AFTER_CACHE_V2_1
exit $LASTEXITCODE
