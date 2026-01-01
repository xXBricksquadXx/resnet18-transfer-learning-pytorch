$ErrorActionPreference = "Stop"

# Prevent native stderr from becoming terminating errors (PowerShell 7+ behavior)
$PSNativeCommandUseErrorActionPreference = $false

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RepoRoot

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Force "logs" | Out-Null

$transcriptPath = Join-Path "logs" "transcript_$ts.txt"
Start-Transcript -Path $transcriptPath -Force

Write-Host "REPO_ROOT: $RepoRoot"
Write-Host "TIMESTAMP: $ts"

if (!(Test-Path ".\.venv\Scripts\Activate.ps1")) {
    throw "Missing venv. Create it with: python -m venv .venv"
}
.\.venv\Scripts\Activate.ps1

Write-Host "`n=== SANITY ==="
python -V
python -c "import torch, torchvision; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('cuda_available', torch.cuda.is_available())"

Write-Host "`n=== DATASET COUNTS (ImageFolder) ==="
python scripts\dataset_counts.py

$device = "cpu"
$epochs = 10
$batch = 16
$lr = 1e-3
$wd = 1e-4
$seed = 1337
$workers = 0
$amp = 0

Write-Host "`n=== TRAIN ==="
$trainLogPath = Join-Path "logs" "train_$ts.txt"

python train.py `
    --data-dir data `
    --device $device `
    --epochs $epochs `
    --batch-size $batch `
    --lr $lr `
    --weight-decay $wd `
    --freeze-backbone 1 `
    --unfreeze-last 0 `
    --unfreeze-all 0 `
    --seed $seed `
    --num-workers $workers `
    --amp $amp `
    --print-model 0 2>&1 | Tee-Object -FilePath $trainLogPath

$runDirLine = Select-String -Path $trainLogPath -Pattern '^RUN_DIR:' | Select-Object -Last 1
if (-not $runDirLine) { throw "Could not find RUN_DIR in $trainLogPath" }

$runDir = $runDirLine.Line.Replace("RUN_DIR:", "").Trim()
Write-Host "PARSED_RUN_DIR: $runDir"

$bestCkpt = Join-Path $runDir "best_acc.pt"
if (!(Test-Path $bestCkpt)) { throw "Missing best checkpoint: $bestCkpt" }

Write-Host "`n=== CHECKPOINT METADATA ==="
python -c "import torch; ckpt=torch.load(r'$bestCkpt', map_location='cpu'); print('best_epoch', ckpt.get('best_epoch')); print('best_val_acc', ckpt.get('best_val_acc')); print('best_val_loss', ckpt.get('best_val_loss')); print('class_names', ckpt.get('class_names')); print('weights', ckpt.get('weights'))"

Write-Host "`n=== PREDICT (folder random sample from val) ==="
python predict.py --image "data\val" --checkpoint $bestCkpt --device $device --topk 2 --random 1 --n 5

Write-Host "`n=== LIST RUN OUTPUTS ==="
Get-ChildItem -Path $runDir | Format-Table -AutoSize

Stop-Transcript
Write-Host "TRANSCRIPT_SAVED: $transcriptPath"
Write-Host "TRAIN_LOG_SAVED: $trainLogPath"
