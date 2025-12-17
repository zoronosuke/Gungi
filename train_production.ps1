# Gungi AI - Production Training Script
# Run this in PowerShell (outside VS Code)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Gungi AI - Production Training" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Settings:" -ForegroundColor Yellow
Write-Host "  - Model: 8 res_blocks (full)"
Write-Host "  - MCTS simulations: 200"
Write-Host "  - Games per iteration: 50"
Write-Host "  - Parallel games: 64"
Write-Host "  - Batch size: 512"
Write-Host "  - FP16 (half precision): enabled"
Write-Host "  - Iterations: 100"
Write-Host ""
Write-Host "Penalty Settings:" -ForegroundColor Yellow
Write-Host "  - Repetition (senjite) penalty: -0.9"
Write-Host "  - Max moves penalty: -0.1"
Write-Host "  - Repetition threshold: 3"
Write-Host "  - Max moves: 200"
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan

Set-Location $PSScriptRoot
& .\.venv\Scripts\Activate.ps1

python scripts/train.py --full --optimized --iterations 100 --games 50 --mcts-sims 200 --batch-size 512 --use-fp16

Write-Host ""
Write-Host "Training completed!" -ForegroundColor Green
