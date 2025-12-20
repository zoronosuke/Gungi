# Gungi AI 本番学習スクリプト
# 千日手対策済み（93チャンネル履歴入力）
# 外部PowerShellで実行: .\train_with_history.ps1

$ErrorActionPreference = "Stop"

# 作業ディレクトリに移動
Set-Location "C:\Users\iamzo\Documents\GUNGI\Gungi"

# 仮想環境をアクティベート
& ".\.venv\Scripts\Activate.ps1"

# チェックポイントをクリア（新しいモデルで開始）
Write-Host "=== Clearing old checkpoints ===" -ForegroundColor Yellow
$checkpointDir = ".\checkpoints"
if (Test-Path $checkpointDir) {
    Remove-Item "$checkpointDir\*.pt" -Force -ErrorAction SilentlyContinue
    Remove-Item "$checkpointDir\*.npz" -Force -ErrorAction SilentlyContinue
    Remove-Item "$checkpointDir\*.json" -Force -ErrorAction SilentlyContinue
}
Write-Host "Checkpoints cleared." -ForegroundColor Green

# 学習開始
Write-Host ""
Write-Host "=== Starting Training with History Channels (93ch) ===" -ForegroundColor Cyan
Write-Host "千日手対策: 局面履歴チャンネル追加済み" -ForegroundColor Cyan
Write-Host ""

# 本番学習パラメータ
# - iterations: 50 (十分な学習回数)
# - games: 64 (1イテレーションあたりの対局数)
# - mcts-sims: 200 (探索の深さ)
# - parallel: 32 (並列ゲーム数、GPUメモリに応じて調整)
# - batch: 128 (学習バッチサイズ)
# - epochs: 4 (1イテレーションの学習エポック数)

python scripts/train.py `
    --iterations 50 `
    --games 64 `
    --mcts-sims 200 `
    --parallel 32 `
    --batch 128 `
    --epochs 4

Write-Host ""
Write-Host "=== Training Complete ===" -ForegroundColor Green
Write-Host "次のステップ: python scripts/diagnose_repetition.py --games 20" -ForegroundColor Yellow
