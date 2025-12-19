# Gungi AI - 深層強化学習トレーニング（修正版）
# MCTS 200回で本格的な学習を実行

Set-Location "C:\Users\iamzo\Documents\GUNGI\Gungi"

Write-Host "`n====================================================" -ForegroundColor Cyan
Write-Host "  Gungi AI - 深層強化学習トレーニング" -ForegroundColor Green
Write-Host "  修正版: 引き分けバグ修正済み" -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "  MCTS シミュレーション: 200回" -ForegroundColor White
Write-Host "  ゲーム数/イテレーション: 10" -ForegroundColor White
Write-Host "  総イテレーション数: 20" -ForegroundColor White
Write-Host "`n  修正内容:" -ForegroundColor Yellow
Write-Host "    1. 引き分け報酬バグ修正 (DRAW_VALUE -> DRAW_VALUE_MAX_MOVES)" -ForegroundColor White
Write-Host "    2. MCTS内の引き分け評価を0.0から-0.1に変更" -ForegroundColor White
Write-Host "    3. MAX_MOVES設定を200に統一" -ForegroundColor White
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""

# 仮想環境をアクティベート
& .\.venv\Scripts\Activate.ps1

# 学習開始
python scripts\train.py --mcts-sims 200 --games 10 --iterations 20 --batch-size 256

Write-Host "`n学習が完了しました！" -ForegroundColor Green
Read-Host "Enterキーを押して終了"
