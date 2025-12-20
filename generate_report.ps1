# Gungi AI 評価レポート生成スクリプト
# 学習完了後に実行してレポート用データを収集
# 外部PowerShellで実行: .\generate_report.ps1

$ErrorActionPreference = "Continue"

# 作業ディレクトリに移動
Set-Location "C:\Users\iamzo\Documents\GUNGI\Gungi"

# 仮想環境をアクティベート
& ".\.venv\Scripts\Activate.ps1"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$reportDir = ".\reports\$timestamp"

Write-Host "=== Gungi AI Evaluation Report Generator ===" -ForegroundColor Cyan
Write-Host "Report directory: $reportDir" -ForegroundColor Yellow
Write-Host ""

# レポートディレクトリ作成
New-Item -ItemType Directory -Path $reportDir -Force | Out-Null

# 1. 千日手診断
Write-Host "[1/4] Running repetition diagnosis..." -ForegroundColor Green
python scripts/diagnose_repetition.py --games 30 --verbose
if (Test-Path ".\repetition_diagnosis.json") {
    Copy-Item ".\repetition_diagnosis.json" "$reportDir\"
}
if (Test-Path ".\repetition_diagnosis.png") {
    Copy-Item ".\repetition_diagnosis.png" "$reportDir\"
}

# 2. モデル評価（vs Random, Greedy, SimpleMCTS）
Write-Host ""
Write-Host "[2/4] Running model evaluation vs baseline AIs..." -ForegroundColor Green
python scripts/evaluate_model.py --games 20 --mcts-sims 100
if (Test-Path ".\evaluation_results.json") {
    Copy-Item ".\evaluation_results.json" "$reportDir\"
}

# 3. 学習曲線の可視化
Write-Host ""
Write-Host "[3/4] Generating training curves..." -ForegroundColor Green
python scripts/plot_training.py --all
if (Test-Path ".\training_report.png") {
    Copy-Item ".\training_report.png" "$reportDir\"
}
if (Test-Path ".\training_curves.png") {
    Copy-Item ".\training_curves.png" "$reportDir\"
}

# 4. Elo計算（時間がかかる場合はスキップ可能）
Write-Host ""
Write-Host "[4/4] Calculating Elo ratings..." -ForegroundColor Green
python scripts/calculate_elo.py --games 6 --mcts-sims 50
if (Test-Path ".\elo_results.json") {
    Copy-Item ".\elo_results.json" "$reportDir\"
}

# 5. 学習状態をコピー
if (Test-Path ".\checkpoints\training_state.json") {
    Copy-Item ".\checkpoints\training_state.json" "$reportDir\"
}

# サマリーを生成
Write-Host ""
Write-Host "=== Generating Summary ===" -ForegroundColor Cyan

$summary = @"
# Gungi AI Training Report
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

## Files in this report:
- repetition_diagnosis.json: 千日手診断結果
- repetition_diagnosis.png: 千日手診断の可視化
- evaluation_results.json: 対ベースラインAI評価結果
- training_report.png: 学習曲線
- training_curves.png: 詳細Loss曲線
- elo_results.json: Elo rating推移
- training_state.json: 学習状態

## Key Metrics:
"@

# 学習状態から指標を抽出
if (Test-Path "$reportDir\training_state.json") {
    $state = Get-Content "$reportDir\training_state.json" | ConvertFrom-Json
    $summary += "`n- Total iterations: $($state.iteration)"
    $summary += "`n- Total games: $($state.total_games)"
    $summary += "`n- Total examples: $($state.total_examples)"
    if ($state.policy_loss_history.Count -gt 0) {
        $lastPolicyLoss = $state.policy_loss_history[-1]
        $summary += "`n- Final Policy Loss: $([math]::Round($lastPolicyLoss, 4))"
    }
    if ($state.value_loss_history.Count -gt 0) {
        $lastValueLoss = $state.value_loss_history[-1]
        $summary += "`n- Final Value Loss: $([math]::Round($lastValueLoss, 4))"
    }
}

# 千日手診断から指標を抽出
if (Test-Path "$reportDir\repetition_diagnosis.json") {
    $diag = Get-Content "$reportDir\repetition_diagnosis.json" | ConvertFrom-Json
    $summary += "`n- Repetition Rate: $([math]::Round($diag.repetition_rate * 100, 1))%"
}

$summary | Out-File "$reportDir\SUMMARY.md" -Encoding UTF8

Write-Host ""
Write-Host "=== Report Generation Complete ===" -ForegroundColor Green
Write-Host "Report saved to: $reportDir" -ForegroundColor Yellow
Write-Host ""
Write-Host "Files generated:" -ForegroundColor Cyan
Get-ChildItem $reportDir | ForEach-Object { Write-Host "  - $($_.Name)" }
