# 軍儀 デプロイ準備スクリプト
# 量子化モデルを作成してCloud Runにデプロイ

$ErrorActionPreference = "Stop"

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "軍儀 (Gungi) デプロイ準備" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# 仮想環境をアクティベート
Write-Host "🔧 仮想環境をアクティベート中..." -ForegroundColor Yellow
.\.venv\Scripts\Activate.ps1

# チェックポイントの存在確認
$latestModel = "checkpoints\latest.pt"
$quantizedModel = "checkpoints\model_quantized.pt"

if (-not (Test-Path $latestModel)) {
    Write-Host "❌ エラー: $latestModel が見つかりません" -ForegroundColor Red
    Write-Host "   学習を実行してからデプロイしてください" -ForegroundColor Red
    exit 1
}

# 量子化モデルの作成
Write-Host ""
Write-Host "📦 モデルを量子化中..." -ForegroundColor Green
python scripts/quantize_model.py

if (-not (Test-Path $quantizedModel)) {
    Write-Host "❌ エラー: 量子化に失敗しました" -ForegroundColor Red
    exit 1
}

# サイズ表示
$originalSize = (Get-Item $latestModel).Length / 1MB
$quantizedSize = (Get-Item $quantizedModel).Length / 1MB
Write-Host ""
Write-Host "✅ 量子化完了!" -ForegroundColor Green
Write-Host "   オリジナル: $([math]::Round($originalSize, 2)) MB"
Write-Host "   量子化版:   $([math]::Round($quantizedSize, 2)) MB"
Write-Host "   削減率:     $([math]::Round((1 - $quantizedSize/$originalSize) * 100, 1))%"

# Cloud Runへデプロイ
Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Cloud Runにデプロイしますか？" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
$deploy = Read-Host "(y/n)"

if ($deploy -eq 'y' -or $deploy -eq 'Y') {
    # 設定
    $PROJECT_ID = "gungi-game"
    $SERVICE_NAME = "gungi-game"
    $REGION = "asia-northeast1"
    $MEMORY = "2Gi"
    $CPU = "1"
    $MAX_INSTANCES = "10"
    $MIN_INSTANCES = "0"

    Write-Host ""
    Write-Host "🚀 Cloud Runにデプロイ中..." -ForegroundColor Green
    Write-Host "   プロジェクト: $PROJECT_ID"
    Write-Host "   サービス名: $SERVICE_NAME"
    Write-Host "   リージョン: $REGION"
    Write-Host ""

    # プロジェクトを設定
    gcloud config set project $PROJECT_ID

    # デプロイ
    gcloud run deploy $SERVICE_NAME `
      --source . `
      --region $REGION `
      --platform managed `
      --memory $MEMORY `
      --cpu $CPU `
      --max-instances $MAX_INSTANCES `
      --min-instances $MIN_INSTANCES `
      --allow-unauthenticated `
      --set-env-vars="PYTHONUNBUFFERED=1"

    Write-Host ""
    Write-Host "✅ デプロイ完了!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📱 アクセスURL:" -ForegroundColor Cyan
    Write-Host "   https://gungi-game-57998005741.asia-northeast1.run.app" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "デプロイをスキップしました" -ForegroundColor Yellow
    Write-Host "後でデプロイするには: .\deploy_with_model.ps1 を再実行してください"
}

Write-Host ""
Write-Host "完了!" -ForegroundColor Green
