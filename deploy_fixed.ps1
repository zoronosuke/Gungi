# 軍儀 GCP Cloud Run デプロイスクリプト (Windows PowerShell版)

$ErrorActionPreference = "Stop"

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "軍儀 (Gungi) をGCP Cloud Runにデプロイ" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# 設定
$PROJECT_ID = "gungi-game"
$SERVICE_NAME = "gungi-game"
$REGION = "asia-northeast1"
$MEMORY = "2Gi"
$CPU = "1"
$MAX_INSTANCES = "10"
$MIN_INSTANCES = "0"

# プロジェクトIDの確認
Write-Host "📋 使用するGCPプロジェクト: $PROJECT_ID" -ForegroundColor Yellow
$confirmation = Read-Host "このプロジェクトIDで正しいですか？ (y/n)"
if ($confirmation -ne 'y' -and $confirmation -ne 'Y') {
    Write-Host "deploy.ps1を編集してPROJECT_IDを変更してください" -ForegroundColor Red
    exit 1
}

# プロジェクトを設定
Write-Host ""
Write-Host "GCPプロジェクトを設定中..." -ForegroundColor Green
gcloud config set project $PROJECT_ID

# 必要なAPIを有効化
Write-Host ""
Write-Host "必要なAPIを有効化中..." -ForegroundColor Green
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# デプロイ
Write-Host ""
Write-Host "Cloud Runにデプロイ中..." -ForegroundColor Green
Write-Host "   サービス名: $SERVICE_NAME"
Write-Host "   リージョン: $REGION"
Write-Host "   メモリ: $MEMORY"
Write-Host "   CPU: $CPU"
Write-Host ""

gcloud run deploy $SERVICE_NAME `
  --source . `
  --region $REGION `
  --allow-unauthenticated `
  --memory $MEMORY `
  --cpu $CPU `
  --max-instances $MAX_INSTANCES `
  --min-instances $MIN_INSTANCES `
  --timeout 300 `
  --platform managed

Write-Host ""
Write-Host "デプロイが完了しました！" -ForegroundColor Green
Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "アプリケーションURL:" -ForegroundColor Yellow
$url = gcloud run services describe $SERVICE_NAME --region $REGION --format='value(status.url)'
Write-Host $url -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "ログを確認する場合:" -ForegroundColor Yellow
Write-Host "   gcloud run logs read $SERVICE_NAME --region $REGION --limit 50"
Write-Host ""
Write-Host "サービスを削除する場合:" -ForegroundColor Yellow
Write-Host "   gcloud run services delete $SERVICE_NAME --region $REGION"
Write-Host ""
