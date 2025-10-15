#!/bin/bash
# 軍儀 GCP Cloud Run デプロイスクリプト

set -e  # エラーが発生したら停止

echo "======================================"
echo "軍儀 (Gungi) をGCP Cloud Runにデプロイ"
echo "======================================"
echo ""

# 設定
PROJECT_ID="gungi-game"  # 必要に応じて変更
SERVICE_NAME="gungi-game"
REGION="asia-northeast1"  # 東京リージョン
MEMORY="2Gi"  # 2GB RAM（PyTorch用）
CPU="1"
MAX_INSTANCES="10"
MIN_INSTANCES="0"  # コスト削減のため0（必要に応じて1に変更）

# プロジェクトIDの確認
echo "📋 使用するGCPプロジェクト: $PROJECT_ID"
read -p "このプロジェクトIDで正しいですか？ (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "deploy.shを編集してPROJECT_IDを変更してください"
    exit 1
fi

# プロジェクトを設定
echo ""
echo "🔧 GCPプロジェクトを設定中..."
gcloud config set project $PROJECT_ID

# 必要なAPIを有効化
echo ""
echo "🔌 必要なAPIを有効化中..."
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# デプロイ
echo ""
echo "🚀 Cloud Runにデプロイ中..."
echo "   サービス名: $SERVICE_NAME"
echo "   リージョン: $REGION"
echo "   メモリ: $MEMORY"
echo "   CPU: $CPU"
echo ""

gcloud run deploy $SERVICE_NAME \
  --source . \
  --region $REGION \
  --allow-unauthenticated \
  --memory $MEMORY \
  --cpu $CPU \
  --max-instances $MAX_INSTANCES \
  --min-instances $MIN_INSTANCES \
  --timeout 300 \
  --platform managed

echo ""
echo "✅ デプロイが完了しました！"
echo ""
echo "======================================"
echo "🎮 アプリケーションURL:"
gcloud run services describe $SERVICE_NAME --region $REGION --format='value(status.url)'
echo "======================================"
echo ""
echo "💡 ログを確認する場合:"
echo "   gcloud run logs read $SERVICE_NAME --region $REGION --limit 50"
echo ""
echo "💡 サービスを削除する場合:"
echo "   gcloud run services delete $SERVICE_NAME --region $REGION"
echo ""
