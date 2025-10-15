# 軍儀 (Gungi) - GCP Cloud Run用 Dockerfile

# Python 3.11ベースイメージ (3.12の互換性問題を回避)
FROM python:3.11-slim

# 作業ディレクトリを設定
WORKDIR /app

# システムパッケージの更新と必要なツールのインストール
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Pythonの依存関係ファイルをコピー
COPY requirements.txt .

# pipを最新にアップグレードし、ビルドツールをインストール
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 依存関係をインストール
# PyTorchはCPU版を使用(軽量化)
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# 環境変数を設定
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# ポートを公開
EXPOSE 8080

# FastAPIサーバーを起動
# Cloud Runは環境変数PORTを提供するので、それを使用
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}
