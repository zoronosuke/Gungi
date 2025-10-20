# 軍儀 (Gungi) GCP Cloud Run デプロイガイド

このガイドに従って、軍儀ゲームをGoogle Cloud Platform (GCP) のCloud Runにデプロイします。

## 📋 前提条件

- Googleアカウント
- GCPプロジェクト（無料枠で十分）
- Google Cloud SDK（インストール手順は下記）

---

## 🚀 デプロイ手順（所要時間: 30分）

### Step 1: Google Cloud SDKのインストール

#### Windowsの場合：

1. **PowerShellを管理者権限で開く**

2. **インストーラーをダウンロード＆実行**
```powershell
# インストーラーをダウンロード
Invoke-WebRequest -Uri "https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe" -OutFile "$env:TEMP\GoogleCloudSDKInstaller.exe"

# インストーラーを実行
Start-Process -FilePath "$env:TEMP\GoogleCloudSDKInstaller.exe" -Wait
```

3. **インストール完了後、新しいPowerShellウィンドウを開く**

4. **インストール確認**
```powershell
gcloud --version
```

期待される出力:
```
Google Cloud SDK 457.0.0
...
```

---

### Step 2: GCP初期設定

1. **GCPにログイン**
```powershell
gcloud init
```

このコマンドで以下が行われます：
- ブラウザが開き、Googleアカウントでログイン
- 使用するGCPプロジェクトを選択（または新規作成）
- デフォルトリージョンの設定

2. **対話形式の質問に答える**

```
Pick configuration to use:
→ [1] Re-initialize this configuration
  または [2] Create a new configuration

Choose the account you would like to use:
→ あなたのGoogleアカウントを選択

Pick cloud project to use:
→ [Create a new project] を選択
  プロジェクト名: gungi-game

Do you want to configure a default Compute Region and Zone? (Y/n)?
→ Y
→ asia-northeast1 (Tokyo) を選択
```

---

### Step 3: 請求先アカウントの設定（重要）

GCP無料枠を使うには、請求先アカウントの登録が必要です（クレジットカード情報）。
無料枠内であれば**課金されません**。

1. [GCP Console](https://console.cloud.google.com/) を開く
2. 「お支払い」→「請求先アカウントをリンク」
3. クレジットカード情報を入力（無料枠確認のため）

💡 **無料枠の範囲:**
- Cloud Run: 月間200万リクエスト無料
- CPU時間: 180,000 vCPU秒/月
- メモリ: 360,000 GiB秒/月
- ネットワーク: 1GB/月の下り転送

**軍儀の想定使用量:** 月100ゲーム程度なら完全に無料枠内！

---

### Step 4: プロジェクトIDの確認と設定

1. **現在のプロジェクトIDを確認**
```powershell
gcloud config get-value project
```

2. **プロジェクトIDをメモ**（例: `gungi-game-123456`）

3. **`deploy.ps1` を編集**
```powershell
notepad deploy.ps1
```

以下の行を変更：
```powershell
$PROJECT_ID = "gungi-game"  # ← あなたのプロジェクトIDに変更
```

---

### Step 5: デプロイ実行！

1. **プロジェクトディレクトリに移動**
```powershell
cd c:\Users\okada\Documents\2025\memo\Gungi\Gungi
```

2. **デプロイスクリプトを実行**
```powershell
.\deploy.ps1
```

3. **確認プロンプトで `y` を入力**

4. **デプロイ開始！**（5-10分かかります）

デプロイ中の出力例：
```
🔧 GCPプロジェクトを設定中...
🔌 必要なAPIを有効化中...
🚀 Cloud Runにデプロイ中...
Building using Dockerfile...
✓ Creating Container Repository...
✓ Uploading sources...
✓ Building image...
✓ Pushing to Container Registry...
✓ Deploying to Cloud Run...
✅ デプロイが完了しました！

======================================
🎮 アプリケーションURL:
https://gungi-game-2urmfnxx4q-an.a.run.app
======================================
```

5. **URLをブラウザで開く**

🎉 **完成！ゲームをプレイできます！**

---

## 🎮 デプロイ後の操作

### ログの確認
```powershell
gcloud run logs read gungi-game --region asia-northeast1 --limit 50
```

### リアルタイムログ監視
```powershell
gcloud run logs tail gungi-game --region asia-northeast1
```

### サービス情報の確認
```powershell
gcloud run services describe gungi-game --region asia-northeast1
```

### サービスの削除（必要な場合）
```powershell
gcloud run services delete gungi-game --region asia-northeast1
```

---

## 💰 コスト管理

### 使用状況の確認
```powershell
# Cloud Runの使用状況
gcloud run services describe gungi-game --region asia-northeast1 --format="value(status.traffic[0].latestRevision)"

# 請求情報の確認
# GCP Consoleの「お支払い」→「レポート」で確認
```

### コスト削減のヒント

1. **最小インスタンス数を0に**（既に設定済み）
   - アクセスがない時はコストゼロ
   - 初回アクセス時に5-10秒の起動時間

2. **最小インスタンス数を1に変更（常時起動）**
   ```powershell
   # deploy.ps1の以下の行を変更
   $MIN_INSTANCES = "1"  # 0から1に変更
   ```
   - コスト: 約$5-7/月
   - メリット: 常に高速レスポンス

3. **メモリを削減（AIを使わない場合）**
   ```powershell
   $MEMORY = "1Gi"  # 2Giから1Giに変更
   ```

---

## 🔧 トラブルシューティング

### エラー1: `gcloud: command not found`
**原因:** Google Cloud SDKがインストールされていない

**解決策:**
1. PowerShellを再起動
2. それでもダメなら、SDKを再インストール

---

### エラー2: `Permission denied`
**原因:** 必要な権限がない

**解決策:**
```powershell
# プロジェクトのオーナー権限を確認
gcloud projects get-iam-policy $(gcloud config get-value project)

# 必要に応じて権限を追加（GCP Consoleで）
```

---

### エラー3: `Billing account is required`
**原因:** 請求先アカウントが設定されていない

**解決策:**
1. [GCP Console](https://console.cloud.google.com/billing) を開く
2. 請求先アカウントをプロジェクトにリンク

---

### エラー4: デプロイは成功したがアクセスできない
**原因:** ファイアウォールまたはCORS設定

**解決策:**
```powershell
# Cloud Runサービスのログを確認
gcloud run logs read gungi-game --region asia-northeast1 --limit 100

# エラーメッセージから原因を特定
```

---

### エラー5: メモリ不足でクラッシュ
**原因:** PyTorchがメモリを使いすぎ

**解決策:**
```powershell
# メモリを4Giに増やす
# deploy.ps1を編集
$MEMORY = "4Gi"

# 再デプロイ
.\deploy.ps1
```

---

## 📊 パフォーマンス最適化

### 1. コールドスタート対策

**問題:** 最初のアクセス時に5-10秒かかる

**解決策A:** 最小インスタンス数を1に
```powershell
$MIN_INSTANCES = "1"
```

**解決策B:** Cloud Schedulerで定期的にウォームアップ
```powershell
# 5分ごとにアクセスしてインスタンスを起動状態に保つ
gcloud scheduler jobs create http gungi-warmup \
  --schedule="*/5 * * * *" \
  --uri="https://your-service-url.run.app/health" \
  --http-method=GET
```

### 2. PyTorchの最適化

**問題:** 推論が遅い

**解決策:** ONNX変換（将来的に実装）

---

## 🔐 セキュリティ

### CORS設定の制限（本番環境推奨）

現在はすべてのオリジンを許可していますが、本番環境では制限すべきです。

`src/api/main.py` を編集：
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-domain.com",
        "https://gungi-game-xxxxx.run.app"
    ],  # 特定のドメインのみ許可
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
```

---

## 🌐 カスタムドメインの設定（オプション）

独自ドメインを使いたい場合：

1. **ドメインを取得**（Google Domains、Cloudflareなど）

2. **Cloud Runにドメインをマッピング**
```powershell
gcloud run domain-mappings create --service gungi-game --domain your-domain.com --region asia-northeast1
```

3. **DNSレコードを設定**（手順が表示されます）

---

## 📈 監視とアラート

### Cloud Monitoring の設定

1. [GCP Console](https://console.cloud.google.com/monitoring) を開く
2. 「アラートポリシー」を作成
3. 条件を設定：
   - エラー率が5%を超えたら通知
   - レスポンス時間が3秒を超えたら通知

---

## 🎉 完了！

これで軍儀ゲームがGCP Cloud Runで稼働しています！

**次のステップ:**
- 友達に共有してプレイしてもらう
- AIモデルを改善
- 新機能の追加

**サポートが必要な場合:**
- GitHubのIssuesで質問
- デプロイログを共有

楽しんでください！🎮✨
