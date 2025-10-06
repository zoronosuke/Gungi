# 軍儀プロジェクト - セットアップガイド

## 📦 必要な環境

- **Python**: 3.10以降
- **pip**: 最新版
- **ブラウザ**: Chrome, Firefox, Safari など

## 🚀 クイックスタート

### 1. 依存パッケージのインストール

```powershell
# プロジェクトディレクトリに移動
cd c:\Users\okada\Documents\2025\memo\Gungi\Gungi

# 仮想環境を作成（推奨）
python -m venv venv

# 仮想環境を有効化
.\venv\Scripts\Activate.ps1

# 依存パッケージをインストール
pip install -r requirements.txt
```

### 2. エンジンのテスト

```powershell
# ゲームエンジンが正しく動作するか確認
python tests\test_engine.py
```

### 3. APIサーバの起動

```powershell
# 開発サーバを起動
python run_server.py
```

サーバが起動したら、以下のURLにアクセスできます：
- **API エンドポイント**: http://localhost:8000
- **APIドキュメント**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 4. フロントエンドの起動

別のターミナルを開いて：

```powershell
cd frontend

# シンプルなHTTPサーバで起動（Python標準機能）
python -m http.server 3000
```

ブラウザで http://localhost:3000 にアクセス

## 📁 プロジェクト構造

```
Gungi/
├── src/
│   ├── engine/              # ゲームルールエンジン
│   │   ├── __init__.py
│   │   ├── piece.py         # 駒の定義
│   │   ├── board.py         # 盤面管理
│   │   ├── move.py          # 手の表現
│   │   ├── rules.py         # ルール判定
│   │   └── initial_setup.py # 初期配置
│   ├── model/               # 深層学習モデル
│   │   ├── __init__.py
│   │   ├── network.py       # ニューラルネットワーク
│   │   └── mcts.py          # モンテカルロ木探索
│   └── api/                 # FastAPIサーバ
│       ├── __init__.py
│       └── main.py          # APIエンドポイント
├── frontend/                # Web UI
│   ├── index.html
│   ├── game.js
│   └── styles.css
├── tests/                   # テストコード
│   └── test_engine.py
├── requirements.txt         # Python依存パッケージ
├── run_server.py           # サーバ起動スクリプト
└── README.md
```

## 🎮 使い方

### Webブラウザでプレイ

1. APIサーバとフロントエンドサーバを起動
2. ブラウザで http://localhost:3000 にアクセス
3. 「新しいゲーム」ボタンをクリック
4. 盤面の駒をクリックして選択 → 移動先をクリック
5. 「AIに手を打たせる」でコンピュータと対戦

### APIを直接使う

#### 新しいゲームを開始

```bash
curl -X POST http://localhost:8000/new_game
```

#### 手を適用

```bash
curl -X POST http://localhost:8000/apply_move/{game_id} \
  -H "Content-Type: application/json" \
  -d '{
    "from_row": 6,
    "from_col": 0,
    "to_row": 5,
    "to_col": 0,
    "move_type": "NORMAL"
  }'
```

#### AIの手を取得

```bash
curl -X POST http://localhost:8000/predict/{game_id} \
  -H "Content-Type: application/json" \
  -d '{"game_id": "...", "depth": 1}'
```

## 🧪 テストの実行

```powershell
# エンジンのテスト
python tests\test_engine.py

# 将来: pytestでのテスト
pytest tests/
```

## 🔧 トラブルシューティング

### ImportError が発生する場合

Pythonのパスを確認してください：

```powershell
# プロジェクトルートで実行
$env:PYTHONPATH = "."
python run_server.py
```

### CORS エラーが発生する場合

`src/api/main.py` で CORS 設定を確認してください。
開発中は `allow_origins=["*"]` になっているはずです。

### ポートが使用中の場合

別のポートを使用：

```powershell
# APIサーバ
python -m uvicorn src.api.main:app --reload --port 8001

# フロントエンド
cd frontend
python -m http.server 3001
```

## 📚 次のステップ

### Phase 2: 深層強化学習の実装

1. **自己対戦データの収集**
   ```python
   # src/model/trainer.py を実装
   # 自己対戦でゲームデータを生成
   ```

2. **モデルの学習**
   ```python
   # AlphaZero型の学習ループ
   # MCTSで探索 → ニューラルネットを更新
   ```

3. **学習済みモデルの保存・読み込み**
   ```python
   # モデルの保存
   torch.save(model.state_dict(), 'gungi_model.pt')
   ```

### Phase 3: 機能拡張

- [ ] 持ち駒を打つ機能の実装
- [ ] 特殊な駒の能力（謀の寝返りなど）
- [ ] リプレイ機能
- [ ] オンライン対戦機能
- [ ] 棋譜の保存・読み込み

## 📄 ライセンス

このプロジェクトは学習目的で作成されています。

## 🙏 謝辞

- 『HUNTER×HUNTER』 冨樫義博先生
- 軍儀のルール考察をされた西辻三九郎氏ほか多くのファンの方々
