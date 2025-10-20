# 軍儀（Gungi）プロジェクト

『HUNTER×HUNTER』に登場する架空のボードゲーム「軍儀」を実装し、深層強化学習によるAI対戦をWebブラウザで実現するプロジェクトです。

## 🎮 デモ

**オンラインでプレイ**: [https://gungi-game-57998005741.asia-northeast1.run.app](https://gungi-game-57998005741.asia-northeast1.run.app)

ブラウザで今すぐ軍儀をプレイできます！（GCP Cloud Run上でホスティング）

## プロジェクト構成

```
Gungi/
├── src/
│   ├── engine/          # ゲームルールエンジン
│   │   ├── board.py     # 盤面管理
│   │   ├── piece.py     # 駒の定義と動き
│   │   ├── move.py      # 手の表現
│   │   └── rules.py     # ルール判定
│   ├── model/           # 深層学習モデル
│   │   ├── network.py   # ニューラルネットワーク
│   │   ├── mcts.py      # モンテカルロ木探索
│   │   └── trainer.py   # 学習ロジック
│   └── api/             # FastAPIサーバ
│       └── main.py      # APIエンドポイント
├── frontend/            # Web UI
│   ├── index.html       # メインページ
│   ├── game.js          # ゲームロジック
│   └── styles.css       # スタイル
├── tests/               # テストコード
└── requirements.txt     # Python依存パッケージ
```

## 機能

### 実装済み
- ✅ ルールエンジンの基本実装
- ✅ 公式初期配置からのゲーム開始
- ✅ 各駒の移動ルール（本家準拠）
- ✅ スタック機能（ツケ）の実装
- ✅ スタックレベルに応じた駒の動きの変化（1段目→2段目→3段目/極）
- ✅ 自分より高いスタックへの移動制限
- ✅ 帥の保護ルール（上に乗せられない）
- ✅ ジャンプ機能（3段目の一部駒）
- ✅ 持ち駒を打つ（新）のルール（最前線制限）

### 実装予定
1. **Phase 1**: FastAPI サーバの構築
   - `/new_game` - 新しいゲーム開始
   - `/apply_move` - 手を適用
   - `/predict` - AIの次の手を予測

2. **Phase 2**: 深層強化学習モデル
   - AlphaZero型の方策+価値ネットワーク
   - 自己対戦による学習
   - MCTSによる推論

3. **Phase 3**: Web UI
   - 盤面の視覚化
   - クリック操作での駒移動
   - リアルタイムAI対戦

## セットアップ

### 必要な環境
- Python 3.10以降
- Node.js 18以降（フロントエンド用）

### インストール

```bash
# Python依存パッケージ
pip install -r requirements.txt

# フロントエンド依存パッケージ（必要に応じて）
cd frontend
npm install
```

### 開発サーバの起動

```bash
# APIサーバ起動
python -m uvicorn src.api.main:app --reload --port 8000

# フロントエンド開発サーバ（別ターミナル）
cd frontend
npm run dev
```

## ゲームルール概要

- 9×9マスの盤面
- 各プレイヤー25枚の駒（14種類）
- 駒を最大3段まで積み重ね可能（ツケ）
- 相手の帥（すい）を取れば勝利

詳細は `ルール.txt` を参照してください。

## ライセンス

このプロジェクトは学習目的で作成されています。
