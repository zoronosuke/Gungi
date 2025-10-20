# 軍儀テストスイート

このディレクトリには、軍儀プロジェクトの包括的なテストが含まれています。

## 📁 ディレクトリ構造

```
tests/
├── conftest.py              # pytest共通設定とフィクスチャ
├── unit/                    # 単体テスト
│   ├── test_board.py        # 盤面管理のテスト
│   ├── test_piece.py        # 駒の定義と動きのテスト
│   ├── test_stack.py        # スタック機能のテスト
│   └── test_rules.py        # ルール判定のテスト
├── integration/             # 統合テスト
│   ├── test_game_flow.py    # ゲーム全体の流れ
│   └── test_victory_conditions.py # 勝利条件のテスト
├── api/                     # APIテスト（未実装）
│   └── test_endpoints.py    # FastAPIエンドポイントのテスト
└── scenarios/               # シナリオテスト
    ├── test_edge_cases.py   # エッジケースと境界条件
    └── test_complex_stacks.py # 複雑なスタック状況（未実装）
```

## 🚀 テストの実行方法

### 全テストの実行
```powershell
pytest tests/ -v
```

### 特定のカテゴリのみ実行
```powershell
# 単体テストのみ
pytest tests/unit/ -v

# 統合テストのみ
pytest tests/integration/ -v

# シナリオテストのみ
pytest tests/scenarios/ -v
```

### 特定のテストファイルのみ実行
```powershell
pytest tests/unit/test_board.py -v
```

### 特定のテストクラス・メソッドのみ実行
```powershell
# 特定のクラス
pytest tests/unit/test_board.py::TestBoard -v

# 特定のメソッド
pytest tests/unit/test_board.py::TestBoard::test_board_initialization -v
```

### カバレッジレポート付き実行
```powershell
pytest tests/ --cov=src --cov-report=html
```

## 📋 テストカテゴリ

### 1. 単体テスト（Unit Tests）
個別のクラス・関数の動作を検証します。

- **test_board.py**: 盤面管理
  - 盤面の初期化
  - 駒の追加・削除
  - スタック高さの管理
  - 帥の位置追跡

- **test_piece.py**: 駒の定義と動き
  - 全14種類の駒の動きパターン
  - スタックレベルによる動きの変化
  - プレイヤー方向による動きの反転
  - ジャンプ機能

- **test_stack.py**: スタック機能
  - 3段までのスタック
  - 帥の上には乗せられない
  - 自分より高いスタックには移動不可
  - スタック全体の取得

- **test_rules.py**: ルール判定
  - 合法手の生成
  - 手の適用（移動・取得・ツケ・新）
  - 違法手の排除
  - ゲーム終了判定

### 2. 統合テスト（Integration Tests）
複数のコンポーネントを組み合わせた動作を検証します。

- **test_game_flow.py**: ゲーム全体の流れ
  - 初期配置の読み込み
  - 手番の交代
  - 複数手の進行
  - 長時間プレイの安定性

- **test_victory_conditions.py**: 勝利条件
  - 帥の捕獲による勝利
  - ゲーム終了判定
  - 勝者の決定

### 3. シナリオテスト（Scenario Tests）
特殊な状況や境界条件を検証します。

- **test_edge_cases.py**: エッジケースと境界条件
  - 盤の端・角での動き
  - 駒で埋まった盤面
  - 帥同士の隣接
  - 持ち駒の使い切り

## ✅ テスト項目チェックリスト

### Phase 1: 基本機能（必須）
- [x] 盤面の作成・初期化
- [x] スタックの追加・削除（最大3段）
- [x] 全14種類の駒の動き（1段目・2段目・3段目）
- [x] プレイヤー方向による動きの反転
- [x] 合法手の生成と違法手の排除
- [x] 手の適用（移動・取得・ツケ・新）

### Phase 2: ゲームフロー
- [x] 初期配置の読み込み
- [x] 手番の交代
- [x] ゲーム終了判定（帥の捕獲）
- [x] 複数手の進行

### Phase 3: エッジケース
- [x] 盤の端・角での動き
- [x] 帥の上にはスタックできない
- [x] 自分より高いスタックには移動・取得・ツケ不可
- [x] スタック全体の取得
- [ ] 最前線より前には打てない（新のルール）
- [ ] ジャンプ機能の詳細テスト

### Phase 4: 特殊ルール
- [ ] 砦は他の駒の上に乗れない
- [ ] 謀の寝返り（実装状況次第）
- [ ] 同種駒の重複禁止（実装状況次第）

## 🐛 デバッグのヒント

### テストが失敗した場合

1. **詳細なエラーメッセージを表示**
```powershell
pytest tests/unit/test_board.py -v --tb=long
```

2. **失敗したテストだけ再実行**
```powershell
pytest --lf -v
```

3. **標準出力を表示**
```powershell
pytest tests/ -v -s
```

4. **特定の警告を無視**
```powershell
pytest tests/ -v -W ignore::DeprecationWarning
```

### よくある問題

- **ImportError**: `sys.path`が正しく設定されているか確認
- **AttributeError**: メソッド名やクラス名のタイポを確認
- **AssertionError**: 期待値と実際の値を確認

## 📝 新しいテストの追加

新しいテストを追加する場合は、以下のテンプレートを使用してください：

```python
"""
テストの説明
"""

import pytest
from src.engine import Board, Player, PieceType, Piece, Rules


class TestNewFeature:
    """新機能のテストクラス"""
    
    def test_feature_works(self, empty_board):
        """機能が正しく動作することを確認"""
        # Arrange（準備）
        expected = True
        
        # Act（実行）
        result = some_function()
        
        # Assert（検証）
        assert result == expected, "エラーメッセージ"
```

## 🔧 継続的な改善

テストは継続的に改善していく必要があります：

1. **カバレッジの向上**: 未テストのコードを特定して追加
2. **テストの高速化**: 遅いテストを最適化
3. **テストの可読性**: わかりやすいテスト名とコメント
4. **テストの保守性**: 重複を削除し、共通処理をフィクスチャ化

## 📚 参考資料

- [pytest公式ドキュメント](https://docs.pytest.org/)
- [テスト駆動開発入門](https://www.oreilly.co.jp/books/9784274217883/)
