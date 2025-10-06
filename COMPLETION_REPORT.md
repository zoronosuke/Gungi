# 軍儀 駒の動きとスタック機能 実装完了報告

## 概要

GitHubリポジトリ https://github.com/nigaor/gungi を参考に、軍儀の本家ルールに準拠した駒の動きとスタック（ツケ）機能を完全実装しました。

## 実装した主要機能

### 1. 駒の動きの詳細定義 ✅

全14種類の駒について、スタックレベル（1段目、2段目、3段目/極）に応じた動きパターンを定義：

- 帥（SUI）、大（DAI）、中（CHUU）、小（SHO）
- 侍（SAMURAI）、兵（HYO）、馬（UMA）、忍（SHINOBI）
- 槍（YARI）、砦（TORIDE）、弓（YUMI）、筒（TSUTU）
- 砲（HOU）、謀（BOU）

### 2. スタック（ツケ）のルール ✅

- 最大3段までスタック可能
- 帥の上には乗せられない（`can_be_stacked_on()`）
- 自分より高いスタックには移動・取得・ツケ不可
- スタックレベルに応じて駒の動きが変化

### 3. ジャンプ機能 ✅

3段目（極）の駒の一部は、途中に駒があってもジャンプ可能：
- 兵、馬、忍、砦、弓、筒、砲、謀

### 4. 持ち駒を打つ（新）のルール ✅

- 最前線より前には打てない
- 空マスまたは味方の駒の上に打てる
- 帥の上には打てない
- スタック高さ3には打てない

### 5. プレイヤー方向の考慮 ✅

- 黒プレイヤー（先手）：上方向が前
- 白プレイヤー（後手）：下方向が前
- 駒の動きを自動的に反転

## テスト結果

### 基本機能テスト（test_engine.py）
- ✅ 盤面作成
- ✅ 初期配置
- ✅ 合法手生成（46手）
- ✅ 手の適用
- ✅ ゲームフロー

### スタック機能テスト（test_stack.py）
- ✅ スタックレベルによる動きの変化
  - 1段目：基本動き
  - 2段目：強化動き
  - 3段目：最強動き（ジャンプ可能）
- ✅ 帥の保護（上に乗せられない）
- ✅ 4段目制限（3段までのみ）
- ✅ スタック高さ制限（自分より高いスタックへの移動不可）

## 変更ファイル

### `src/engine/piece.py`
- `PIECE_MOVE_PATTERNS`辞書を追加（全駒の動きパターン定義）
- `get_move_pattern(stack_level)`メソッドを実装
- スタックレベル（1/2/3）に応じた動きを返す

### `src/engine/rules.py`
- `_get_possible_move_positions()`を更新
  - 新しい動きパターンを使用
  - ジャンプ機能のサポート
  - プレイヤー方向の考慮
- `_get_piece_legal_moves()`を更新
  - スタック高さ制限の実装
  - 帥の保護ルールの実装
- `_get_drop_moves()`を実装
  - 最前線制限
  - 持ち駒を打つルール
- `_get_frontline_row()`を追加
- `_can_drop_piece_at()`を追加

### テストファイル
- `tests/test_stack.py`を新規作成
  - スタック機能の包括的テスト

### ドキュメント
- `IMPLEMENTATION_NOTES.md`を新規作成
- `README.md`を更新

## 参照リポジトリとの互換性

参照した https://github.com/nigaor/gungi のTypeScript実装と同等の機能を実現：

```typescript
// 参照リポジトリの構造
pieceRules: {
  base: { moves, maxSteps, canJump, directional },
  evolved: { ... },
  mastered: { ... }
}
```

↓ Python実装

```python
PIECE_MOVE_PATTERNS = {
    PieceType.XXX: {
        'base': {'moves': [...], 'maxSteps': 1, 'canJump': False},
        'evolved': {'moves': [...], 'maxSteps': 1, 'canJump': False},
        'mastered': {'moves': [...], 'maxSteps': 1, 'canJump': True}
    }
}
```

## 動作確認

すべてのテストが成功し、以下を確認：
1. 駒が正しく動く
2. スタックレベルによって動きが変化する
3. 不正な移動が制限される
4. 帥が保護される
5. 持ち駒を打つルールが正しく機能する

## 次のステップ

この実装により、軍儀のコアルールエンジンが完成しました。次は以下を実装できます：

1. FastAPI サーバとの統合
2. フロントエンドとの連携
3. 深層学習モデルの学習
4. AI対戦機能

---

実装日: 2025年10月7日
参照: https://github.com/nigaor/gungi
