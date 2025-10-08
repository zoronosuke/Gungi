# 軍儀 - 「新」（持ち駒配置）クイックガイド

## 「新」とは？

「新」（あらた）は、持ち駒を盤面に配置する手です。将棋の「駒を打つ」に似ていますが、軍儀独自のルールがあります。

---

## 📋 基本的な使い方

### 1. APIで「新」を実行する

```bash
curl -X POST "http://localhost:8003/apply_move/ゲームID" \
  -H "Content-Type: application/json" \
  -d '{
    "move_type": "DROP",
    "piece_type": "HYO",
    "to_row": 7,
    "to_col": 5
  }'
```

### 2. パラメータ

| パラメータ | 必須 | 説明 | 例 |
|-----------|------|------|-----|
| `move_type` | ✅ | `"DROP"` 固定 | `"DROP"` |
| `piece_type` | ✅ | 駒の種類 | `"HYO"`, `"YARI"` |
| `to_row` | ✅ | 配置先の行（0-8） | `7` |
| `to_col` | ✅ | 配置先の列（0-8） | `5` |
| `from_row` | - | 不要（`null`） | `null` |
| `from_col` | - | 不要（`null`） | `null` |

---

## 🎴 駒の種類一覧

| 表示 | piece_type | 読み | 初期持ち駒数 |
|------|------------|------|-------------|
| 帥 | `SUI` | スイ | 0 |
| 大 | `DAI` | ダイ | 0 |
| 中 | `CHUU` | チュウ | 0 |
| 小 | `SHO` | ショウ | 2 |
| 侍 | `SAMURAI` | サムライ | 0 |
| 兵 | `HYO` | ヒョウ | 1 |
| 馬 | `UMA` | ウマ | 1 |
| 忍 | `SHINOBI` | シノビ | 1 |
| 槍 | `YARI` | ヤリ | 2 |
| 砦 | `TORIDE` | トリデ | 0 |
| 弓 | `YUMI` | ユミ | 0 |
| 筒 | `TSUTU` | ツツ | 1 |
| 砲 | `HOU` | ホウ | 1 |
| 謀 | `BOU` | ボウ | 1 |

**初期配置完了時の持ち駒例:**
- 初期配置で盤面に配置しなかった駒が持ち駒になります
- 上記は一例です（配置方法によって変わります）

---

## ✅ 配置できる条件

1. **持ち駒がある**
   - その駒を持っている必要があります

2. **最前線より後ろ**
   - 黒（先手）：自分の駒がある最も小さい行より小さい行には置けない
   - 白（後手）：自分の駒がある最も大きい行より大きい行には置けない

3. **空マスまたは味方の駒の上**
   - 空いているマスに置ける
   - 味方の駒の上に置ける（スタックを形成）

4. **スタック高さ3未満**
   - 既に3枚積まれているマスには置けない

---

## ❌ 配置できない場合

| 制限 | 説明 |
|------|------|
| 帥の上 | 帥（王）の上には配置できません |
| 敵の駒の上 | 敵の駒の上には配置できません |
| 砦を他の駒の上 | 砦は地上にしか配置できません |
| 最前線より前 | 自軍の最前線より敵側には置けません |
| スタック高さ3 | 既に3枚積まれている場所には置けません |

---

## 📍 最前線の判定

### 黒（先手）の場合
- 盤面下側（行8）から開始
- 上（行0方向）が敵側
- **最前線 = 最も小さい行番号**

```
行0 ← 敵側（白の陣地）
行1
行2
---
行3
行4
行5 ← 最前線の例（黒の駒がある最も敵側の行）
---
行6
行7
行8 ← 自陣（黒の陣地）
```

→ 行5が最前線なら、行0-4には配置不可、行5-8には配置可

### 白（後手）の場合
- 盤面上側（行0）から開始
- 下（行8方向）が敵側
- **最前線 = 最も大きい行番号**

```
行0 ← 自陣（白の陣地）
行1
行2
---
行3 ← 最前線の例（白の駒がある最も敵側の行）
行4
行5
---
行6
行7
行8 ← 敵側（黒の陣地）
```

→ 行3が最前線なら、行4-8には配置不可、行0-3には配置可

---

## 💻 プログラムでの実装例

### Python

```python
import requests

API_URL = 'http://localhost:8003'

def drop_piece(game_id, piece_type, to_row, to_col):
    """持ち駒を配置"""
    response = requests.post(
        f'{API_URL}/apply_move/{game_id}',
        json={
            'move_type': 'DROP',
            'piece_type': piece_type,
            'to_row': to_row,
            'to_col': to_col
        }
    )
    return response.json()

# 使用例
result = drop_piece('game-id', 'HYO', 7, 5)
print(result['message'])
```

### JavaScript

```javascript
async function dropPiece(gameId, pieceType, toRow, toCol) {
    const response = await fetch(`http://localhost:8003/apply_move/${gameId}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            move_type: 'DROP',
            piece_type: pieceType,
            to_row: toRow,
            to_col: toCol
        })
    });
    return await response.json();
}

// 使用例
dropPiece('game-id', 'HYO', 7, 5)
    .then(result => console.log(result.message));
```

---

## 🔍 合法手の確認

配置可能な場所を確認するには:

```bash
curl "http://localhost:8003/get_legal_moves/ゲームID"
```

レスポンスから「新」の手だけをフィルタ:

```python
legal_moves = get_legal_moves(game_id)
drop_moves = [m for m in legal_moves if m['type'] == 'DROP']

for move in drop_moves:
    print(f"{move['piece_type']} -> {move['to']}")
```

---

## 📊 持ち駒の確認

現在の持ち駒を確認:

```bash
curl "http://localhost:8003/get_game/ゲームID"
```

レスポンス例:
```json
{
  "hand_pieces": {
    "BLACK": {
      "HYO": 1,
      "YARI": 2,
      "UMA": 1
    },
    "WHITE": {
      "HYO": 1,
      "YARI": 2
    }
  }
}
```

---

## 🎯 実践例

### 例1: 空マスに兵を配置

```bash
curl -X POST "http://localhost:8003/apply_move/abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "move_type": "DROP",
    "piece_type": "HYO",
    "to_row": 7,
    "to_col": 5
  }'
```

### 例2: 味方の駒の上に槍を配置（スタック形成）

```bash
# (7,4)に既に黒の駒がある場合
curl -X POST "http://localhost:8003/apply_move/abc123" \
  -H "Content-Type": application/json" \
  -d '{
    "move_type": "DROP",
    "piece_type": "YARI",
    "to_row": 7,
    "to_col": 4
  }'
```

→ (7,4)にスタックが形成されます

---

## ❗ トラブルシューティング

### エラー: "DROPには piece_type が必要です"
→ `piece_type` パラメータを指定してください

### エラー: "無効な手です"
原因の可能性:
- ✗ 持ち駒がない
- ✗ 最前線より前に配置しようとしている
- ✗ 帥の上または敵の駒の上に配置しようとしている
- ✗ スタック高さが既に3
- ✗ 砦を他の駒の上に配置しようとしている

**解決方法:**
1. 持ち駒を確認: `GET /get_game/{game_id}`
2. 合法手を確認: `GET /get_legal_moves/{game_id}`
3. 配置可能な「新」の手をフィルタして確認

---

## 🎮 ブラウザでテスト

1. サーバー起動:
   ```bash
   python run_server.py
   ```

2. ブラウザで開く:
   http://localhost:8003/docs

3. `/apply_move/{game_id}` を選択

4. "Try it out" をクリック

5. パラメータを入力して "Execute"

---

## 📝 まとめ

「新」を使うには:

```json
{
  "move_type": "DROP",
  "piece_type": "駒の種類",
  "to_row": 行,
  "to_col": 列
}
```

**重要なルール:**
- ✅ 持ち駒があること
- ✅ 最前線より後ろ
- ✅ 空マスか味方の駒の上（帥を除く）
- ✅ スタック高さ3未満
- ❌ 砦は地上のみ

持ち駒を戦略的に配置して、軍儀の戦局を有利に進めましょう！
