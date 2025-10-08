# 軍儀 - 「新」（持ち駒配置）の使い方

## 「新」とは？

「新」（あらた）は、軍儀で持ち駒を盤面に配置する手のことです。将棋の「駒を打つ」に似ていますが、軍儀独自のルールがあります。

## API経由での「新」の実行方法

### エンドポイント
```
POST /apply_move/{game_id}
```

### リクエストパラメータ（JSON）

```json
{
  "move_type": "DROP",
  "piece_type": "HYO",
  "to_row": 7,
  "to_col": 5,
  "from_row": null,
  "from_col": null
}
```

### パラメータ説明

- `move_type`: `"DROP"` を指定（必須）
- `piece_type`: 配置する駒の種類（必須）
  - 例: `"HYO"`, `"YARI"`, `"UMA"`, `"SHINOBI"` など
- `to_row`: 配置先の行（0-8）
- `to_col`: 配置先の列（0-8）
- `from_row`: `null` にする（DROPでは不要）
- `from_col`: `null` にする（DROPでは不要）

## 駒の種類（piece_type）

| 駒名 | piece_type | 表示 |
|------|------------|------|
| 帥 | SUI | 帥 |
| 大将 | DAI | 大 |
| 中将 | CHUU | 中 |
| 小将 | SHO | 小 |
| 侍 | SAMURAI | 侍 |
| 兵 | HYO | 兵 |
| 馬 | UMA | 馬 |
| 忍 | SHINOBI | 忍 |
| 槍 | YARI | 槍 |
| 砦 | TORIDE | 砦 |
| 弓 | YUMI | 弓 |
| 筒 | TSUTU | 筒 |
| 砲 | HOU | 砲 |
| 謀 | BOU | 謀 |

## 「新」のルール・制限

### ✅ 配置できる条件
1. **持ち駒がある** - その駒を持っている必要があります
2. **最前線より後ろ** - 自軍の最前線より前（敵寄り）には配置できません
3. **空マスまたは味方の駒の上** - 空いているマスか、味方の駒の上に配置できます
4. **スタック高さ3未満** - 既に3枚積まれているマスには配置できません

### ❌ 配置できない場合
1. **帥の上** - 帥の上には配置できません
2. **敵の駒の上** - 敵の駒の上には配置できません
3. **砦を他の駒の上** - 砦は他の駒の上に配置できません（地上のみ）
4. **最前線より前** - 自分の駒がある最も敵側の行より前には配置できません

## 最前線の判定

- **黒（先手）**: 最も小さい行番号が前線
  - 例: 行6に駒があれば、行0-5には配置不可、行6-8には配置可
- **白（後手）**: 最も大きい行番号が前線
  - 例: 行2に駒があれば、行3-8には配置不可、行0-2には配置可

## 使用例

### 例1: 空マスに兵を配置

```bash
curl -X POST "http://localhost:8003/apply_move/your-game-id" \
  -H "Content-Type: application/json" \
  -d '{
    "move_type": "DROP",
    "piece_type": "HYO",
    "to_row": 7,
    "to_col": 5
  }'
```

### 例2: 味方の駒の上に槍を配置（スタックを形成）

```bash
curl -X POST "http://localhost:8003/apply_move/your-game-id" \
  -H "Content-Type: application/json" \
  -d '{
    "move_type": "DROP",
    "piece_type": "YARI",
    "to_row": 7,
    "to_col": 4
  }'
```

## JavaScriptでの実装例

```javascript
/**
 * 持ち駒を配置する（新）
 */
async function dropPiece(gameId, pieceType, toRow, toCol) {
    try {
        const response = await fetch(`${API_BASE_URL}/apply_move/${gameId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                move_type: 'DROP',
                piece_type: pieceType,
                to_row: toRow,
                to_col: toCol,
                from_row: null,
                from_col: null
            })
        });
        
        if (!response.ok) {
            throw new Error('駒の配置に失敗しました');
        }
        
        const data = await response.json();
        
        if (data.success) {
            console.log('駒を配置しました:', data.message);
            updateGameState(data.game_state);
        } else {
            console.error('無効な手です:', data.message);
        }
        
        return data;
    } catch (error) {
        console.error('Error dropping piece:', error);
        throw error;
    }
}

// 使用例
dropPiece('game-id-123', 'HYO', 7, 5)
    .then(result => console.log('Success:', result))
    .catch(error => console.error('Error:', error));
```

## レスポンス例

### 成功時

```json
{
  "success": true,
  "message": "手を適用しました",
  "game_state": {
    "game_id": "abc123",
    "board": {...},
    "current_player": "WHITE",
    "move_count": 5,
    "hand_pieces": {
      "BLACK": {
        "HYO": 1,
        "YARI": 2
      },
      "WHITE": {...}
    },
    "game_over": false,
    "winner": null
  },
  "legal_moves": [...]
}
```

### 失敗時

```json
{
  "success": false,
  "message": "無効な手です",
  "game_state": {...}
}
```

## ブラウザでのテスト

1. サーバーを起動:
```bash
python run_server.py
```

2. ブラウザで http://localhost:8003/docs を開く

3. `/apply_move/{game_id}` エンドポイントを選択

4. "Try it out" をクリック

5. パラメータを入力:
   - game_id: ゲームIDを入力
   - Request body:
```json
{
  "move_type": "DROP",
  "piece_type": "HYO",
  "to_row": 7,
  "to_col": 5,
  "from_row": null,
  "from_col": null
}
```

6. "Execute" をクリック

## トラブルシューティング

### エラー: "DROPには piece_type が必要です"
- `piece_type` パラメータを指定してください

### エラー: "無効な手です"
考えられる原因:
- 持ち駒がない
- 最前線より前に配置しようとしている
- 帥の上または敵の駒の上に配置しようとしている
- スタック高さが既に3に達している
- 砦を他の駒の上に配置しようとしている

### デバッグ方法

合法手を確認:
```bash
curl "http://localhost:8003/get_legal_moves/your-game-id"
```

DROPの手のみをフィルタ:
```javascript
const legalMoves = await getLegalMoves(gameId);
const dropMoves = legalMoves.filter(m => m.type === 'DROP');
console.log('配置可能な手:', dropMoves);
```

## 持ち駒の確認

ゲーム状態から持ち駒を確認:
```javascript
const gameState = await getGameState(gameId);
console.log('黒の持ち駒:', gameState.hand_pieces.BLACK);
console.log('白の持ち駒:', gameState.hand_pieces.WHITE);
```

## まとめ

「新」を使用するには:
1. `move_type: "DROP"` を指定
2. `piece_type` で配置する駒を指定
3. `to_row`, `to_col` で配置先を指定
4. ルールの制限を守る（最前線、帥の上、敵の駒の上、スタック高さなど）

持ち駒を戦略的に配置することで、軍儀の戦局を有利に進めることができます！
