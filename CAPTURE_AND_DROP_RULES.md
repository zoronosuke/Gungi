# 軍儀 - 駒の取得と配置ルールの実装

## 実装した軍儀のルール

### 1. 駒を取る際のルール

**軍儀のルール**: 駒を取るときは、そのマスの駒全部を取る

#### 実装内容
- `Board.capture_piece()`: スタック全体を取る（全ての駒を削除）
- `Rules.apply_move()`: CAPTURE moveでスタック全体の駒リストを返す
- 取った駒は全て除外される（持ち駒として使用できない）

```python
def capture_piece(self, position: Tuple[int, int]) -> List[Piece]:
    """
    指定位置の駒スタック全体を取る（盤面から除去）
    軍儀のルール: 駒を取るときは、そのマスの駒全部を取る
    返り値: 取った駒のリスト（下から上への順）
    """
    captured_pieces = []
    stack = self.get_stack(position)
    
    # スタック内の全ての駒を取得
    while not stack.is_empty():
        piece = self.remove_piece(position)
        if piece:
            captured_pieces.append(piece)
    
    return captured_pieces
```

### 2. 取った駒の扱い

**軍儀のルール**: 相手の取った駒は自分の持ち駒として使えない

#### 実装内容
- 取った駒は盤面から除去され、ゲームから除外される
- 将棋と異なり、取った駒を再利用できない（チェスと同じ）
- `Rules.apply_move()`は取った駒のリストを返すが、それは記録用のみ

### 3. 持ち駒の配置（「新」）

**軍儀のルール**: 自分の持ち駒は「新」で配置できる

#### 実装内容

##### 配置の制限
1. **最前線より前（敵寄り）には置けない**
   - 黒（先手）: 最前線の行より上（小さい行番号）には置けない
   - 白（後手）: 最前線の行より下（大きい行番号）には置けない
   - 最前線 = 最も敵側に近い自分の駒がある行

2. **配置可能な場所**
   - 空マス
   - 味方の駒の上（帥の上を除く）
   - スタック高さが3未満

3. **配置できない場所**
   - 敵の駒の上
   - 帥の上
   - スタック高さが3のマス
   - 砦は他の駒の上に置けない

```python
@staticmethod
def _can_drop_piece_at(
    board: Board,
    pos: Tuple[int, int],
    piece_type: PieceType,
    player: Player
) -> bool:
    """
    指定位置に駒を打てるか確認
    
    軍儀のルール:
    - 空マスには打てる
    - 味方の駒の上には打てる（帥の上を除く、スタック高さ3未満）
    - 敵の駒の上には打てない
    - 砦は他の駒の上に乗れない
    """
    target_piece = board.get_top_piece(pos)
    stack_height = board.get_stack_height(pos)
    
    # スタック高さが3の場合は打てない
    if stack_height >= 3:
        return False
    
    # 砦は他の駒の上に乗れない
    if piece_type == PieceType.TORIDE and stack_height > 0:
        return False
    
    # 空マスには打てる
    if target_piece is None:
        return True
    
    # 味方の駒の上には打てる（帥を除く）
    if target_piece.owner == player:
        if target_piece.piece_type == PieceType.SUI:
            return False  # 帥の上には打てない
        return True
    
    # 敵の駒の上には打てない
    return False
```

### 4. 最前線の判定

```python
@staticmethod
def _get_frontline_row(board: Board, player: Player) -> Optional[int]:
    """
    指定プレイヤーの最前線の行を取得
    
    軍儀のルール: 
    - 黒（先手）: 盤面下側から開始、上（小さい行番号）が敵側
    - 白（後手）: 盤面上側から開始、下（大きい行番号）が敵側
    - 最前線 = 最も敵側に近い自分の駒がある行
    
    返り値: 最前線の行番号、駒がない場合はNone
    """
```

## 使用例

### 持ち駒を配置する

```python
# 持ち駒の辞書を用意
hand_pieces = {
    PieceType.HYO: 2,
    PieceType.YARI: 1
}

# 合法な「新」の手を取得
legal_moves = Rules.get_legal_moves(board, player, hand_pieces)

# 「新」の手をフィルタ
drop_moves = [m for m in legal_moves if m.move_type == MoveType.DROP]

# 手を適用
for move in drop_moves:
    success, captured = Rules.apply_move(board, move, hand_pieces)
    if success:
        print(f"駒を配置: {move.piece_type.name} -> {move.to_pos}")
```

### 駒を取る

```python
# CAPTUREの手を実行
move = Move.create_capture_move(from_pos=(7, 4), to_pos=(5, 4), player=Player.BLACK)
success, captured_pieces = Rules.apply_move(board, move)

if success and captured_pieces:
    print(f"取った駒: {len(captured_pieces)}枚")
    for piece in captured_pieces:
        print(f"  - {piece}")
    # 注: 取った駒は持ち駒として使えない（除外される）
```

## 変更したファイル

1. `src/engine/board.py`
   - `Board.capture_piece()`: 戻り値を`List[Piece]`に変更

2. `src/engine/rules.py`
   - `Rules.apply_move()`: 戻り値を`Tuple[bool, Optional[List[Piece]]]`に変更
   - `Rules._get_drop_moves()`: 最前線チェックのロジックを改善
   - `Rules._get_frontline_row()`: スタック内の全駒をチェック
   - `Rules._can_drop_piece_at()`: 砦の制限を追加

## テスト項目

以下の動作を確認してください：

1. ✅ 駒を取るとスタック全体が除去される
2. ✅ 取った駒は持ち駒として使えない
3. ✅ 持ち駒を「新」で配置できる
4. ✅ 最前線より前（敵寄り）には配置できない
5. ✅ 空マスに配置できる
6. ✅ 味方の駒の上に配置できる（帥を除く）
7. ✅ 敵の駒の上には配置できない
8. ✅ 砦は他の駒の上に配置できない
9. ✅ スタック高さ3のマスには配置できない
