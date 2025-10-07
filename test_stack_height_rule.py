"""
スタック高さのルールをテスト
1段の駒が2段の駒の上にツケることができないことを確認
"""

from src.engine.board import Board
from src.engine.piece import PieceType, Player, Piece
from src.engine.rules import Rules

print("="*60)
print("テスト: 1段の駒は2段の駒の上にツケられない")
print("="*60)

board = Board()

# (5, 5)に2段スタックを作成（味方）
piece1 = Piece(PieceType.HYO, Player.BLACK)
piece2 = Piece(PieceType.HYO, Player.BLACK)
board.add_piece((5, 5), piece1)
board.add_piece((5, 5), piece2)

# (5, 6)に1段の駒を配置（味方）小を使用（横移動可能）
piece3 = Piece(PieceType.SHO, Player.BLACK)
board.add_piece((5, 6), piece3)

print(f"(5,5): 黒の兵x2（2段）")
print(f"(5,6): 黒の小（1段）")
print(f"スタック高さ: (5,5)={board.get_stack_height((5, 5))}, (5,6)={board.get_stack_height((5, 6))}")
print()

hand_pieces = {}
legal_moves = Rules.get_legal_moves(board, Player.BLACK, hand_pieces)

# (5,6)の小からの合法手
hyo_moves = [m for m in legal_moves if m.from_pos == (5, 6)]

print(f"(5,6)の小の合法手: {len(hyo_moves)} 手")
for move in hyo_moves:
    print(f"  {move.from_pos} -> {move.to_pos}, タイプ: {move.move_type.name}")

# (5,5)へのSTACK移動があるかチェック
can_stack_to_55 = any(m.to_pos == (5, 5) and m.move_type.name == 'STACK' for m in hyo_moves)
print(f"\n(5,5)へツケられる: {can_stack_to_55}")
print(f"期待: False（1段 < 2段 なのでツケられない）")

if can_stack_to_55:
    print("\n❌ エラー: 1段の駒が2段の駒の上にツケられてしまっている")
else:
    print("\n✅ 正しい: 1段の駒は2段の駒の上にツケられない")

print("\n" + "="*60)
print("テスト: 2段の駒は2段の駒の上にツケられる")
print("="*60)

board2 = Board()

# (5, 5)に2段スタックを作成（味方）
piece4 = Piece(PieceType.HYO, Player.BLACK)
piece5 = Piece(PieceType.HYO, Player.BLACK)
board2.add_piece((5, 5), piece4)
board2.add_piece((5, 5), piece5)

# (5, 6)に2段スタックを作成（味方）小を使用（横移動可能）
piece6 = Piece(PieceType.HYO, Player.BLACK)
piece7 = Piece(PieceType.SHO, Player.BLACK)
board2.add_piece((5, 6), piece6)
board2.add_piece((5, 6), piece7)

print(f"(5,5): 黒の兵x2（2段）")
print(f"(5,6): 黒の小（2段）")
print(f"スタック高さ: (5,5)={board2.get_stack_height((5, 5))}, (5,6)={board2.get_stack_height((5, 6))}")
print()

legal_moves2 = Rules.get_legal_moves(board2, Player.BLACK, hand_pieces)

# (5,6)の小からの合法手
hyo_moves2 = [m for m in legal_moves2 if m.from_pos == (5, 6)]

print(f"(5,6)の小の合法手: {len(hyo_moves2)} 手")
for move in hyo_moves2:
    print(f"  {move.from_pos} -> {move.to_pos}, タイプ: {move.move_type.name}")

# (5,5)へのSTACK移動があるかチェック
can_stack_to_55_2 = any(m.to_pos == (5, 5) and m.move_type.name == 'STACK' for m in hyo_moves2)
print(f"\n(5,5)へツケられる: {can_stack_to_55_2}")
print(f"期待: True（2段 >= 2段 なのでツケられる）")

if can_stack_to_55_2:
    print("\n✅ 正しい: 2段の駒は2段の駒の上にツケられる")
else:
    print("\n❌ エラー: 2段の駒が2段の駒の上にツケられない")

print("\n" + "="*60)
print("テスト: 2段の駒は1段の駒の上にツケられる")
print("="*60)

board3 = Board()

# (5, 5)に1段の駒を配置（味方）
piece8 = Piece(PieceType.HYO, Player.BLACK)
board3.add_piece((5, 5), piece8)

# (5, 6)に2段スタックを作成（味方）小を使用（横移動可能）
piece9 = Piece(PieceType.HYO, Player.BLACK)
piece10 = Piece(PieceType.SHO, Player.BLACK)
board3.add_piece((5, 6), piece9)
board3.add_piece((5, 6), piece10)

print(f"(5,5): 黒の兵（1段）")
print(f"(5,6): 黒の小（2段）")
print(f"スタック高さ: (5,5)={board3.get_stack_height((5, 5))}, (5,6)={board3.get_stack_height((5, 6))}")
print()

legal_moves3 = Rules.get_legal_moves(board3, Player.BLACK, hand_pieces)

# (5,6)の小からの合法手
hyo_moves3 = [m for m in legal_moves3 if m.from_pos == (5, 6)]

print(f"(5,6)の小の合法手: {len(hyo_moves3)} 手")
for move in hyo_moves3:
    print(f"  {move.from_pos} -> {move.to_pos}, タイプ: {move.move_type.name}")

# (5,5)へのSTACK移動があるかチェック
can_stack_to_55_3 = any(m.to_pos == (5, 5) and m.move_type.name == 'STACK' for m in hyo_moves3)
print(f"\n(5,5)へツケられる: {can_stack_to_55_3}")
print(f"期待: True（2段 >= 1段 なのでツケられる）")

if can_stack_to_55_3:
    print("\n✅ 正しい: 2段の駒は1段の駒の上にツケられる")
else:
    print("\n❌ エラー: 2段の駒が1段の駒の上にツケられない")

print("\n" + "="*60)
print("テスト: 1段の駒は2段の敵駒の上にツケられない")
print("="*60)

board4 = Board()

# (5, 5)に2段スタックを作成（敵）
piece11 = Piece(PieceType.HYO, Player.WHITE)
piece12 = Piece(PieceType.HYO, Player.WHITE)
board4.add_piece((5, 5), piece11)
board4.add_piece((5, 5), piece12)

# (5, 6)に1段の駒を配置（自分）小を使用（横移動可能）
piece13 = Piece(PieceType.SHO, Player.BLACK)
board4.add_piece((5, 6), piece13)

print(f"(5,5): 白の兵x2（2段）- 敵")
print(f"(5,6): 黒の小（1段）- 自分")
print(f"スタック高さ: (5,5)={board4.get_stack_height((5, 5))}, (5,6)={board4.get_stack_height((5, 6))}")
print()

legal_moves4 = Rules.get_legal_moves(board4, Player.BLACK, hand_pieces)

# (5,6)の小からの合法手
hyo_moves4 = [m for m in legal_moves4 if m.from_pos == (5, 6)]

print(f"(5,6)の小の合法手: {len(hyo_moves4)} 手")
for move in hyo_moves4:
    print(f"  {move.from_pos} -> {move.to_pos}, タイプ: {move.move_type.name}")

# (5,5)へのSTACK移動があるかチェック
can_stack_to_55_4 = any(m.to_pos == (5, 5) and m.move_type.name == 'STACK' for m in hyo_moves4)
print(f"\n(5,5)の敵駒にツケられる: {can_stack_to_55_4}")
print(f"期待: False（1段 < 2段 なのでツケられない）")

if can_stack_to_55_4:
    print("\n❌ エラー: 1段の駒が2段の敵駒の上にツケられてしまっている")
else:
    print("\n✅ 正しい: 1段の駒は2段の敵駒の上にツケられない")
