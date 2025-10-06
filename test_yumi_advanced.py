"""
弓の特殊ルールをより詳細にテストするスクリプト
"""

from src.engine.board import Board
from src.engine.piece import PieceType, Player, Piece
from src.engine.rules import Rules

print("=== テスト5: 2段の弓 + 3段の障害物（飛び越えられない）===")
board5 = Board()

# 2段スタックに弓を配置 (5, 5)
base5 = Piece(PieceType.HYO, Player.BLACK)
yumi5 = Piece(PieceType.YUMI, Player.BLACK)
board5.add_piece((5, 5), base5)
board5.add_piece((5, 5), yumi5)

# 1つ前に3段スタックを配置 (4, 5)
obstacle5a = Piece(PieceType.HYO, Player.BLACK)
obstacle5b = Piece(PieceType.HYO, Player.BLACK)
obstacle5c = Piece(PieceType.HYO, Player.BLACK)
board5.add_piece((4, 5), obstacle5a)
board5.add_piece((4, 5), obstacle5b)
board5.add_piece((4, 5), obstacle5c)

print(f"(5,5): 黒の弓（2段）")
print(f"(4,5): 黒の兵x3（3段）- 1つ前")
print(f"弓のスタック高さ: {board5.get_stack_height((5, 5))}")
print(f"障害物のスタック高さ: {board5.get_stack_height((4, 5))}")
print()

hand_pieces = {}
legal_moves5 = Rules.get_legal_moves(board5, Player.BLACK, hand_pieces)
yumi_moves5 = [m for m in legal_moves5 if m.from_pos == (5, 5)]

print(f"弓の合法手: {len(yumi_moves5)} 手")
for move in yumi_moves5:
    print(f"  {move.from_pos} -> {move.to_pos}, タイプ: {move.move_type.name}")

can_jump_to_35_5 = any(m.to_pos == (3, 5) for m in yumi_moves5)
print(f"\n(3,5)へジャンプ可能: {can_jump_to_35_5}")
print(f"期待: False（2段 < 3段 なので飛び越えられない）")

print("\n" + "="*60)
print("=== テスト6: 3段の弓 + 3段の障害物（飛び越えられる）===")
board6 = Board()

# 3段スタックに弓を配置 (5, 5)
base6a = Piece(PieceType.HYO, Player.BLACK)
base6b = Piece(PieceType.HYO, Player.BLACK)
yumi6 = Piece(PieceType.YUMI, Player.BLACK)
board6.add_piece((5, 5), base6a)
board6.add_piece((5, 5), base6b)
board6.add_piece((5, 5), yumi6)

# 1つ前に3段スタックを配置 (4, 5)
obstacle6a = Piece(PieceType.HYO, Player.BLACK)
obstacle6b = Piece(PieceType.HYO, Player.BLACK)
obstacle6c = Piece(PieceType.HYO, Player.BLACK)
board6.add_piece((4, 5), obstacle6a)
board6.add_piece((4, 5), obstacle6b)
board6.add_piece((4, 5), obstacle6c)

print(f"(5,5): 黒の弓（3段）")
print(f"(4,5): 黒の兵x3（3段）- 1つ前")
print(f"弓のスタック高さ: {board6.get_stack_height((5, 5))}")
print(f"障害物のスタック高さ: {board6.get_stack_height((4, 5))}")
print()

legal_moves6 = Rules.get_legal_moves(board6, Player.BLACK, hand_pieces)
yumi_moves6 = [m for m in legal_moves6 if m.from_pos == (5, 5)]

print(f"弓の合法手: {len(yumi_moves6)} 手")
for move in yumi_moves6:
    print(f"  {move.from_pos} -> {move.to_pos}, タイプ: {move.move_type.name}")

can_jump_to_35_6 = any(m.to_pos == (3, 5) for m in yumi_moves6)
print(f"\n(3,5)へジャンプ可能: {can_jump_to_35_6}")
print(f"期待: True（3段 >= 3段 なので飛び越えられる）")
print(f"注意: 3段目なので通常のジャンプ機能もあり、より遠くまで移動可能")

print("\n" + "="*60)
print("=== テスト7: 弓の前2マス・左右方向（飛び越え可能か）===")
board7 = Board()

# 黒の弓を (5, 5) に配置（1段目）
yumi7 = Piece(PieceType.YUMI, Player.BLACK)
board7.add_piece((5, 5), yumi7)

# 前1マス・左1の位置に駒を配置 (4, 4)
obstacle7 = Piece(PieceType.HYO, Player.BLACK)
board7.add_piece((4, 4), obstacle7)

print(f"(5,5): 黒の弓（1段）")
print(f"(4,4): 黒の兵（1段）- 前1マス・左1")
print()

legal_moves7 = Rules.get_legal_moves(board7, Player.BLACK, hand_pieces)
yumi_moves7 = [m for m in legal_moves7 if m.from_pos == (5, 5)]

print(f"弓の合法手: {len(yumi_moves7)} 手")
for move in yumi_moves7:
    print(f"  {move.from_pos} -> {move.to_pos}, タイプ: {move.move_type.name}")

can_move_to_34 = any(m.to_pos == (3, 4) for m in yumi_moves7)
print(f"\n(3,4)へ移動可能: {can_move_to_34}")
print(f"期待: True（前2マス・左1への移動。(4,4)は経路上にないので問題なし）")

print("\n" + "="*60)
print("=== テスト8: 敵の駒を飛び越えて攻撃可能か ===")
board8 = Board()

# 黒の弓を (5, 5) に配置（1段目）
yumi8 = Piece(PieceType.YUMI, Player.BLACK)
board8.add_piece((5, 5), yumi8)

# 前1マスに味方の駒 (4, 5)
ally8 = Piece(PieceType.HYO, Player.BLACK)
board8.add_piece((4, 5), ally8)

# 前2マスに敵の駒 (3, 5)
enemy8 = Piece(PieceType.HYO, Player.WHITE)
board8.add_piece((3, 5), enemy8)

print(f"(5,5): 黒の弓（1段）")
print(f"(4,5): 黒の兵（1段）- 前1マス")
print(f"(3,5): 白の兵（1段）- 前2マス")
print()

legal_moves8 = Rules.get_legal_moves(board8, Player.BLACK, hand_pieces)
yumi_moves8 = [m for m in legal_moves8 if m.from_pos == (5, 5)]

print(f"弓の合法手: {len(yumi_moves8)} 手")
for move in yumi_moves8:
    print(f"  {move.from_pos} -> {move.to_pos}, タイプ: {move.move_type.name}")

can_capture_35 = any(m.to_pos == (3, 5) and m.move_type.name == 'CAPTURE' for m in yumi_moves8)
print(f"\n(3,5)の敵駒を取れる: {can_capture_35}")
print(f"期待: True（味方を飛び越えて敵を攻撃可能）")
