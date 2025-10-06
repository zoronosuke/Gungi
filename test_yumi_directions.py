"""
弓の前2マス（左・中央・右）の移動をテスト
"""

from src.engine.board import Board
from src.engine.piece import PieceType, Player, Piece
from src.engine.rules import Rules

print("=== テスト: 弓の前2マス（左・中央・右）への移動 ===")
board = Board()

# 黒の弓を (5, 5) に配置（1段目）
yumi = Piece(PieceType.YUMI, Player.BLACK)
board.add_piece((5, 5), yumi)

# 前1マスに味方の弓を配置 (4, 5)
ally_yumi = Piece(PieceType.YUMI, Player.BLACK)
board.add_piece((4, 5), ally_yumi)

print(f"(5,5): 黒の弓（1段）")
print(f"(4,5): 黒の弓（1段）- 前1マス（味方）")
print()

hand_pieces = {}
legal_moves = Rules.get_legal_moves(board, Player.BLACK, hand_pieces)

# (5,5)の弓からの合法手
yumi_moves = [m for m in legal_moves if m.from_pos == (5, 5)]

print(f"(5,5)の弓の合法手: {len(yumi_moves)} 手")
for move in yumi_moves:
    print(f"  {move.from_pos} -> {move.to_pos}, タイプ: {move.move_type.name}")

print("\n期待される移動先:")
print("- (3, 4): 前2マス・左")
print("- (3, 5): 前2マス・中央（前1マスに味方がいるが、飛び越えて移動可能）")
print("- (3, 6): 前2マス・右")
print("- (6, 5): 後ろ1マス")

print("\n実際の移動可能箇所:")
can_move_to_34 = any(m.to_pos == (3, 4) for m in yumi_moves)
can_move_to_35 = any(m.to_pos == (3, 5) for m in yumi_moves)
can_move_to_36 = any(m.to_pos == (3, 6) for m in yumi_moves)
can_move_to_65 = any(m.to_pos == (6, 5) for m in yumi_moves)

print(f"(3, 4) 前2マス・左: {'✅' if can_move_to_34 else '❌'}")
print(f"(3, 5) 前2マス・中央: {'✅' if can_move_to_35 else '❌'}")
print(f"(3, 6) 前2マス・右: {'✅' if can_move_to_36 else '❌'}")
print(f"(6, 5) 後ろ1マス: {'✅' if can_move_to_65 else '✅'}")
