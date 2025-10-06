"""
弓の周囲に駒がある場合の移動をテスト
"""

from src.engine.board import Board
from src.engine.piece import PieceType, Player, Piece
from src.engine.rules import Rules

print("=== テスト: 弓の周囲に駒がある場合 ===")
board = Board()

# 黒の弓を (5, 5) に配置（1段目）
yumi = Piece(PieceType.YUMI, Player.BLACK)
board.add_piece((5, 5), yumi)

# 前1マスに味方の駒を配置 (4, 5)
ally1 = Piece(PieceType.YUMI, Player.BLACK)
board.add_piece((4, 5), ally1)

# 前1マス・左に駒を配置 (4, 4)
ally2 = Piece(PieceType.HYO, Player.BLACK)
board.add_piece((4, 4), ally2)

# 前1マス・右に駒を配置 (4, 6)
ally3 = Piece(PieceType.HYO, Player.BLACK)
board.add_piece((4, 6), ally3)

print("盤面:")
print(f"(5,5): 黒の弓（1段）")
print(f"(4,5): 黒の弓（1段）- 前1マス・中央")
print(f"(4,4): 黒の兵（1段）- 前1マス・左")
print(f"(4,6): 黒の兵（1段）- 前1マス・右")
print()

hand_pieces = {}
legal_moves = Rules.get_legal_moves(board, Player.BLACK, hand_pieces)

# (5,5)の弓からの合法手
yumi_moves = [m for m in legal_moves if m.from_pos == (5, 5)]

print(f"(5,5)の弓の合法手: {len(yumi_moves)} 手")
for move in yumi_moves:
    row_diff = move.from_pos[0] - move.to_pos[0]
    col_diff = move.to_pos[1] - move.from_pos[1]
    direction = ""
    if row_diff == 2:
        if col_diff == -1:
            direction = "前2マス・左"
        elif col_diff == 0:
            direction = "前2マス・中央"
        elif col_diff == 1:
            direction = "前2マス・右"
    elif row_diff == -1:
        direction = "後ろ1マス"
    print(f"  {move.from_pos} -> {move.to_pos} ({direction}), タイプ: {move.move_type.name}")

print("\n分析:")
print("前1マス・左右に駒があっても、前2マス・左右への移動は")
print("経路上に障害物がないので可能なはずです。")
print()

can_move_to_34 = any(m.to_pos == (3, 4) for m in yumi_moves)
can_move_to_35 = any(m.to_pos == (3, 5) for m in yumi_moves)
can_move_to_36 = any(m.to_pos == (3, 6) for m in yumi_moves)

print(f"(3, 4) 前2マス・左: {'✅' if can_move_to_34 else '❌ ← これは移動可能なはず'}")
print(f"(3, 5) 前2マス・中央: {'✅' if can_move_to_35 else '❌'}")
print(f"(3, 6) 前2マス・右: {'✅' if can_move_to_36 else '❌ ← これは移動可能なはず'}")
