"""
弓が前1マスに移動できないことを確認するテスト
"""

from src.engine.board import Board
from src.engine.piece import PieceType, Player, Piece
from src.engine.rules import Rules

print("=== テスト: 弓は前1マスに移動できない ===")
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

# (4,5)への移動があるかチェック
can_move_to_45 = any(m.to_pos == (4, 5) for m in yumi_moves)
print(f"\n(4,5)へ移動可能: {can_move_to_45}")
print(f"期待: False（弓は前1マスに移動できない）")

if can_move_to_45:
    print("\n❌ エラー: 弓が前1マスに移動できてしまっている")
    moves_to_45 = [m for m in yumi_moves if m.to_pos == (4, 5)]
    for m in moves_to_45:
        print(f"  不正な手: {m.move_type.name}")
else:
    print("\n✅ 正しい: 弓は前1マスに移動できない")
