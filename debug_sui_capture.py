"""帥の捕獲のデバッグスクリプト"""
from src.engine import Board, Player, PieceType, Piece, Rules, Move

# 空の盤面を作成
board = Board()

# 黒のDAIと白のSUIを配置
board.add_piece((4, 4), Piece(PieceType.DAI, Player.BLACK))
board.add_piece((4, 5), Piece(PieceType.SUI, Player.WHITE))

print("初期状態:")
print(f"黒の帥の位置: {board.get_sui_position(Player.BLACK)}")
print(f"白の帥の位置: {board.get_sui_position(Player.WHITE)}")

# 白の帥を取得
move = Move.create_capture_move(
    from_pos=(4, 4),
    to_pos=(4, 5),
    player=Player.BLACK
)

print("\n移動を適用...")
success, captured = Rules.apply_move(board, move)
print(f"移動成功: {success}")
print(f"捕獲した駒: {captured}")

print("\n移動後:")
print(f"黒の帥の位置: {board.get_sui_position(Player.BLACK)}")
print(f"白の帥の位置: {board.get_sui_position(Player.WHITE)}")

# ゲーム終了判定
is_over, winner = Rules.is_game_over(board)
print(f"\nゲーム終了: {is_over}")
print(f"勝者: {winner}")
print(f"期待される勝者: {Player.BLACK}")
