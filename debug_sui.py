"""
帥の捕獲問題をデバッグ
"""

from src.engine.board import Board
from src.engine.piece import Piece, Player, PieceType
from src.engine.rules import Rules
from src.engine.move import Move, MoveType

def debug_sui_capture():
    """帥の捕獲をデバッグ"""
    print("=" * 60)
    print("帥の捕獲デバッグ")
    print("=" * 60)
    
    board = Board()
    
    # 両方の帥を配置
    sui_black = Piece(PieceType.SUI, Player.BLACK)
    sui_white = Piece(PieceType.SUI, Player.WHITE)
    board.add_piece((8, 4), sui_black)
    board.add_piece((0, 4), sui_white)
    
    # 黒の侍を配置（白の帥の隣）
    samurai = Piece(PieceType.SAMURAI, Player.BLACK)
    board.add_piece((1, 4), samurai)
    
    print("初期配置:")
    print(f"  黒の帥: (8, 4)")
    print(f"  白の帥: (0, 4)")
    print(f"  黒の侍: (1, 4)")
    
    # 合法手を取得
    legal_moves = Rules.get_legal_moves(board, Player.BLACK)
    sui_moves = [m for m in legal_moves if m.to_pos == (0, 4)]
    
    print(f"\n白の帥への攻撃手: {len(sui_moves)}個")
    for move in sui_moves:
        print(f"  - タイプ: {move.move_type.name}")
        print(f"    from: {move.from_pos} -> to: {move.to_pos}")
    
    # 捕獲手を実行
    if sui_moves:
        capture_move = [m for m in sui_moves if m.move_type == MoveType.CAPTURE][0]
        print(f"\n捕獲を実行...")
        print(f"  移動前の白の帥の位置: {board.get_sui_position(Player.WHITE)}")
        print(f"  移動前の(0,4)の駒: {board.get_top_piece((0, 4))}")
        
        # apply_moveを呼び出す
        success, captured = Rules.apply_move(board, capture_move)
        
        print(f"\n結果:")
        print(f"  成功: {success}")
        print(f"  捕獲した駒: {captured}")
        print(f"  移動後の白の帥の位置: {board.get_sui_position(Player.WHITE)}")
        print(f"  移動後の(0,4)の駒: {board.get_top_piece((0, 4))}")
        print(f"  移動後の(1,4)の駒: {board.get_top_piece((1, 4))}")
        
        # ゲーム終了判定
        is_over, winner = Rules.is_game_over(board)
        print(f"\nゲーム終了: {is_over}")
        if winner:
            print(f"勝者: {winner.name}")
    else:
        print("\n警告: 白の帥への攻撃手が見つかりません！")


if __name__ == "__main__":
    debug_sui_capture()
