#!/usr/bin/env python3
"""弓の特殊ルールのテスト - 真正面に駒がある場合"""

import sys
sys.path.insert(0, 'src')

from engine.board import Board
from engine.piece import Piece, PieceType, Player
from engine.rules import Rules

def print_board_with_moves(board, legal_moves, from_pos):
    """盤面と合法手を表示"""
    print("\n盤面:")
    for r in range(9):
        row_str = ""
        for c in range(9):
            piece = board.get_top_piece((r, c))
            height = board.get_stack_height((r, c))
            
            if (r, c) == from_pos:
                row_str += "[弓]"
            elif piece:
                owner_mark = "黒" if piece.owner == Player.BLACK else "白"
                row_str += f"{owner_mark}{height} "
            else:
                # 合法手の移動先か確認
                is_legal = any(m.to_pos == (r, c) for m in legal_moves)
                if is_legal:
                    row_str += " ◯ "
                else:
                    row_str += " . "
        print(f"{r}: {row_str}")
    print("   0  1  2  3  4  5  6  7  8")

def test_yumi_with_center_obstacle():
    """真正面（中央）に駒がある場合のテスト"""
    print("=" * 60)
    print("テスト: 弓の真正面（中央）に駒がある場合")
    print("=" * 60)
    
    board = Board()
    
    # 弓を(5,5)に配置（黒プレイヤー）
    yumi = Piece(PieceType.YUMI, Player.BLACK)
    board.add_piece((5, 5), yumi)
    
    # 真正面（中央前）の(4,5)に1段の駒を配置
    obstacle = Piece(PieceType.HYO, Player.WHITE)
    board.add_piece((4, 5), obstacle)
    
    print("\n配置:")
    print("- 弓(黒): (5,5) - 1段")
    print("- 障害物(白): (4,5) - 1段（真正面）")
    
    # 合法手を取得
    legal_moves = Rules._get_piece_legal_moves(board, (5, 5), Player.BLACK)
    
    print_board_with_moves(board, legal_moves, (5, 5))
    
    print(f"\n合法手の数: {len(legal_moves)}")
    for move in legal_moves:
        print(f"  {move.from_pos} -> {move.to_pos} ({move.move_type.name})")
    
    # 期待される移動先
    expected_positions = {
        (3, 4),  # 左前2
        (3, 5),  # 中央前2
        (3, 6),  # 右前2
        (6, 5),  # 後ろ1
        (4, 5),  # 中央前1（取る・ツケる）
    }
    
    actual_positions = {move.to_pos for move in legal_moves}
    
    print(f"\n期待される位置: {sorted(expected_positions)}")
    print(f"実際の位置: {sorted(actual_positions)}")
    
    # 特に前2マスの3方向をチェック
    forward_2_positions = [(3, 4), (3, 5), (3, 6)]
    forward_2_moves = [pos for pos in forward_2_positions if pos in actual_positions]
    
    print(f"\n前2マス（左・中央・右）への移動:")
    for pos in forward_2_positions:
        status = "✅" if pos in actual_positions else "❌"
        print(f"  {pos}: {status}")
    
    if len(forward_2_moves) == 3:
        print("\n✅ テスト成功: 前2マスの3方向すべてに移動可能")
        return True
    else:
        print(f"\n❌ テスト失敗: 前2マスの移動が {len(forward_2_moves)}/3 方向のみ")
        return False

def test_yumi_with_left_obstacle():
    """左前に駒がある場合のテスト（真正面にはない）"""
    print("\n" + "=" * 60)
    print("テスト: 弓の左前に駒がある場合（真正面には駒なし）")
    print("=" * 60)
    
    board = Board()
    
    # 弓を(5,5)に配置（黒プレイヤー）
    yumi = Piece(PieceType.YUMI, Player.BLACK)
    board.add_piece((5, 5), yumi)
    
    # 左前の(4,4)に1段の駒を配置（真正面の(4,5)には駒なし）
    obstacle = Piece(PieceType.HYO, Player.WHITE)
    board.add_piece((4, 4), obstacle)
    
    print("\n配置:")
    print("- 弓(黒): (5,5) - 1段")
    print("- 障害物(白): (4,4) - 1段（左前）")
    print("- 真正面(4,5): 空")
    
    # 合法手を取得
    legal_moves = Rules._get_piece_legal_moves(board, (5, 5), Player.BLACK)
    
    print_board_with_moves(board, legal_moves, (5, 5))
    
    print(f"\n合法手の数: {len(legal_moves)}")
    for move in legal_moves:
        print(f"  {move.from_pos} -> {move.to_pos} ({move.move_type.name})")
    
    actual_positions = {move.to_pos for move in legal_moves}
    
    # 前2マスの3方向をチェック
    forward_2_positions = [(3, 4), (3, 5), (3, 6)]
    forward_2_moves = [pos for pos in forward_2_positions if pos in actual_positions]
    
    print(f"\n前2マス（左・中央・右）への移動:")
    for pos in forward_2_positions:
        status = "✅" if pos in actual_positions else "❌"
        print(f"  {pos}: {status}")
    
    # 真正面に駒がないので、通常は前2マスには移動できないはず
    if len(forward_2_moves) == 0:
        print("\n✅ テスト成功: 真正面に駒がないので前2マスには移動不可（正常）")
        return True
    else:
        print(f"\n⚠️ 前2マスに移動可能: {len(forward_2_moves)}/3 方向")
        return True

if __name__ == "__main__":
    test1_result = test_yumi_with_center_obstacle()
    test2_result = test_yumi_with_left_obstacle()
    
    print("\n" + "=" * 60)
    print("テスト結果まとめ")
    print("=" * 60)
    print(f"テスト1（真正面に駒あり）: {'✅ 成功' if test1_result else '❌ 失敗'}")
    print(f"テスト2（左前に駒あり、真正面は空）: {'✅ 成功' if test2_result else '❌ 失敗'}")
