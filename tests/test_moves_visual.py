"""
駒の動きとスタックの詳細テスト
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine import Board, Player, PieceType, Piece, Rules
from src.engine.initial_setup import load_initial_board


def test_piece_moves_detailed():
    """各駒の動きを詳細に確認"""
    print("=" * 60)
    print("駒の動きの詳細確認")
    print("=" * 60)
    
    board = Board()
    
    # 兵を中央に配置
    pawn = Piece(PieceType.HYO, Player.BLACK)
    board.add_piece((4, 4), pawn)
    
    print("\n【1段目の兵 (4,4)】")
    print(board)
    
    legal_moves = Rules._get_piece_legal_moves(board, (4, 4), Player.BLACK)
    print(f"合法手数: {len(legal_moves)}")
    for move in legal_moves:
        print(f"  {move}")
    
    # 2段目にする
    pawn2 = Piece(PieceType.HYO, Player.BLACK)
    board.add_piece((4, 4), pawn2)
    
    print("\n【2段目の兵 (4,4)】")
    print(board)
    
    legal_moves = Rules._get_piece_legal_moves(board, (4, 4), Player.BLACK)
    print(f"合法手数: {len(legal_moves)}")
    for move in legal_moves:
        print(f"  {move}")
    
    # 3段目にする
    pawn3 = Piece(PieceType.HYO, Player.BLACK)
    board.add_piece((4, 4), pawn3)
    
    print("\n【3段目の兵 (4,4)】")
    print(board)
    
    legal_moves = Rules._get_piece_legal_moves(board, (4, 4), Player.BLACK)
    print(f"合法手数: {len(legal_moves)}")
    for move in legal_moves:
        print(f"  {move}")
    print()


def test_stacking_restrictions():
    """スタック制限の詳細テスト"""
    print("=" * 60)
    print("スタック制限の確認")
    print("=" * 60)
    
    board = Board()
    
    # 1段の兵を (5,5) に
    pawn1 = Piece(PieceType.HYO, Player.BLACK)
    board.add_piece((5, 5), pawn1)
    
    # 2段の兵を (4,5) に
    enemy1 = Piece(PieceType.HYO, Player.WHITE)
    board.add_piece((4, 5), enemy1)
    enemy2 = Piece(PieceType.HYO, Player.WHITE)
    board.add_piece((4, 5), enemy2)
    
    # 3段の兵を (3,5) に
    enemy3 = Piece(PieceType.HYO, Player.WHITE)
    board.add_piece((3, 5), enemy3)
    enemy4 = Piece(PieceType.HYO, Player.WHITE)
    board.add_piece((3, 5), enemy4)
    enemy5 = Piece(PieceType.HYO, Player.WHITE)
    board.add_piece((3, 5), enemy5)
    
    print("\n初期配置:")
    print(f"  (5,5): 1段 (BLACK)")
    print(f"  (4,5): 2段 (WHITE)")
    print(f"  (3,5): 3段 (WHITE)")
    print(board)
    
    # (5,5)の兵の合法手
    legal_moves = Rules._get_piece_legal_moves(board, (5, 5), Player.BLACK)
    
    print(f"\n(5,5)の兵の合法手: {len(legal_moves)}個")
    for move in legal_moves:
        to_pos = move.to_pos
        to_height = board.get_stack_height(to_pos)
        print(f"  {move} -> スタック高さ: {to_height}")
    
    # (4,5)や(3,5)への移動が含まれていないことを確認
    moves_to_45 = [m for m in legal_moves if m.to_pos == (4, 5)]
    moves_to_35 = [m for m in legal_moves if m.to_pos == (3, 5)]
    
    print(f"\n(5,5) -> (4,5)への手: {len(moves_to_45)}個 (0であるべき)")
    print(f"(5,5) -> (3,5)への手: {len(moves_to_35)}個 (0であるべき)")
    print()


def test_initial_board_moves():
    """初期盤面からの動きを確認"""
    print("=" * 60)
    print("初期盤面での動きの確認")
    print("=" * 60)
    
    board = load_initial_board()
    
    print("\n初期盤面:")
    print(board)
    
    # いくつかの駒の動きを確認
    test_positions = [
        ((6, 0), "兵 (6,0)"),
        ((6, 2), "砦 (6,2)"),
        ((7, 1), "馬 (7,1)"),
        ((8, 4), "帥 (8,4)"),
    ]
    
    for pos, name in test_positions:
        piece = board.get_top_piece(pos)
        if piece:
            legal_moves = Rules._get_piece_legal_moves(board, pos, Player.BLACK)
            print(f"\n{name} の合法手: {len(legal_moves)}個")
            for move in legal_moves:
                print(f"  {move}")


def test_direction_reversal():
    """プレイヤーによる方向の反転を確認"""
    print("=" * 60)
    print("プレイヤー方向の反転確認")
    print("=" * 60)
    
    board = Board()
    
    # 黒の兵を (5,5) に
    black_pawn = Piece(PieceType.HYO, Player.BLACK)
    board.add_piece((5, 5), black_pawn)
    
    # 白の兵を (3,5) に
    white_pawn = Piece(PieceType.HYO, Player.WHITE)
    board.add_piece((3, 5), white_pawn)
    
    print("\n配置:")
    print(board)
    
    print("\n黒の兵 (5,5) の動き:")
    black_moves = Rules._get_piece_legal_moves(board, (5, 5), Player.BLACK)
    for move in black_moves:
        print(f"  {move}")
    
    print("\n白の兵 (3,5) の動き:")
    white_moves = Rules._get_piece_legal_moves(board, (3, 5), Player.WHITE)
    for move in white_moves:
        print(f"  {move}")
    print()


def visualize_moves(board: Board, pos: tuple, player: Player):
    """指定位置の駒が動ける場所を盤面上に表示"""
    legal_moves = Rules._get_piece_legal_moves(board, pos, player)
    move_positions = set(m.to_pos for m in legal_moves)
    
    print("\n盤面（○=移動可能、★=選択中の駒）:")
    print("     0   |  1   |  2   |  3   |  4   |  5   |  6   |  7   |  8   |")
    print("  " + "-" * 64)
    
    for row in range(9):
        row_str = f"{row} |"
        for col in range(9):
            current_pos = (row, col)
            stack = board.get_stack(current_pos)
            
            if current_pos == pos:
                cell = "  ★  "
            elif current_pos in move_positions:
                if stack.is_empty():
                    cell = "  ○  "
                else:
                    cell = " ○駒 "
            elif stack.is_empty():
                cell = "      "
            else:
                piece_str = str(stack)
                cell = f"{piece_str:^6}"
            
            row_str += cell + "|"
        print(row_str)
        print("  " + "-" * 64)


def test_visual_moves():
    """動ける範囲を視覚的に表示"""
    print("=" * 60)
    print("動ける範囲の可視化")
    print("=" * 60)
    
    # テスト1: 1段目の兵
    print("\n【テスト1: 1段目の兵】")
    board = Board()
    pawn1 = Piece(PieceType.HYO, Player.BLACK)
    board.add_piece((4, 4), pawn1)
    visualize_moves(board, (4, 4), Player.BLACK)
    
    # テスト2: 2段目の兵
    print("\n【テスト2: 2段目の兵】")
    board = Board()
    pawn1 = Piece(PieceType.HYO, Player.BLACK)
    board.add_piece((4, 4), pawn1)
    pawn2 = Piece(PieceType.HYO, Player.BLACK)
    board.add_piece((4, 4), pawn2)
    visualize_moves(board, (4, 4), Player.BLACK)
    
    # テスト3: 3段目の兵
    print("\n【テスト3: 3段目の兵（極）】")
    board = Board()
    pawn1 = Piece(PieceType.HYO, Player.BLACK)
    board.add_piece((4, 4), pawn1)
    pawn2 = Piece(PieceType.HYO, Player.BLACK)
    board.add_piece((4, 4), pawn2)
    pawn3 = Piece(PieceType.HYO, Player.BLACK)
    board.add_piece((4, 4), pawn3)
    visualize_moves(board, (4, 4), Player.BLACK)
    
    # テスト4: 障害物がある場合
    print("\n【テスト4: 2段目の兵（障害物あり）】")
    board = Board()
    pawn1 = Piece(PieceType.HYO, Player.BLACK)
    board.add_piece((4, 4), pawn1)
    pawn2 = Piece(PieceType.HYO, Player.BLACK)
    board.add_piece((4, 4), pawn2)
    
    # 障害物を配置
    enemy = Piece(PieceType.HYO, Player.WHITE)
    board.add_piece((3, 4), enemy)
    board.add_piece((5, 4), enemy)
    
    visualize_moves(board, (4, 4), Player.BLACK)


if __name__ == "__main__":
    print("\n" + "*" * 60)
    print("駒の動きとスタック 詳細テスト")
    print("*" * 60 + "\n")
    
    test_piece_moves_detailed()
    test_stacking_restrictions()
    test_initial_board_moves()
    test_direction_reversal()
    test_visual_moves()
    
    print("*" * 60)
    print("テスト完了")
    print("*" * 60)
