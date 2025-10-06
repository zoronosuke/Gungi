"""
実際のゲームシナリオで駒の動きをテスト
スタックレベルの変化を確認
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.board import Board
from src.engine.piece import Piece, PieceType, Player
from src.engine.rules import Rules

def visualize_board(board: Board, legal_positions=None):
    """盤面を視覚的に表示"""
    if legal_positions is None:
        legal_positions = set()
    
    print("  1 2 3 4 5 6 7 8 9")
    for row in range(9):
        row_label = chr(65 + row)  # A-I
        print(f"{row_label} ", end='')
        for col in range(9):
            piece = board.get_top_piece((row, col))
            stack_level = board.get_top_piece_stack_level((row, col))
            
            if piece:
                from src.engine.piece import PIECE_NAMES
                piece_name = PIECE_NAMES[piece.piece_type]
                if stack_level > 1:
                    # スタックレベルを表示
                    print(f"{piece_name}{stack_level}", end='')
                else:
                    print(f"{piece_name} ", end='')
            elif (row, col) in legal_positions:
                print("○ ", end='')
            else:
                print("· ", end='')
        print()

def test_stacking_scenario():
    """スタック時の動きの変化をテスト"""
    print("=" * 60)
    print("スタック時の動き変化テスト")
    print("=" * 60)
    
    # シナリオ1: 兵を重ねる
    print("\n【シナリオ1】兵を重ねていく")
    print("-" * 60)
    
    board = Board()
    test_pos = (4, 4)
    player = Player.BLACK
    
    # 1段目：兵を配置
    print("\n1段目の兵:")
    piece1 = Piece(PieceType.HYO, player)
    board.add_piece(test_pos, piece1)
    
    legal_moves = Rules._get_piece_legal_moves(board, test_pos, player)
    legal_positions = {move.to_pos for move in legal_moves}
    print(f"移動可能な手数: {len(legal_moves)}")
    visualize_board(board, legal_positions)
    
    # 2段目：もう1つ兵を重ねる
    print("\n2段目の兵:")
    piece2 = Piece(PieceType.HYO, player)
    board.add_piece(test_pos, piece2)
    
    legal_moves = Rules._get_piece_legal_moves(board, test_pos, player)
    legal_positions = {move.to_pos for move in legal_moves}
    print(f"移動可能な手数: {len(legal_moves)}")
    visualize_board(board, legal_positions)
    
    # 3段目：さらに兵を重ねる
    print("\n3段目の兵:")
    piece3 = Piece(PieceType.HYO, player)
    board.add_piece(test_pos, piece3)
    
    legal_moves = Rules._get_piece_legal_moves(board, test_pos, player)
    legal_positions = {move.to_pos for move in legal_moves}
    print(f"移動可能な手数: {len(legal_moves)}")
    visualize_board(board, legal_positions)
    
    # シナリオ2: 馬のスタック
    print("\n\n【シナリオ2】馬を重ねていく")
    print("-" * 60)
    
    board2 = Board()
    test_pos2 = (4, 4)
    
    # 1段目：馬を配置
    print("\n1段目の馬:")
    uma1 = Piece(PieceType.UMA, player)
    board2.add_piece(test_pos2, uma1)
    
    legal_moves = Rules._get_piece_legal_moves(board2, test_pos2, player)
    legal_positions = {move.to_pos for move in legal_moves}
    print(f"移動可能な手数: {len(legal_moves)}")
    visualize_board(board2, legal_positions)
    
    # 2段目：もう1つ馬を重ねる
    print("\n2段目の馬:")
    uma2 = Piece(PieceType.UMA, player)
    board2.add_piece(test_pos2, uma2)
    
    legal_moves = Rules._get_piece_legal_moves(board2, test_pos2, player)
    legal_positions = {move.to_pos for move in legal_moves}
    print(f"移動可能な手数: {len(legal_moves)}")
    visualize_board(board2, legal_positions)
    
    # 3段目：さらに馬を重ねる
    print("\n3段目の馬 (ジャンプ可能):")
    uma3 = Piece(PieceType.UMA, player)
    board2.add_piece(test_pos2, uma3)
    
    legal_moves = Rules._get_piece_legal_moves(board2, test_pos2, player)
    legal_positions = {move.to_pos for move in legal_moves}
    print(f"移動可能な手数: {len(legal_moves)}")
    visualize_board(board2, legal_positions)
    
    # シナリオ3: 異なる駒を重ねる
    print("\n\n【シナリオ3】異なる駒を重ねる（砦の上に兵）")
    print("-" * 60)
    
    board3 = Board()
    test_pos3 = (4, 4)
    
    # 1段目：砦を配置
    print("\n1段目: 砦")
    toride = Piece(PieceType.TORIDE, player)
    board3.add_piece(test_pos3, toride)
    
    legal_moves = Rules._get_piece_legal_moves(board3, test_pos3, player)
    legal_positions = {move.to_pos for move in legal_moves}
    print(f"砦の移動可能な手数: {len(legal_moves)}")
    visualize_board(board3, legal_positions)
    
    # 2段目：兵を砦の上に重ねる
    print("\n2段目: 砦の上に兵を重ねる")
    hyo = Piece(PieceType.HYO, player)
    board3.add_piece(test_pos3, hyo)
    
    legal_moves = Rules._get_piece_legal_moves(board3, test_pos3, player)
    legal_positions = {move.to_pos for move in legal_moves}
    print(f"兵(2段目)の移動可能な手数: {len(legal_moves)}")
    visualize_board(board3, legal_positions)

if __name__ == "__main__":
    test_stacking_scenario()
