"""
全ての駒が1段、2段、3段で正しく動くかテストする
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.board import Board
from src.engine.piece import Piece, PieceType, Player
from src.engine.rules import Rules

def test_piece_movement(piece_type: PieceType, stack_level: int):
    """指定の駒タイプとスタックレベルで動きをテスト"""
    board = Board()
    player = Player.BLACK
    
    # 中央に駒を配置
    test_pos = (4, 4)
    
    # スタックレベルに応じて駒を配置
    for i in range(stack_level):
        piece = Piece(piece_type, player)
        success = board.add_piece(test_pos, piece)
        if not success:
            print(f"  ⚠️  駒の追加に失敗: {piece_type.name} (レベル{i+1})")
            return False
    
    # 合法手を取得
    legal_moves = Rules._get_piece_legal_moves(board, test_pos, player)
    
    # 動きがあるか確認
    move_count = len(legal_moves)
    
    if move_count == 0 and piece_type != PieceType.TORIDE:
        # 砦以外で動きがない場合はエラー（砦は1段では動けない可能性がある）
        print(f"  ❌ {piece_type.name} (レベル{stack_level}): 動きなし")
        return False
    else:
        print(f"  ✓ {piece_type.name} (レベル{stack_level}): {move_count}手")
        return True

def visualize_moves(piece_type: PieceType, stack_level: int):
    """駒の動きを視覚的に表示"""
    board = Board()
    player = Player.BLACK
    
    # 中央に駒を配置
    test_pos = (4, 4)
    
    # スタックレベルに応じて駒を配置
    for i in range(stack_level):
        piece = Piece(piece_type, player)
        board.add_piece(test_pos, piece)
    
    # 合法手を取得
    legal_moves = Rules._get_piece_legal_moves(board, test_pos, player)
    legal_positions = set()
    for move in legal_moves:
        if hasattr(move, 'to_pos'):
            legal_positions.add(move.to_pos)
    
    print(f"\n{piece_type.name} (スタックレベル{stack_level}):")
    print("  1 2 3 4 5 6 7 8 9")
    for row in range(9):
        row_label = chr(65 + row)  # A-I
        print(f"{row_label} ", end='')
        for col in range(9):
            if (row, col) == test_pos:
                print("★ ", end='')
            elif (row, col) in legal_positions:
                print("○ ", end='')
            else:
                print("· ", end='')
        print()

def main():
    print("=" * 60)
    print("全駒の動きテスト")
    print("=" * 60)
    
    all_piece_types = [
        PieceType.SUI,
        PieceType.DAI,
        PieceType.CHUU,
        PieceType.SHO,
        PieceType.SAMURAI,
        PieceType.HYO,
        PieceType.UMA,
        PieceType.SHINOBI,
        PieceType.YARI,
        PieceType.TORIDE,
        PieceType.YUMI,
        PieceType.TSUTU,
        PieceType.HOU,
        PieceType.BOU
    ]
    
    print("\n1. 基本テスト（動きがあるか確認）")
    print("-" * 60)
    
    for piece_type in all_piece_types:
        print(f"\n{piece_type.name}:")
        for level in [1, 2, 3]:
            test_piece_movement(piece_type, level)
    
    print("\n" + "=" * 60)
    print("2. 視覚的表示（いくつかの駒を選んで表示）")
    print("=" * 60)
    
    # 動きが少ない駒や特徴的な駒を表示
    test_pieces = [
        (PieceType.HYO, "兵"),
        (PieceType.UMA, "馬"),
        (PieceType.TORIDE, "砦"),
        (PieceType.SUI, "帥"),
        (PieceType.DAI, "大"),
    ]
    
    for piece_type, name in test_pieces:
        for level in [1, 2, 3]:
            visualize_moves(piece_type, level)

if __name__ == "__main__":
    main()
