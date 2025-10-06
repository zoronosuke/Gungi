"""
スタック（ツケ）機能のテスト
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine import Board, Player, PieceType, Piece, Rules, Move


def test_stack_movement():
    """スタックレベルによる駒の動きの変化をテスト"""
    print("=" * 60)
    print("スタック機能テスト - 駒の動きの変化")
    print("=" * 60)
    
    board = Board()
    
    # 兵を配置（1段目）
    pawn1 = Piece(PieceType.HYO, Player.BLACK)
    board.add_piece((4, 4), pawn1)
    
    print("1段目の兵の動き:")
    pattern1 = pawn1.get_move_pattern(1)
    print(f"  移動パターン: {pattern1['moves']}")
    print(f"  最大ステップ: {pattern1['maxSteps']}")
    print(f"  ジャンプ可能: {pattern1['canJump']}")
    
    # 別の兵を重ねる（2段目）
    pawn2 = Piece(PieceType.HYO, Player.BLACK)
    board.add_piece((4, 4), pawn2)
    
    print("\n2段目の兵の動き:")
    pattern2 = pawn2.get_move_pattern(2)
    print(f"  移動パターン: {pattern2['moves']}")
    print(f"  最大ステップ: {pattern2['maxSteps']}")
    print(f"  ジャンプ可能: {pattern2['canJump']}")
    
    # さらに重ねる（3段目）
    pawn3 = Piece(PieceType.HYO, Player.BLACK)
    board.add_piece((4, 4), pawn3)
    
    print("\n3段目の兵の動き（極）:")
    pattern3 = pawn3.get_move_pattern(3)
    print(f"  移動パターン: {pattern3['moves']}")
    print(f"  最大ステップ: {pattern3['maxSteps']}")
    print(f"  ジャンプ可能: {pattern3['canJump']}")
    
    print("\n盤面:")
    print(board)
    print()


def test_stack_rules():
    """スタックに関するルールのテスト"""
    print("=" * 60)
    print("スタックルールテスト")
    print("=" * 60)
    
    board = Board()
    
    # 帥を配置
    sui = Piece(PieceType.SUI, Player.BLACK)
    board.add_piece((4, 4), sui)
    
    # 帥の上に駒を乗せようとする（失敗するべき）
    pawn = Piece(PieceType.HYO, Player.BLACK)
    result = board.add_piece((4, 4), pawn)
    print(f"帥の上に兵を乗せる: {result} (Falseであるべき)")
    
    # 別の場所に兵を配置
    board.add_piece((5, 5), pawn)
    
    # 兵を3段まで積む
    pawn2 = Piece(PieceType.HYO, Player.BLACK)
    board.add_piece((5, 5), pawn2)
    pawn3 = Piece(PieceType.HYO, Player.BLACK)
    board.add_piece((5, 5), pawn3)
    
    # 4段目を乗せようとする（失敗するべき）
    pawn4 = Piece(PieceType.HYO, Player.BLACK)
    result = board.add_piece((5, 5), pawn4)
    print(f"4段目に兵を乗せる: {result} (Falseであるべき)")
    
    print("\n盤面:")
    print(board)
    print()


def test_stack_height_restriction():
    """自分より高いスタックには移動できないルールのテスト"""
    print("=" * 60)
    print("スタック高さ制限テスト")
    print("=" * 60)
    
    board = Board()
    
    # 1段目に兵を配置
    pawn1 = Piece(PieceType.HYO, Player.BLACK)
    board.add_piece((5, 5), pawn1)
    
    # 別の場所に3段積みを作成
    pawn2 = Piece(PieceType.HYO, Player.WHITE)
    board.add_piece((4, 5), pawn2)
    pawn3 = Piece(PieceType.HYO, Player.WHITE)
    board.add_piece((4, 5), pawn3)
    pawn4 = Piece(PieceType.HYO, Player.WHITE)
    board.add_piece((4, 5), pawn4)
    
    print("初期盤面:")
    print(board)
    print(f"(5,5)のスタック高さ: {board.get_stack_height((5, 5))}")
    print(f"(4,5)のスタック高さ: {board.get_stack_height((4, 5))}")
    
    # (5,5)から(4,5)への移動を試みる
    legal_moves = Rules.get_legal_moves(board, Player.BLACK)
    
    # (5,5)から(4,5)への移動が含まれていないことを確認
    moves_to_target = [m for m in legal_moves if m.from_pos == (5, 5) and m.to_pos == (4, 5)]
    
    print(f"\n(5,5)から(4,5)への移動可能な手: {len(moves_to_target)}個 (0であるべき)")
    
    # (5,5)からの全ての合法手を表示
    all_moves_from_55 = [m for m in legal_moves if m.from_pos == (5, 5)]
    print(f"(5,5)からの合法手: {len(all_moves_from_55)}個")
    for move in all_moves_from_55:
        print(f"  {move}")
    print()


def test_evolved_piece_moves():
    """進化した駒（2段目、3段目）の動きのテスト"""
    print("=" * 60)
    print("進化した駒の動きテスト")
    print("=" * 60)
    
    board = Board()
    
    # 帥を1段目に配置
    sui = Piece(PieceType.SUI, Player.BLACK)
    board.add_piece((5, 5), sui)
    
    print("1段目の帥の合法手:")
    legal_moves_1 = Rules._get_piece_legal_moves(board, (5, 5), Player.BLACK)
    print(f"  合法手数: {len(legal_moves_1)}")
    
    # 帥を2段目にする
    samurai = Piece(PieceType.SAMURAI, Player.BLACK)
    board.add_piece((5, 5), samurai)
    
    print("\n2段目の帥の合法手:")
    legal_moves_2 = Rules._get_piece_legal_moves(board, (5, 5), Player.BLACK)
    print(f"  合法手数: {len(legal_moves_2)}")
    
    # 帥を3段目にする
    pawn = Piece(PieceType.HYO, Player.BLACK)
    board.add_piece((5, 5), pawn)
    
    print("\n3段目の帥の合法手（極）:")
    legal_moves_3 = Rules._get_piece_legal_moves(board, (5, 5), Player.BLACK)
    print(f"  合法手数: {len(legal_moves_3)}")
    
    print(f"\n動きの拡大: 1段目({len(legal_moves_1)}) -> 2段目({len(legal_moves_2)}) -> 3段目({len(legal_moves_3)})")
    print()


if __name__ == "__main__":
    print("\n" + "*" * 60)
    print("軍儀スタック機能 テストスイート")
    print("*" * 60 + "\n")
    
    test_stack_movement()
    test_stack_rules()
    test_stack_height_restriction()
    test_evolved_piece_moves()
    
    print("*" * 60)
    print("すべてのテストが完了しました")
    print("*" * 60)
