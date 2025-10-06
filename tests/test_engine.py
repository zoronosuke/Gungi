"""
軍儀エンジンのテスト
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine import Board, Player, PieceType, Piece, Rules, Move
from src.engine.initial_setup import load_initial_board


def test_board_creation():
    """盤面作成のテスト"""
    print("=" * 60)
    print("盤面作成テスト")
    print("=" * 60)
    
    board = Board()
    print("空の盤面を作成しました")
    print(board)
    print()


def test_initial_setup():
    """初期配置のテスト"""
    print("=" * 60)
    print("初期配置テスト")
    print("=" * 60)
    
    board = load_initial_board()
    print("公式初期配置を読み込みました")
    print(board)
    print()
    
    # 帥の位置確認
    black_sui = board.get_sui_position(Player.BLACK)
    white_sui = board.get_sui_position(Player.WHITE)
    print(f"先手（黒）の帥の位置: {black_sui}")
    print(f"後手（白）の帥の位置: {white_sui}")
    print()


def test_legal_moves():
    """合法手生成のテスト"""
    print("=" * 60)
    print("合法手生成テスト")
    print("=" * 60)
    
    board = load_initial_board()
    legal_moves = Rules.get_legal_moves(board, Player.BLACK)
    
    print(f"先手（黒）の合法手数: {len(legal_moves)}")
    print("最初の5手:")
    for i, move in enumerate(legal_moves[:5]):
        print(f"  {i+1}. {move}")
    print()


def test_move_application():
    """手の適用テスト"""
    print("=" * 60)
    print("手の適用テスト")
    print("=" * 60)
    
    board = load_initial_board()
    
    # 駒を動かす（例: 兵を前進）
    move = Move.create_normal_move(
        from_pos=(6, 0),  # 7段1列の兵
        to_pos=(5, 0),    # 6段1列へ移動
        player=Player.BLACK
    )
    
    print(f"手を適用: {move}")
    success = Rules.apply_move(board, move)
    print(f"結果: {'成功' if success else '失敗'}")
    print()
    print("移動後の盤面:")
    print(board)
    print()


def test_game_flow():
    """ゲームフローのテスト"""
    print("=" * 60)
    print("ゲームフローテスト")
    print("=" * 60)
    
    board = load_initial_board()
    current_player = Player.BLACK
    move_count = 0
    max_moves = 5
    
    print(f"最初の{max_moves}手をシミュレーション")
    print()
    
    while move_count < max_moves:
        legal_moves = Rules.get_legal_moves(board, current_player)
        
        if not legal_moves:
            print("合法手がありません")
            break
        
        # 最初の合法手を選択
        move = legal_moves[0]
        
        print(f"{move_count + 1}手目: {current_player.name}")
        print(f"  手: {move}")
        
        Rules.apply_move(board, move)
        
        # ゲーム終了判定
        is_over, winner = Rules.is_game_over(board)
        if is_over:
            print(f"ゲーム終了！勝者: {winner.name}")
            break
        
        # 手番交代
        current_player = current_player.opponent
        move_count += 1
    
    print()
    print("最終盤面:")
    print(board)
    print()


if __name__ == "__main__":
    print("\n")
    print("*" * 60)
    print("軍儀エンジン テストスイート")
    print("*" * 60)
    print("\n")
    
    test_board_creation()
    test_initial_setup()
    test_legal_moves()
    test_move_application()
    test_game_flow()
    
    print("*" * 60)
    print("すべてのテストが完了しました")
    print("*" * 60)
