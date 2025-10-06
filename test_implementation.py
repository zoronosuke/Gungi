"""
実装確認用テストスクリプト
"""

from src.engine.board import Board
from src.engine.piece import Piece, Player, PieceType
from src.engine.rules import Rules
from src.engine.move import Move

def test_uma_movement():
    """馬の動きをテスト（上下2マス、左右1マス）"""
    print("=" * 60)
    print("テスト1: 馬の動き")
    print("=" * 60)
    
    board = Board()
    # 馬を中央に配置
    uma = Piece(PieceType.UMA, Player.BLACK)
    board.add_piece((4, 4), uma)
    
    legal_moves = Rules.get_legal_moves(board, Player.BLACK)
    
    print(f"馬の位置: (4, 4)")
    print(f"合法手の数: {len(legal_moves)}")
    
    expected_positions = [
        (3, 4),  # 上1マス
        (2, 4),  # 上2マス
        (5, 4),  # 下1マス
        (6, 4),  # 下2マス
        (4, 3),  # 左1マス
        (4, 5),  # 右1マス
    ]
    
    move_positions = [(m.to_pos[0], m.to_pos[1]) for m in legal_moves]
    
    print("期待される移動先:")
    for pos in expected_positions:
        status = "✓" if pos in move_positions else "✗"
        print(f"  {status} {pos}")
    
    print()
    return len(expected_positions) == len(legal_moves)


def test_stack_level_capture():
    """スタックレベルによる捕獲制限をテスト"""
    print("=" * 60)
    print("テスト2: スタックレベルによる捕獲制限")
    print("=" * 60)
    
    board = Board()
    
    # テストケース1: 同じレベル（1段目同士）
    hyo1 = Piece(PieceType.HYO, Player.BLACK)
    hyo2 = Piece(PieceType.HYO, Player.WHITE)
    board.add_piece((4, 4), hyo1)
    board.add_piece((3, 4), hyo2)
    
    moves = Rules.get_legal_moves(board, Player.BLACK)
    capture_moves = [m for m in moves if m.from_pos == (4, 4) and m.to_pos == (3, 4)]
    
    print("ケース1: 1段目の黒兵 vs 1段目の白兵")
    print(f"  捕獲可能: {len([m for m in capture_moves if m.move_type.name == 'CAPTURE']) > 0} (期待: True)")
    print(f"  ツケ可能: {len([m for m in capture_moves if m.move_type.name == 'STACK']) > 0} (期待: True)")
    
    # テストケース2: スタックを作る
    board2 = Board()
    samurai1 = Piece(PieceType.SAMURAI, Player.BLACK)
    samurai2 = Piece(PieceType.SAMURAI, Player.BLACK)
    hyo3 = Piece(PieceType.HYO, Player.WHITE)
    
    board2.add_piece((4, 4), samurai1)
    board2.add_piece((4, 4), samurai2)  # 2段目
    board2.add_piece((3, 4), hyo3)  # 1段目の敵
    
    print("\nケース2: 2段目の黒侍 vs 1段目の白兵")
    moves2 = Rules.get_legal_moves(board2, Player.BLACK)
    capture_moves2 = [m for m in moves2 if m.from_pos == (4, 4) and m.to_pos == (3, 4)]
    
    print(f"  スタックレベル: 黒=2, 白=1")
    print(f"  捕獲可能: {len([m for m in capture_moves2 if m.move_type.name == 'CAPTURE']) > 0} (期待: True)")
    print(f"  ツケ可能: {len([m for m in capture_moves2 if m.move_type.name == 'STACK']) > 0} (期待: True)")
    
    # テストケース3: 低い位置から高い位置への攻撃
    board3 = Board()
    samurai3 = Piece(PieceType.SAMURAI, Player.WHITE)
    samurai4 = Piece(PieceType.SAMURAI, Player.WHITE)
    hyo4 = Piece(PieceType.HYO, Player.BLACK)
    
    board3.add_piece((3, 4), samurai3)
    board3.add_piece((3, 4), samurai4)  # 2段目の白
    board3.add_piece((4, 4), hyo4)  # 1段目の黒
    
    print("\nケース3: 1段目の黒兵 vs 2段目の白侍")
    moves3 = Rules.get_legal_moves(board3, Player.BLACK)
    capture_moves3 = [m for m in moves3 if m.from_pos == (4, 4) and m.to_pos == (3, 4)]
    
    print(f"  スタックレベル: 黒=1, 白=2")
    print(f"  捕獲可能: {len([m for m in capture_moves3 if m.move_type.name == 'CAPTURE']) > 0} (期待: False)")
    print(f"  ツケ可能: {len([m for m in capture_moves3 if m.move_type.name == 'STACK']) > 0} (期待: True)")
    
    print()


def test_sui_capture_and_victory():
    """帥の捕獲と勝利条件をテスト"""
    print("=" * 60)
    print("テスト3: 帥の捕獲と勝利条件")
    print("=" * 60)
    
    board = Board()
    
    # 両方の帥を配置（ゲームの前提条件）
    sui_black = Piece(PieceType.SUI, Player.BLACK)
    sui_white = Piece(PieceType.SUI, Player.WHITE)
    board.add_piece((8, 4), sui_black)
    board.add_piece((0, 4), sui_white)
    
    # 黒の侍を配置
    samurai = Piece(PieceType.SAMURAI, Player.BLACK)
    board.add_piece((1, 4), samurai)
    
    print("初期状態:")
    print(f"  黒の帥: (8, 4)")
    print(f"  白の帥: (0, 4)")
    print(f"  黒の侍: (1, 4)")
    
    # ゲーム終了チェック（まだ終わっていない）
    is_over, winner = Rules.is_game_over(board)
    print(f"\nゲーム終了: {is_over} (期待: False)")
    
    # 黒の侍の合法手を取得
    moves = Rules.get_legal_moves(board, Player.BLACK)
    sui_capture_moves = [m for m in moves if m.to_pos == (0, 4)]
    
    print(f"\n帥への攻撃手:")
    for move in sui_capture_moves:
        print(f"  - {move.move_type.name}: {move.from_pos} -> {move.to_pos}")
    
    # 帥を取る
    if sui_capture_moves:
        capture_move = [m for m in sui_capture_moves if m.move_type.name == 'CAPTURE'][0]
        success, captured = Rules.apply_move(board, capture_move)
        print(f"\n帥を捕獲: {success}")
        
        # ゲーム終了チェック
        is_over, winner = Rules.is_game_over(board)
        print(f"ゲーム終了: {is_over} (期待: True)")
        print(f"勝者: {winner.name if winner else None} (期待: BLACK)")
    
    print()


def main():
    """メインテスト実行"""
    print("\n軍儀実装確認テスト\n")
    
    try:
        test_uma_movement()
        test_stack_level_capture()
        test_sui_capture_and_victory()
        
        print("=" * 60)
        print("すべてのテスト完了")
        print("=" * 60)
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
