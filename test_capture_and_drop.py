"""
軍儀の駒の取得と配置ルールのテスト
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from engine.board import Board
from engine.piece import Piece, Player, PieceType
from engine.move import Move, MoveType
from engine.rules import Rules


def test_capture_entire_stack():
    """駒を取るとスタック全体が除去されることをテスト"""
    print("=" * 60)
    print("テスト1: 駒を取るとスタック全体が除去される")
    print("=" * 60)
    
    board = Board()
    
    # 白のスタックを作成 (5, 4): HYO -> YARI -> UMA
    board.add_piece((5, 4), Piece(PieceType.HYO, Player.WHITE))
    board.add_piece((5, 4), Piece(PieceType.YARI, Player.WHITE))
    board.add_piece((5, 4), Piece(PieceType.UMA, Player.WHITE))
    
    # 黒の駒を配置
    board.add_piece((7, 4), Piece(PieceType.DAI, Player.BLACK))
    
    print("\n初期状態:")
    print(board)
    print(f"\n(5, 4)のスタック高さ: {board.get_stack_height((5, 4))}")
    print(f"(5, 4)の駒: {board.get_stack((5, 4))}")
    
    # 黒が白のスタック全体を取る
    move = Move.create_capture_move(from_pos=(7, 4), to_pos=(5, 4), player=Player.BLACK)
    success, captured_pieces = Rules.apply_move(board, move)
    
    print(f"\n黒が(7, 4)から(5, 4)へ移動して取る")
    print(f"成功: {success}")
    print(f"取った駒の数: {len(captured_pieces) if captured_pieces else 0}")
    if captured_pieces:
        print("取った駒:")
        for i, piece in enumerate(captured_pieces):
            print(f"  {i+1}. {piece}")
    
    print("\n移動後:")
    print(board)
    print(f"(5, 4)のスタック高さ: {board.get_stack_height((5, 4))}")
    
    assert success, "移動が失敗しました"
    assert captured_pieces is not None, "取った駒がNone"
    assert len(captured_pieces) == 3, f"取った駒の数が3ではなく{len(captured_pieces)}"
    assert board.get_stack_height((5, 4)) == 1, "移動後のスタック高さが1ではない"
    
    print("\n✅ テスト1合格: スタック全体が取れました")


def test_drop_piece_basic():
    """持ち駒を「新」で配置できることをテスト"""
    print("\n" + "=" * 60)
    print("テスト2: 持ち駒を「新」で配置")
    print("=" * 60)
    
    board = Board()
    
    # 黒の駒を配置（最前線）
    board.add_piece((7, 4), Piece(PieceType.DAI, Player.BLACK))
    board.add_piece((6, 3), Piece(PieceType.HYO, Player.BLACK))
    
    print("\n初期状態:")
    print(board)
    
    # 持ち駒
    hand_pieces = {
        PieceType.HYO: 2,
        PieceType.YARI: 1
    }
    
    print(f"\n黒の持ち駒: {hand_pieces}")
    
    # 合法手を取得
    legal_moves = Rules.get_legal_moves(board, Player.BLACK, hand_pieces)
    drop_moves = [m for m in legal_moves if m.move_type == MoveType.DROP]
    
    print(f"\n「新」で配置可能な手の数: {len(drop_moves)}")
    
    # 最前線より前には置けないことを確認
    for move in drop_moves:
        row, col = move.to_pos
        assert row >= 6, f"最前線(6)より前の行{row}に配置しようとしています"
    
    # 空マスに配置
    drop_move = Move.create_drop_move(to_pos=(7, 5), piece_type=PieceType.HYO, player=Player.BLACK)
    success, _ = Rules.apply_move(board, drop_move, hand_pieces)
    
    print(f"\n黒がHYOを(7, 5)に配置: {'成功' if success else '失敗'}")
    print(f"残りの持ち駒: {hand_pieces}")
    
    print("\n配置後:")
    print(board)
    
    assert success, "配置が失敗しました"
    assert hand_pieces[PieceType.HYO] == 1, "持ち駒が減っていません"
    
    print("\n✅ テスト2合格: 持ち駒を配置できました")


def test_drop_piece_restrictions():
    """「新」の配置制限をテスト"""
    print("\n" + "=" * 60)
    print("テスト3: 「新」の配置制限")
    print("=" * 60)
    
    board = Board()
    
    # 黒の駒を配置
    board.add_piece((7, 4), Piece(PieceType.SUI, Player.BLACK))  # 帥
    board.add_piece((7, 5), Piece(PieceType.HYO, Player.BLACK))  # 味方
    board.add_piece((6, 4), Piece(PieceType.YARI, Player.BLACK))  # 最前線
    
    # 白の駒を配置
    board.add_piece((5, 4), Piece(PieceType.DAI, Player.WHITE))  # 敵
    
    print("\n初期状態:")
    print(board)
    
    hand_pieces = {PieceType.HYO: 1}
    
    # テスト3-1: 帥の上には置けない
    print("\n--- テスト3-1: 帥の上には置けない ---")
    can_drop_on_sui = Rules._can_drop_piece_at(board, (7, 4), PieceType.HYO, Player.BLACK)
    print(f"帥の上に配置可能: {can_drop_on_sui}")
    assert not can_drop_on_sui, "帥の上に配置できてしまいます"
    print("✅ 帥の上には配置できません")
    
    # テスト3-2: 味方の駒の上には置ける
    print("\n--- テスト3-2: 味方の駒の上には置ける ---")
    can_drop_on_ally = Rules._can_drop_piece_at(board, (7, 5), PieceType.HYO, Player.BLACK)
    print(f"味方の駒の上に配置可能: {can_drop_on_ally}")
    assert can_drop_on_ally, "味方の駒の上に配置できません"
    print("✅ 味方の駒の上には配置できます")
    
    # テスト3-3: 敵の駒の上には置けない
    print("\n--- テスト3-3: 敵の駒の上には置けない ---")
    can_drop_on_enemy = Rules._can_drop_piece_at(board, (5, 4), PieceType.HYO, Player.BLACK)
    print(f"敵の駒の上に配置可能: {can_drop_on_enemy}")
    assert not can_drop_on_enemy, "敵の駒の上に配置できてしまいます"
    print("✅ 敵の駒の上には配置できません")
    
    # テスト3-4: 最前線より前には置けない
    print("\n--- テスト3-4: 最前線より前には置けない ---")
    frontline = Rules._get_frontline_row(board, Player.BLACK)
    print(f"黒の最前線: 行{frontline}")
    
    legal_moves = Rules.get_legal_moves(board, Player.BLACK, hand_pieces)
    drop_moves = [m for m in legal_moves if m.move_type == MoveType.DROP]
    
    for move in drop_moves:
        row, col = move.to_pos
        print(f"配置可能な位置: {move.to_pos}")
        assert row >= frontline, f"最前線({frontline})より前の行{row}に配置できてしまいます"
    
    print("✅ 最前線より前には配置できません")
    
    # テスト3-5: 砦は他の駒の上に置けない
    print("\n--- テスト3-5: 砦は他の駒の上に置けない ---")
    can_drop_toride = Rules._can_drop_piece_at(board, (7, 5), PieceType.TORIDE, Player.BLACK)
    print(f"砦を味方の駒の上に配置可能: {can_drop_toride}")
    assert not can_drop_toride, "砦を他の駒の上に配置できてしまいます"
    print("✅ 砦は他の駒の上には配置できません")
    
    print("\n✅ テスト3合格: すべての配置制限が正しく機能しています")


def test_drop_on_ally_creates_stack():
    """味方の駒の上に配置するとスタックが形成されることをテスト"""
    print("\n" + "=" * 60)
    print("テスト4: 味方の駒の上に配置してスタック形成")
    print("=" * 60)
    
    board = Board()
    
    # 黒の駒を配置
    board.add_piece((7, 4), Piece(PieceType.HYO, Player.BLACK))
    
    print("\n初期状態:")
    print(board)
    print(f"(7, 4)のスタック高さ: {board.get_stack_height((7, 4))}")
    
    # 持ち駒をその上に配置
    hand_pieces = {PieceType.YARI: 1}
    drop_move = Move.create_drop_move(to_pos=(7, 4), piece_type=PieceType.YARI, player=Player.BLACK)
    success, _ = Rules.apply_move(board, drop_move, hand_pieces)
    
    print(f"\nYARIを(7, 4)に配置（HYOの上）: {'成功' if success else '失敗'}")
    
    print("\n配置後:")
    print(board)
    print(f"(7, 4)のスタック高さ: {board.get_stack_height((7, 4))}")
    print(f"(7, 4)のスタック: {board.get_stack((7, 4))}")
    
    assert success, "配置が失敗しました"
    assert board.get_stack_height((7, 4)) == 2, "スタック高さが2ではない"
    
    top_piece = board.get_top_piece((7, 4))
    assert top_piece.piece_type == PieceType.YARI, "最上段がYARIではない"
    
    print("\n✅ テスト4合格: スタックが正しく形成されました")


if __name__ == "__main__":
    try:
        test_capture_entire_stack()
        test_drop_piece_basic()
        test_drop_piece_restrictions()
        test_drop_on_ally_creates_stack()
        
        print("\n" + "=" * 60)
        print("すべてのテストが合格しました！ ✅")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
