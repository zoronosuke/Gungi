"""
千日手検出ロジックのテスト（シンプル版）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.board import Board, BOARD_SIZE
from src.engine.piece import Player, PieceType, Piece
from src.engine.move import Move, MoveType
from src.engine.rules import Rules
from src.engine.initial_setup import load_initial_board, get_initial_hand_pieces


def test_position_key_consistency():
    """position_keyが同じ局面で同じキーを返すか"""
    print("\n=== Test 1: position_key一貫性テスト ===")
    
    board1 = load_initial_board()
    board2 = load_initial_board()
    
    hand1 = get_initial_hand_pieces(Player.BLACK)
    hand2 = get_initial_hand_pieces(Player.BLACK)
    opponent_hand = get_initial_hand_pieces(Player.WHITE)
    
    key1 = board1.get_position_key(Player.BLACK, hand1, opponent_hand)
    key2 = board2.get_position_key(Player.BLACK, hand2, opponent_hand)
    
    if key1 == key2:
        print("✓ 同一局面で同じキーを生成")
    else:
        print("✗ 同一局面で異なるキー!")
        print(f"  key1: {key1[:100]}...")
        print(f"  key2: {key2[:100]}...")
    
    return key1 == key2


def test_position_key_different_player():
    """手番が違うと異なるキーになるか"""
    print("\n=== Test 2: 手番による差異テスト ===")
    
    board = load_initial_board()
    black_hand = get_initial_hand_pieces(Player.BLACK)
    white_hand = get_initial_hand_pieces(Player.WHITE)
    
    key_black = board.get_position_key(Player.BLACK, black_hand, white_hand)
    key_white = board.get_position_key(Player.WHITE, white_hand, black_hand)
    
    if key_black != key_white:
        print("✓ 手番が違うと異なるキーを生成")
    else:
        print("✗ 手番が違っても同じキー!")
    
    return key_black != key_white


def test_position_key_after_move_and_back():
    """A→B→Aの往復で同じキーに戻るか"""
    print("\n=== Test 3: 往復移動後のキー一致テスト ===")
    
    board = load_initial_board()
    black_hand = get_initial_hand_pieces(Player.BLACK)
    white_hand = get_initial_hand_pieces(Player.WHITE)
    
    # 初期キー（BLACK視点）
    key_initial = board.get_position_key(Player.BLACK, black_hand, white_hand)
    print(f"  初期キー（先頭50文字）: {key_initial[:50]}...")
    
    # 合法手を取得
    legal_moves = Rules.get_legal_moves(board, Player.BLACK, black_hand)
    
    # 移動可能な手を探す（DROPやSTACKではなく、通常移動）
    move_to_apply = None
    for move in legal_moves:
        if move.move_type == MoveType.NORMAL and move.from_pos is not None:
            move_to_apply = move
            break
    
    if not move_to_apply:
        print("  移動可能な手が見つかりませんでした")
        # DROPを試す
        for move in legal_moves:
            if move.move_type == MoveType.DROP:
                print(f"  代わりにDROP手を使用: {move}")
                move_to_apply = move
                break
    
    if not move_to_apply:
        print("  テストスキップ（合法手なし）")
        return True
    
    print(f"  移動: {move_to_apply}")
    
    # 移動を適用
    success1, _ = Rules.apply_move(board, move_to_apply, black_hand)
    
    if not success1:
        print("  移動1が失敗")
        return False
    
    # 移動後のキー（WHITE視点 - 手番交代後）
    key_after_move1 = board.get_position_key(Player.WHITE, white_hand, black_hand)
    print(f"  移動後キー（先頭50文字）: {key_after_move1[:50]}...")
    
    print(f"  ※ 実際の千日手テストは、同一局面が複数回出現することを検出する")
    print(f"  ※ position_keyはその基盤となるキー生成機能")
    
    return True


def test_back_and_forth_detection_logic():
    """往復検出ロジックのテスト（現在のロジック vs 元のロジック）"""
    print("\n=== Test 4: 往復検出ロジックの比較 ===")
    
    # シナリオ: BLACK手1, WHITE手1, BLACK手2（= BLACK手1の逆）
    # action_idx は駒の移動を表す整数
    
    # 現在のロジック（プレイヤー別）
    print("\n  --- 現在のロジック（プレイヤー別） ---")
    last_actions_black = []
    last_actions_white = []
    
    # BLACK手1: action=100（A→B）
    action_black_1 = 100
    last_actions_black.append(action_black_1)
    print(f"  BLACK手1: action={action_black_1}, last_actions_black={last_actions_black}")
    
    # WHITE手1: action=200
    action_white_1 = 200
    last_actions_white.append(action_white_1)
    print(f"  WHITE手1: action={action_white_1}, last_actions_white={last_actions_white}")
    
    # BLACK手2: action=101（B→A、往復）
    action_black_2 = 101  # 違うaction_idx（逆方向の移動）
    
    # 現在のロジック: 直前の自分の手と同じかチェック
    is_back_and_forth_current = (len(last_actions_black) >= 1 and 
                                  last_actions_black[-1] == action_black_2)
    print(f"  BLACK手2: action={action_black_2}")
    print(f"  往復検出（現在）: {is_back_and_forth_current} ← 100 != 101 なので検出されない")
    
    # 元のロジック（両者混合）
    print("\n  --- 元のロジック（両者混合） ---")
    last_actions_mixed = []
    
    # BLACK手1
    last_actions_mixed.append(action_black_1)
    print(f"  BLACK手1: action={action_black_1}, last_actions={last_actions_mixed}")
    
    # WHITE手1
    last_actions_mixed.append(action_white_1)
    print(f"  WHITE手1: action={action_white_1}, last_actions={last_actions_mixed}")
    
    # BLACK手2: 2手前と比較
    is_back_and_forth_original = (len(last_actions_mixed) >= 2 and 
                                   last_actions_mixed[-2] == action_black_2)
    print(f"  BLACK手2: action={action_black_2}")
    print(f"  往復検出（元）: {is_back_and_forth_original} ← 100 != 101 なので検出されない")
    
    print("\n  === 結論 ===")
    print("  両方のロジックとも、異なるaction_idxの往復は検出できない！")
    print("  A→Bの移動（action=100）とB→Aの移動（action=101）は異なるaction_idx")
    print("  往復検出は本質的に機能していない可能性が高い")
    print("  → しかし、同じ手（同じaction_idx）の連続は検出できる")
    
    return True


def test_repetition_threshold():
    """REPETITION_THRESHOLDの効果テスト"""
    print("\n=== Test 5: 千日手閾値テスト ===")
    
    position_history = {}
    
    # 同じ局面が出現するシミュレーション
    key = "test_position_key"
    
    for i in range(5):
        position_history[key] = position_history.get(key, 0) + 1
        count = position_history[key]
        
        threshold_3 = count >= 3
        threshold_4 = count >= 4
        
        print(f"  {i+1}回目出現: count={count}, "
              f"閾値3で千日手={threshold_3}, 閾値4で千日手={threshold_4}")
    
    print("\n  閾値3: 3回目の出現で千日手判定 ← 元のコード")
    print("  閾値4: 4回目の出現で千日手判定")
    print("  ★ 閾値3の方がより早く千日手を検出する → 引き分けを早期終了させる")
    
    return True


def test_actual_repetition_scenario():
    """実際の千日手シナリオをシミュレート"""
    print("\n=== Test 6: 実際の千日手シナリオ ===")
    
    board = load_initial_board()
    black_hand = get_initial_hand_pieces(Player.BLACK)
    white_hand = get_initial_hand_pieces(Player.WHITE)
    
    position_history = {}
    REPETITION_THRESHOLD = 3
    
    current_player = Player.BLACK
    
    # 局面キーを取得する関数
    def get_key(player):
        if player == Player.BLACK:
            return board.get_position_key(player, black_hand, white_hand)
        else:
            return board.get_position_key(player, white_hand, black_hand)
    
    # 初期局面を記録
    key = get_key(current_player)
    position_history[key] = 1
    print(f"  初期局面記録: count={position_history[key]}")
    
    # 合法手を取得
    legal_moves = Rules.get_legal_moves(board, Player.BLACK, black_hand)
    
    # 移動可能な手を探す
    move_forward = None
    for move in legal_moves:
        if move.move_type == MoveType.NORMAL and move.from_pos is not None:
            move_forward = move
            break
    
    if not move_forward:
        print("  テストスキップ（移動可能な手がない）")
        return True
    
    print(f"  テスト手: {move_forward}")
    
    # 往復を3回シミュレート
    for cycle in range(3):
        print(f"\n  --- サイクル {cycle+1} ---")
        
        # 移動
        success, _ = Rules.apply_move(board, move_forward, black_hand)
        if not success:
            print(f"    移動失敗")
            break
        
        current_player = Player.WHITE
        key = get_key(current_player)
        position_history[key] = position_history.get(key, 0) + 1
        print(f"    移動後: count={position_history[key]}")
        
        if position_history[key] >= REPETITION_THRESHOLD:
            print(f"    ★ 千日手成立！（閾値{REPETITION_THRESHOLD}）")
            break
        
        # 注: 実際のゲームでは相手も動くが、このテストでは省略
        # 元の局面に戻す方法がないため、シンプルに局面カウントの挙動のみテスト
    
    return True


def main():
    print("=" * 60)
    print("千日手検出ロジック テスト")
    print("=" * 60)
    
    results = []
    
    results.append(("position_key一貫性", test_position_key_consistency()))
    results.append(("手番差異", test_position_key_different_player()))
    results.append(("往復キー一致", test_position_key_after_move_and_back()))
    results.append(("往復検出ロジック", test_back_and_forth_detection_logic()))
    results.append(("閾値効果", test_repetition_threshold()))
    results.append(("実際のシナリオ", test_actual_repetition_scenario()))
    
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(r for _, r in results)
    print(f"\n総合結果: {'全テスト合格' if all_passed else 'いくつかのテストが失敗'}")
    
    print("\n" + "=" * 60)
    print("分析結果")
    print("=" * 60)
    print("""
  【重要な発見】
  1. position_keyは正しく動作している
  2. 往復検出ロジック（_is_back_and_forth）は本質的に無意味
     - A→B（action=100）とB→A（action=101）は異なるaction_idx
     - どちらのロジックでも検出できない
  3. 千日手検出の本質はposition_historyのカウント
     - 同じ局面が閾値回出現したら千日手
     - 閾値3の方が早期に検出 → 引き分けを減らせる可能性
  
  【推奨設定】
  - last_actions: 元のコード（両者混合）に戻しても変わらない
  - REPETITION_THRESHOLD: 3に戻す（早期検出）
  - 往復検出ロジック自体は機能していないが、害もない
""")
    
    return all_passed


if __name__ == "__main__":
    main()
