"""
深層強化学習の重要コンポーネントのテスト
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.engine.board import Board, BOARD_SIZE
from src.engine.piece import Player, PieceType, Piece
from src.engine.move import Move, MoveType
from src.engine.rules import Rules
from src.engine.initial_setup import load_initial_board, get_initial_hand_pieces
from src.model.encoder import StateEncoder, ActionEncoder, PIECE_TYPES, NUM_PIECE_TYPES
from src.model.network import GungiNetwork


# =========================================
# Test 1: StateEncoder
# =========================================
def test_state_encoder_shape():
    """StateEncoderの出力形状テスト"""
    print("\n=== Test 1.1: StateEncoder 出力形状 ===")
    
    board = load_initial_board()
    player = Player.BLACK
    my_hand = get_initial_hand_pieces(player)
    opponent_hand = get_initial_hand_pieces(player.opponent)
    
    encoder = StateEncoder()
    state = encoder.encode(board, player, my_hand, opponent_hand)
    
    expected_shape = (91, 9, 9)
    if state.shape == expected_shape:
        print(f"✓ 形状正常: {state.shape}")
    else:
        print(f"✗ 形状異常: {state.shape} (期待: {expected_shape})")
        return False
    
    # データ型チェック
    if state.dtype == np.float32:
        print(f"✓ データ型正常: {state.dtype}")
    else:
        print(f"✗ データ型異常: {state.dtype} (期待: float32)")
        return False
    
    return True


def test_state_encoder_piece_representation():
    """駒の表現が正しいかテスト"""
    print("\n=== Test 1.2: StateEncoder 駒表現 ===")
    
    board = load_initial_board()
    player = Player.BLACK
    my_hand = get_initial_hand_pieces(player)
    opponent_hand = get_initial_hand_pieces(player.opponent)
    
    encoder = StateEncoder()
    state = encoder.encode(board, player, my_hand, opponent_hand)
    
    # 盤面上の駒が正しくエンコードされているか
    # チャンネル0-41: 自分の駒（14種類×3段）
    # チャンネル42-83: 相手の駒（14種類×3段）
    
    my_piece_channels = state[0:42]  # 自分の駒
    opp_piece_channels = state[42:84]  # 相手の駒
    
    # 何かしらの駒があるはず
    my_piece_sum = np.sum(my_piece_channels)
    opp_piece_sum = np.sum(opp_piece_channels)
    
    print(f"  自分の駒チャンネル合計: {my_piece_sum}")
    print(f"  相手の駒チャンネル合計: {opp_piece_sum}")
    
    if my_piece_sum > 0 and opp_piece_sum > 0:
        print("✓ 両者の駒が正しくエンコードされている")
    else:
        print("✗ 駒のエンコードに問題あり")
        return False
    
    # 手番チャンネル（ch 90）は全て1
    turn_channel = state[90]
    if np.all(turn_channel == 1.0):
        print("✓ 手番チャンネル正常（全て1）")
    else:
        print("✗ 手番チャンネル異常")
        return False
    
    return True


def test_state_encoder_symmetry():
    """BLACKとWHITEで対称的なエンコードになるか"""
    print("\n=== Test 1.3: StateEncoder 対称性 ===")
    
    board = load_initial_board()
    black_hand = get_initial_hand_pieces(Player.BLACK)
    white_hand = get_initial_hand_pieces(Player.WHITE)
    
    encoder = StateEncoder()
    
    # BLACKの視点
    state_black = encoder.encode(board, Player.BLACK, black_hand, white_hand)
    # WHITEの視点
    state_white = encoder.encode(board, Player.WHITE, white_hand, black_hand)
    
    # 自分の駒と相手の駒のチャンネルが入れ替わっているはず
    black_my_pieces = np.sum(state_black[0:42])
    black_opp_pieces = np.sum(state_black[42:84])
    white_my_pieces = np.sum(state_white[0:42])
    white_opp_pieces = np.sum(state_white[42:84])
    
    print(f"  BLACK視点: 自分={black_my_pieces}, 相手={black_opp_pieces}")
    print(f"  WHITE視点: 自分={white_my_pieces}, 相手={white_opp_pieces}")
    
    # BLACKの自分の駒 ≈ WHITEの相手の駒（配置は異なるが数は同じ）
    if abs(black_my_pieces - white_opp_pieces) < 0.001:
        print("✓ 視点の対称性が保たれている")
        return True
    else:
        print("✗ 視点の対称性に問題あり")
        return False


# =========================================
# Test 2: ActionEncoder
# =========================================
def test_action_encoder_move_roundtrip():
    """移動手のエンコード/デコードの往復テスト"""
    print("\n=== Test 2.1: ActionEncoder 移動手の往復 ===")
    
    encoder = ActionEncoder()
    
    # テスト用の移動手
    test_cases = [
        ((0, 0), (1, 0)),
        ((4, 4), (5, 5)),
        ((8, 8), (7, 7)),
        ((0, 8), (8, 0)),
    ]
    
    all_pass = True
    for from_pos, to_pos in test_cases:
        move = Move(MoveType.NORMAL, from_pos, to_pos, player=Player.BLACK)
        idx = encoder.encode_move(move)
        decoded = encoder.decode_action(idx, Player.BLACK)
        
        if decoded.from_pos == from_pos and decoded.to_pos == to_pos:
            print(f"  ✓ {from_pos}→{to_pos}: idx={idx}")
        else:
            print(f"  ✗ {from_pos}→{to_pos}: デコード失敗 ({decoded.from_pos}→{decoded.to_pos})")
            all_pass = False
    
    return all_pass


def test_action_encoder_drop_roundtrip():
    """DROP手のエンコード/デコードの往復テスト"""
    print("\n=== Test 2.2: ActionEncoder DROP手の往復 ===")
    
    encoder = ActionEncoder()
    
    # テスト用のDROP手
    test_cases = [
        (PieceType.SHO, (4, 4)),
        (PieceType.HYO, (0, 0)),
        (PieceType.DAI, (8, 8)),
        (PieceType.SAMURAI, (3, 5)),
    ]
    
    all_pass = True
    for piece_type, to_pos in test_cases:
        move = Move(MoveType.DROP, None, to_pos, piece_type, Player.BLACK)
        idx = encoder.encode_move(move)
        decoded = encoder.decode_action(idx, Player.BLACK)
        
        if decoded.to_pos == to_pos and decoded.piece_type == piece_type:
            print(f"  ✓ {piece_type.name}→{to_pos}: idx={idx}")
        else:
            print(f"  ✗ {piece_type.name}→{to_pos}: デコード失敗")
            all_pass = False
    
    return all_pass


def test_action_encoder_index_range():
    """アクションインデックスの範囲テスト"""
    print("\n=== Test 2.3: ActionEncoder インデックス範囲 ===")
    
    encoder = ActionEncoder()
    
    # 移動手の範囲: 0 <= idx < 6561
    move_min = Move(MoveType.NORMAL, (0, 0), (0, 0), player=Player.BLACK)
    move_max = Move(MoveType.NORMAL, (8, 8), (8, 8), player=Player.BLACK)
    
    idx_min = encoder.encode_move(move_min)
    idx_max = encoder.encode_move(move_max)
    
    print(f"  移動手インデックス範囲: {idx_min} ～ {idx_max}")
    print(f"  期待範囲: 0 ～ 6560")
    
    move_ok = (0 <= idx_min < 6561) and (0 <= idx_max < 6561)
    
    # DROP手の範囲: 6561 <= idx < 7695
    drop_first = Move(MoveType.DROP, None, (0, 0), PIECE_TYPES[0], Player.BLACK)
    drop_last = Move(MoveType.DROP, None, (8, 8), PIECE_TYPES[-1], Player.BLACK)
    
    idx_drop_first = encoder.encode_move(drop_first)
    idx_drop_last = encoder.encode_move(drop_last)
    
    print(f"  DROP手インデックス範囲: {idx_drop_first} ～ {idx_drop_last}")
    print(f"  期待範囲: 6561 ～ 7694")
    
    drop_ok = (6561 <= idx_drop_first < 7695) and (6561 <= idx_drop_last < 7695)
    
    if move_ok and drop_ok:
        print("✓ インデックス範囲正常")
        return True
    else:
        print("✗ インデックス範囲異常")
        return False


def test_action_encoder_legal_mask():
    """合法手マスクのテスト"""
    print("\n=== Test 2.4: ActionEncoder 合法手マスク ===")
    
    board = load_initial_board()
    player = Player.BLACK
    hand = get_initial_hand_pieces(player)
    
    encoder = ActionEncoder()
    
    # 合法手を取得
    legal_moves = Rules.get_legal_moves(board, player, hand)
    
    # マスクを生成
    mask = encoder.get_legal_mask(board, player, hand, legal_moves)
    
    print(f"  合法手数: {len(legal_moves)}")
    print(f"  マスクの1の数: {np.sum(mask)}")
    print(f"  マスク形状: {mask.shape}")
    
    if mask.shape == (7695,):
        print("✓ マスク形状正常")
    else:
        print("✗ マスク形状異常")
        return False
    
    if np.sum(mask) == len(legal_moves):
        print("✓ マスクの1の数と合法手数が一致")
    else:
        print(f"✗ 不一致: マスク={np.sum(mask)}, 合法手={len(legal_moves)}")
        return False
    
    # 各合法手が正しくマスクされているか
    all_masked = True
    for move in legal_moves:
        idx = encoder.encode_move(move)
        if mask[idx] != 1.0:
            print(f"  ✗ 合法手がマスクされていない: {move}")
            all_masked = False
            break
    
    if all_masked:
        print("✓ 全ての合法手が正しくマスクされている")
    
    return all_masked


# =========================================
# Test 3: Network (Forward Pass)
# =========================================
def test_network_forward_shape():
    """ネットワークの順伝播の形状テスト"""
    print("\n=== Test 3.1: Network 順伝播形状 ===")
    
    network = GungiNetwork(input_channels=91, num_res_blocks=2, num_filters=64)
    network.eval()
    
    # ダミー入力
    batch_size = 4
    dummy_input = torch.randn(batch_size, 91, 9, 9)
    
    with torch.no_grad():
        policy, value = network(dummy_input)
    
    print(f"  入力形状: {dummy_input.shape}")
    print(f"  Policy形状: {policy.shape}")
    print(f"  Value形状: {value.shape}")
    
    policy_ok = policy.shape == (batch_size, 7695)
    value_ok = value.shape == (batch_size, 1)
    
    if policy_ok:
        print("✓ Policy形状正常")
    else:
        print(f"✗ Policy形状異常 (期待: ({batch_size}, 7695))")
    
    if value_ok:
        print("✓ Value形状正常")
    else:
        print(f"✗ Value形状異常 (期待: ({batch_size}, 1))")
    
    return policy_ok and value_ok


def test_network_policy_distribution():
    """Policyが確率分布として正しいか"""
    print("\n=== Test 3.2: Network Policy分布 ===")
    
    network = GungiNetwork(input_channels=91, num_res_blocks=2, num_filters=64)
    network.eval()
    
    dummy_input = torch.randn(1, 91, 9, 9)
    
    with torch.no_grad():
        log_policy, _ = network(dummy_input)
        policy = torch.exp(log_policy)
    
    # 確率の合計が1になるか
    policy_sum = policy.sum().item()
    print(f"  Policy合計: {policy_sum}")
    
    if abs(policy_sum - 1.0) < 0.001:
        print("✓ Policy合計が1（確率分布として正常）")
    else:
        print("✗ Policy合計が1ではない")
        return False
    
    # 全て非負か
    if torch.all(policy >= 0):
        print("✓ 全て非負")
    else:
        print("✗ 負の確率が存在")
        return False
    
    return True


def test_network_value_range():
    """Valueが-1～1の範囲か"""
    print("\n=== Test 3.3: Network Value範囲 ===")
    
    network = GungiNetwork(input_channels=91, num_res_blocks=2, num_filters=64)
    network.eval()
    
    # 複数のランダム入力でテスト
    for i in range(10):
        dummy_input = torch.randn(1, 91, 9, 9) * 10  # 大きめの値
        
        with torch.no_grad():
            _, value = network(dummy_input)
        
        v = value.item()
        if not (-1.0 <= v <= 1.0):
            print(f"✗ Value範囲外: {v}")
            return False
    
    print("✓ 全てのValueが-1～1の範囲内")
    return True


# =========================================
# Test 4: Value Assignment (報酬割り当て)
# =========================================
def test_value_assignment():
    """勝敗による報酬割り当てのテスト"""
    print("\n=== Test 4.1: 報酬割り当て ===")
    
    # 期待される報酬
    # - 勝者: +1.0
    # - 敗者: -1.0
    # - 引き分け（千日手）: -0.9
    # - 引き分け（MAX_MOVES）: -0.2
    
    print("  期待される報酬:")
    print("    勝者: +1.0")
    print("    敗者: -1.0")
    print("    千日手: -0.9")
    print("    MAX_MOVES到達: -0.2")
    
    # コードから実際の値を確認
    from src.model.max_efficiency_selfplay import MaxEfficiencySelfPlay
    
    draw_rep = MaxEfficiencySelfPlay.DRAW_VALUE_REPETITION
    draw_max = MaxEfficiencySelfPlay.DRAW_VALUE_MAX_MOVES
    
    print(f"\n  実際の設定:")
    print(f"    DRAW_VALUE_REPETITION: {draw_rep}")
    print(f"    DRAW_VALUE_MAX_MOVES: {draw_max}")
    
    if draw_rep == -0.9 and draw_max == -0.2:
        print("✓ 報酬設定正常")
        return True
    else:
        print("✗ 報酬設定が期待と異なる")
        return False


# =========================================
# Test 5: Policy Normalization
# =========================================
def test_policy_normalization():
    """MCTS結果からの方策正規化テスト"""
    print("\n=== Test 5.1: Policy正規化 ===")
    
    encoder = ActionEncoder()
    
    # シミュレートされた訪問回数
    visit_counts = {
        100: 50,   # action 100: 50回訪問
        200: 30,   # action 200: 30回訪問
        300: 20,   # action 300: 20回訪問
    }
    
    policy = encoder.moves_to_policy([], visit_counts)
    
    # 正規化されているか
    policy_sum = np.sum(policy)
    print(f"  訪問回数合計: {sum(visit_counts.values())}")
    print(f"  Policy合計: {policy_sum}")
    
    if abs(policy_sum - 1.0) < 0.001:
        print("✓ Policy正規化正常（合計1）")
    else:
        print("✗ Policy正規化異常")
        return False
    
    # 個別の確率が正しいか
    expected_probs = {
        100: 0.5,
        200: 0.3,
        300: 0.2,
    }
    
    all_correct = True
    for idx, expected in expected_probs.items():
        actual = policy[idx]
        if abs(actual - expected) > 0.001:
            print(f"  ✗ action {idx}: 期待={expected}, 実際={actual}")
            all_correct = False
        else:
            print(f"  ✓ action {idx}: {actual}")
    
    return all_correct


# =========================================
# Test 6: End-to-End Integration
# =========================================
def test_end_to_end_inference():
    """エンドツーエンドの推論テスト"""
    print("\n=== Test 6.1: End-to-End推論 ===")
    
    # 1. 盤面を準備
    board = load_initial_board()
    player = Player.BLACK
    my_hand = get_initial_hand_pieces(player)
    opponent_hand = get_initial_hand_pieces(player.opponent)
    
    # 2. 状態をエンコード
    state_encoder = StateEncoder()
    state = state_encoder.encode(board, player, my_hand, opponent_hand)
    print(f"  状態エンコード: {state.shape}")
    
    # 3. ネットワークで推論
    network = GungiNetwork(input_channels=91, num_res_blocks=2, num_filters=64)
    network.eval()
    
    state_tensor = torch.from_numpy(state).unsqueeze(0).float()
    
    with torch.no_grad():
        log_policy, value = network(state_tensor)
        policy = torch.exp(log_policy).numpy()[0]
    
    print(f"  Policy形状: {policy.shape}")
    print(f"  Value: {value.item():.4f}")
    
    # 4. 合法手マスクを適用
    action_encoder = ActionEncoder()
    legal_moves = Rules.get_legal_moves(board, player, my_hand)
    mask = action_encoder.get_legal_mask(board, player, my_hand, legal_moves)
    
    masked_policy = policy * mask
    
    # 正規化
    policy_sum = np.sum(masked_policy)
    if policy_sum > 0:
        masked_policy = masked_policy / policy_sum
    
    print(f"  合法手数: {len(legal_moves)}")
    print(f"  マスク後Policy合計: {np.sum(masked_policy):.4f}")
    
    # 5. 最善手を選択
    best_action = np.argmax(masked_policy)
    best_move = action_encoder.decode_action(best_action, player, board)
    
    print(f"  最善手: {best_move}")
    
    # 6. 選択した手が合法か確認
    is_legal = any(
        action_encoder.encode_move(m) == best_action
        for m in legal_moves
    )
    
    if is_legal:
        print("✓ 選択した手は合法")
        return True
    else:
        print("✗ 選択した手が非合法！")
        return False


# =========================================
# Main
# =========================================
def main():
    print("=" * 60)
    print("深層強化学習 コンポーネントテスト")
    print("=" * 60)
    
    results = []
    
    # Test 1: StateEncoder
    print("\n" + "=" * 40)
    print("【StateEncoder テスト】")
    print("=" * 40)
    results.append(("StateEncoder形状", test_state_encoder_shape()))
    results.append(("StateEncoder駒表現", test_state_encoder_piece_representation()))
    results.append(("StateEncoder対称性", test_state_encoder_symmetry()))
    
    # Test 2: ActionEncoder
    print("\n" + "=" * 40)
    print("【ActionEncoder テスト】")
    print("=" * 40)
    results.append(("ActionEncoder移動往復", test_action_encoder_move_roundtrip()))
    results.append(("ActionEncoder DROP往復", test_action_encoder_drop_roundtrip()))
    results.append(("ActionEncoderインデックス範囲", test_action_encoder_index_range()))
    results.append(("ActionEncoder合法手マスク", test_action_encoder_legal_mask()))
    
    # Test 3: Network
    print("\n" + "=" * 40)
    print("【Network テスト】")
    print("=" * 40)
    results.append(("Network順伝播形状", test_network_forward_shape()))
    results.append(("NetworkPolicy分布", test_network_policy_distribution()))
    results.append(("NetworkValue範囲", test_network_value_range()))
    
    # Test 4: Value Assignment
    print("\n" + "=" * 40)
    print("【報酬割り当て テスト】")
    print("=" * 40)
    results.append(("報酬割り当て", test_value_assignment()))
    
    # Test 5: Policy Normalization
    print("\n" + "=" * 40)
    print("【Policy正規化 テスト】")
    print("=" * 40)
    results.append(("Policy正規化", test_policy_normalization()))
    
    # Test 6: End-to-End
    print("\n" + "=" * 40)
    print("【End-to-End テスト】")
    print("=" * 40)
    results.append(("End-to-End推論", test_end_to_end_inference()))
    
    # Summary
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    
    passed = 0
    failed = 0
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n合計: {passed}/{len(results)} テスト合格")
    
    if failed == 0:
        print("\n★ 全テスト合格！エンコーダーとネットワークは正常に動作しています。")
    else:
        print(f"\n⚠ {failed}件のテストが失敗しました。確認が必要です。")
    
    return failed == 0


if __name__ == "__main__":
    main()
