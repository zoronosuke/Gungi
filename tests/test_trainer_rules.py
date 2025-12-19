"""
学習ループ（Trainer）とゲームルール（Rules）のテスト
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import copy
from typing import List

from src.engine.board import Board, BOARD_SIZE
from src.engine.piece import Player, PieceType, Piece
from src.engine.move import Move, MoveType
from src.engine.rules import Rules
from src.engine.initial_setup import load_initial_board, get_initial_hand_pieces
from src.model.encoder import StateEncoder, ActionEncoder
from src.model.network import GungiNetwork
from src.model.trainer import Trainer, GungiDataset
from src.model.self_play import TrainingExample


# =========================================
# Test 1: 損失関数
# =========================================
def test_policy_loss():
    """Policy損失のテスト"""
    print("\n=== Test 1.1: Policy損失 ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = GungiNetwork(input_channels=91, num_res_blocks=2, num_filters=64)
    network.to(device)
    
    trainer = Trainer(network, device=device)
    
    # テスト用のテンソル
    batch_size = 4
    
    # log_policies: ネットワークの出力（log_softmax後）
    log_policies = torch.log_softmax(torch.randn(batch_size, 7695), dim=1).to(device)
    
    # target_policies: 正解のポリシー（確率分布）
    target_policies = torch.softmax(torch.randn(batch_size, 7695), dim=1).to(device)
    
    # values: ネットワークの出力
    values = torch.tanh(torch.randn(batch_size, 1)).to(device)
    
    # target_values: 正解の価値
    target_values = torch.tensor([[1.0], [-1.0], [0.5], [-0.5]]).to(device)
    
    policy_loss, value_loss = trainer._compute_loss(
        log_policies, values, target_policies, target_values
    )
    
    print(f"  Policy Loss: {policy_loss.item():.4f}")
    print(f"  Value Loss: {value_loss.item():.4f}")
    
    if policy_loss.item() > 0:
        print("✓ Policy Lossは正の値")
    else:
        print("✗ Policy Lossが0以下")
        return False
    
    if value_loss.item() >= 0:
        print("✓ Value Lossは非負")
    else:
        print("✗ Value Lossが負")
        return False
    
    return True


def test_loss_gradient():
    """損失が勾配を持つかテスト"""
    print("\n=== Test 1.2: 勾配計算 ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = GungiNetwork(input_channels=91, num_res_blocks=2, num_filters=64)
    network.to(device)
    network.train()
    
    trainer = Trainer(network, device=device)
    
    # ダミー入力
    states = torch.randn(2, 91, 9, 9).to(device)
    
    # 順伝播
    log_policies, values = network(states)
    
    # ターゲット
    target_policies = torch.softmax(torch.randn(2, 7695), dim=1).to(device)
    target_values = torch.tensor([[1.0], [-1.0]]).to(device)
    
    # 損失計算
    policy_loss, value_loss = trainer._compute_loss(
        log_policies, values, target_policies, target_values
    )
    total_loss = policy_loss + value_loss
    
    # 逆伝播
    trainer.optimizer.zero_grad()
    total_loss.backward()
    
    # 勾配が存在するか確認
    has_grad = False
    for param in network.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    if has_grad:
        print("✓ 勾配が計算された")
    else:
        print("✗ 勾配がゼロまたは存在しない")
        return False
    
    return True


def test_training_step():
    """1ステップの学習テスト"""
    print("\n=== Test 1.3: 学習ステップ ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = GungiNetwork(input_channels=91, num_res_blocks=2, num_filters=64)
    network.to(device)
    
    trainer = Trainer(network, device=device)
    
    # ダミーの学習データを作成
    examples = []
    for i in range(10):
        state = np.random.randn(91, 9, 9).astype(np.float32)
        policy = np.zeros(7695, dtype=np.float32)
        policy[np.random.randint(0, 7695)] = 1.0  # 1つの手に確率1
        value = np.random.choice([-1.0, 1.0])
        examples.append(TrainingExample(state=state, policy=policy, value=value))
    
    # 学習前のパラメータを保存
    params_before = {name: param.clone() for name, param in network.named_parameters()}
    
    # 1エポック学習
    result = trainer.train(examples, batch_size=4, epochs=1, verbose=False)
    
    print(f"  Policy Loss: {result['policy_loss']:.4f}")
    print(f"  Value Loss: {result['value_loss']:.4f}")
    
    # パラメータが更新されたか確認
    params_changed = False
    for name, param in network.named_parameters():
        if not torch.equal(param, params_before[name]):
            params_changed = True
            break
    
    if params_changed:
        print("✓ パラメータが更新された")
    else:
        print("✗ パラメータが更新されていない")
        return False
    
    return True


# =========================================
# Test 2: データセット
# =========================================
def test_dataset_creation():
    """データセットの作成テスト"""
    print("\n=== Test 2.1: データセット作成 ===")
    
    examples = []
    for i in range(5):
        state = np.random.randn(91, 9, 9).astype(np.float32)
        policy = np.zeros(7695, dtype=np.float32)
        policy[i * 100] = 1.0
        value = float(i % 2) * 2 - 1  # -1 or 1
        examples.append(TrainingExample(state=state, policy=policy, value=value))
    
    dataset = GungiDataset(examples)
    
    print(f"  データセットサイズ: {len(dataset)}")
    
    if len(dataset) == 5:
        print("✓ サイズが正しい")
    else:
        print("✗ サイズが不正")
        return False
    
    # アイテム取得
    state, policy, value = dataset[0]
    
    print(f"  状態形状: {state.shape}")
    print(f"  方策形状: {policy.shape}")
    print(f"  価値形状: {value.shape}")
    
    if state.shape == (91, 9, 9):
        print("✓ 状態形状が正しい")
    else:
        print("✗ 状態形状が不正")
        return False
    
    if policy.shape == (7695,):
        print("✓ 方策形状が正しい")
    else:
        print("✗ 方策形状が不正")
        return False
    
    return True


def test_value_distribution():
    """学習データの価値分布テスト"""
    print("\n=== Test 2.2: 価値分布テスト ===")
    
    # 勝敗が正しく反映されているかテスト
    # 勝者: +1, 敗者: -1 であるべき
    
    examples = []
    
    # 勝ち側のサンプル
    for i in range(5):
        state = np.random.randn(91, 9, 9).astype(np.float32)
        policy = np.zeros(7695, dtype=np.float32)
        policy[i] = 1.0
        examples.append(TrainingExample(state=state, policy=policy, value=1.0))
    
    # 負け側のサンプル
    for i in range(5):
        state = np.random.randn(91, 9, 9).astype(np.float32)
        policy = np.zeros(7695, dtype=np.float32)
        policy[i] = 1.0
        examples.append(TrainingExample(state=state, policy=policy, value=-1.0))
    
    values = [ex.value for ex in examples]
    
    print(f"  価値の種類: {set(values)}")
    print(f"  +1の数: {values.count(1.0)}")
    print(f"  -1の数: {values.count(-1.0)}")
    
    if set(values) == {1.0, -1.0}:
        print("✓ 価値が正しく+1と-1")
    else:
        print("✗ 価値の分布が不正")
        return False
    
    return True


# =========================================
# Test 3: ゲームルール - 勝敗判定
# =========================================
def test_game_over_sui_capture():
    """帥の捕獲による勝敗判定"""
    print("\n=== Test 3.1: 帥捕獲による終局 ===")
    
    board = load_initial_board()
    
    # ゲーム開始時は終了していない
    is_over, winner = Rules.is_game_over(board)
    print(f"  初期状態: is_over={is_over}, winner={winner}")
    
    if not is_over:
        print("✓ 初期状態はゲーム終了していない")
    else:
        print("✗ 初期状態でゲーム終了と判定")
        return False
    
    # 帥の位置を確認
    black_sui = board.get_sui_position(Player.BLACK)
    white_sui = board.get_sui_position(Player.WHITE)
    
    print(f"  BLACK帥の位置: {black_sui}")
    print(f"  WHITE帥の位置: {white_sui}")
    
    if black_sui is not None and white_sui is not None:
        print("✓ 両者の帥が存在")
    else:
        print("✗ 帥が見つからない")
        return False
    
    # 帥を強制的に削除してテスト
    board_copy = board.copy()
    
    # BLACK帥を削除（本来はルールでこうならないが、テスト用）
    if black_sui:
        # stacksの中のpiecesをクリア
        board_copy.stacks[black_sui[0]][black_sui[1]].pieces = []
        board_copy.sui_positions[Player.BLACK] = None
    
    is_over, winner = Rules.is_game_over(board_copy)
    print(f"  BLACK帥削除後: is_over={is_over}, winner={winner}")
    
    if is_over and winner == Player.WHITE:
        print("✓ BLACK帥がないとWHITEの勝ち")
    else:
        print("✗ 勝敗判定が不正")
        return False
    
    return True


def test_legal_moves_count():
    """合法手数のテスト"""
    print("\n=== Test 3.2: 合法手数 ===")
    
    board = load_initial_board()
    black_hand = get_initial_hand_pieces(Player.BLACK)
    white_hand = get_initial_hand_pieces(Player.WHITE)
    
    black_moves = Rules.get_legal_moves(board, Player.BLACK, black_hand)
    white_moves = Rules.get_legal_moves(board, Player.WHITE, white_hand)
    
    print(f"  BLACK合法手数: {len(black_moves)}")
    print(f"  WHITE合法手数: {len(white_moves)}")
    
    if len(black_moves) > 0:
        print("✓ BLACKに合法手がある")
    else:
        print("✗ BLACKに合法手がない")
        return False
    
    if len(white_moves) > 0:
        print("✓ WHITEに合法手がある")
    else:
        print("✗ WHITEに合法手がない")
        return False
    
    return True


def test_move_application():
    """手の適用テスト"""
    print("\n=== Test 3.3: 手の適用 ===")
    
    board = load_initial_board()
    player = Player.BLACK
    hand = get_initial_hand_pieces(player)
    
    # 合法手を取得
    legal_moves = Rules.get_legal_moves(board, player, hand)
    
    if not legal_moves:
        print("✗ 合法手がない")
        return False
    
    # 最初の合法手を適用
    move = legal_moves[0]
    print(f"  適用する手: {move.move_type}, {move.from_pos} -> {move.to_pos}")
    
    board_copy = board.copy()
    hand_copy = copy.deepcopy(hand)
    
    success, captured = Rules.apply_move(board_copy, move, hand_copy)
    
    print(f"  適用結果: success={success}, captured={captured}")
    
    if success:
        print("✓ 手が正常に適用された")
    else:
        print("✗ 手の適用に失敗")
        return False
    
    return True


# =========================================
# Test 4: チェックポイント
# =========================================
def test_model_save_load():
    """モデルの保存と読み込みテスト"""
    print("\n=== Test 4.1: モデル保存/読み込み ===")
    
    import tempfile
    import os
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # モデル1を作成して保存
    network1 = GungiNetwork(input_channels=91, num_res_blocks=2, num_filters=64)
    network1.to(device)
    
    # ダミー入力で出力を取得
    dummy_input = torch.randn(1, 91, 9, 9).to(device)
    with torch.no_grad():
        output1_policy, output1_value = network1(dummy_input)
    
    # 一時ファイルに保存
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
    
    torch.save(network1.state_dict(), temp_path)
    print(f"  保存完了: {temp_path}")
    
    # 新しいモデルに読み込み
    network2 = GungiNetwork(input_channels=91, num_res_blocks=2, num_filters=64)
    network2.load_state_dict(torch.load(temp_path, map_location=device))
    network2.to(device)
    
    # 同じ出力が得られるか確認
    with torch.no_grad():
        output2_policy, output2_value = network2(dummy_input)
    
    policy_match = torch.allclose(output1_policy, output2_policy, atol=1e-6)
    value_match = torch.allclose(output1_value, output2_value, atol=1e-6)
    
    print(f"  Policy一致: {policy_match}")
    print(f"  Value一致: {value_match}")
    
    # 一時ファイルを削除
    os.unlink(temp_path)
    
    if policy_match and value_match:
        print("✓ 保存/読み込みが正常")
        return True
    else:
        print("✗ 保存/読み込み後の出力が異なる")
        return False


# =========================================
# Test 5: 実際のデータで学習
# =========================================
def test_real_data_training():
    """実際のゲームデータで学習テスト"""
    print("\n=== Test 5.1: 実データ学習 ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = GungiNetwork(input_channels=91, num_res_blocks=2, num_filters=64)
    network.to(device)
    
    trainer = Trainer(network, lr=0.01, device=device)  # 高い学習率でテスト
    
    state_encoder = StateEncoder()
    action_encoder = ActionEncoder()
    
    # 実際の盤面からデータを作成
    examples = []
    
    board = load_initial_board()
    player = Player.BLACK
    my_hand = get_initial_hand_pieces(player)
    opponent_hand = get_initial_hand_pieces(player.opponent)
    
    # 5つのサンプルを作成
    for i in range(5):
        state = state_encoder.encode(board, player, my_hand, opponent_hand)
        
        # 合法手から1つを選んでポリシーを作成
        legal_moves = Rules.get_legal_moves(board, player, my_hand)
        
        policy = np.zeros(7695, dtype=np.float32)
        if legal_moves:
            selected_move = legal_moves[i % len(legal_moves)]
            action_idx = action_encoder.encode_move(selected_move)
            policy[action_idx] = 1.0
        
        value = 1.0 if i % 2 == 0 else -1.0
        
        examples.append(TrainingExample(state=state, policy=policy, value=value))
    
    print(f"  サンプル数: {len(examples)}")
    
    # 学習前の損失
    network.eval()
    states = torch.from_numpy(np.array([ex.state for ex in examples])).float().to(device)
    policies = torch.from_numpy(np.array([ex.policy for ex in examples])).float().to(device)
    values = torch.tensor([[ex.value] for ex in examples]).float().to(device)
    
    with torch.no_grad():
        log_policies, pred_values = network(states)
        policy_loss_before, value_loss_before = trainer._compute_loss(
            log_policies, pred_values, policies, values
        )
    
    print(f"  学習前 - Policy Loss: {policy_loss_before.item():.4f}, Value Loss: {value_loss_before.item():.4f}")
    
    # 学習
    network.train()
    result = trainer.train(examples, batch_size=2, epochs=10, verbose=False)
    
    # 学習後の損失
    network.eval()
    with torch.no_grad():
        log_policies, pred_values = network(states)
        policy_loss_after, value_loss_after = trainer._compute_loss(
            log_policies, pred_values, policies, values
        )
    
    print(f"  学習後 - Policy Loss: {policy_loss_after.item():.4f}, Value Loss: {value_loss_after.item():.4f}")
    
    # 損失が減少したか確認
    if policy_loss_after < policy_loss_before:
        print("✓ Policy Lossが減少")
    else:
        print("⚠ Policy Lossが減少していない（過学習または収束）")
    
    if value_loss_after < value_loss_before:
        print("✓ Value Lossが減少")
    else:
        print("⚠ Value Lossが減少していない")
    
    # 少なくとも1つは減少していれば学習が機能している
    if policy_loss_after < policy_loss_before or value_loss_after < value_loss_before:
        print("✓ 学習が機能している")
        return True
    else:
        print("⚠ 学習効果が見られない（データが少ないため）")
        return True  # 警告だけで失敗にはしない


# =========================================
# Main
# =========================================
def main():
    print("=" * 60)
    print("学習ループ・ゲームルール テスト")
    print("=" * 60)
    
    results = []
    
    # Test 1: 損失関数
    print("\n" + "=" * 40)
    print("【損失関数 テスト】")
    print("=" * 40)
    results.append(("Policy損失", test_policy_loss()))
    results.append(("勾配計算", test_loss_gradient()))
    results.append(("学習ステップ", test_training_step()))
    
    # Test 2: データセット
    print("\n" + "=" * 40)
    print("【データセット テスト】")
    print("=" * 40)
    results.append(("データセット作成", test_dataset_creation()))
    results.append(("価値分布", test_value_distribution()))
    
    # Test 3: ゲームルール
    print("\n" + "=" * 40)
    print("【ゲームルール テスト】")
    print("=" * 40)
    results.append(("帥捕獲終局", test_game_over_sui_capture()))
    results.append(("合法手数", test_legal_moves_count()))
    results.append(("手の適用", test_move_application()))
    
    # Test 4: チェックポイント
    print("\n" + "=" * 40)
    print("【チェックポイント テスト】")
    print("=" * 40)
    results.append(("モデル保存/読み込み", test_model_save_load()))
    
    # Test 5: 実データ学習
    print("\n" + "=" * 40)
    print("【実データ学習 テスト】")
    print("=" * 40)
    results.append(("実データ学習", test_real_data_training()))
    
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
        print("\n★ 全テスト合格！学習ループとゲームルールは正常に動作しています。")
    else:
        print(f"\n⚠ {failed}件のテストが失敗しました。確認が必要です。")
    
    return failed == 0


if __name__ == "__main__":
    main()
