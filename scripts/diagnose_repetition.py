#!/usr/bin/env python
"""
千日手診断・可視化スクリプト
学習中の千日手発生パターンを分析して可視化

使用例:
    python scripts/diagnose_repetition.py
    python scripts/diagnose_repetition.py --games 10 --verbose
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import copy

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.engine.board import Board
from src.engine.piece import Player, PieceType
from src.engine.move import Move, MoveType
from src.engine.rules import Rules
from src.engine.initial_setup import load_initial_board, get_initial_hand_pieces
from src.model.network import GungiNetwork, create_model
from src.model.encoder import StateEncoder, ActionEncoder


def analyze_game_for_repetition(
    model: GungiNetwork,
    state_encoder: StateEncoder,
    action_encoder: ActionEncoder,
    device: str,
    mcts_sims: int = 50,
    max_moves: int = 300,
    verbose: bool = False
) -> Dict:
    """1ゲームを分析して千日手パターンを検出"""
    
    board = load_initial_board()
    hands = {
        Player.BLACK: get_initial_hand_pieces(Player.BLACK),
        Player.WHITE: get_initial_hand_pieces(Player.WHITE)
    }
    
    current_player = Player.BLACK
    move_count = 0
    position_history = {}
    action_history = []
    
    # 診断データ
    diagnostics = {
        'total_moves': 0,
        'repetition_occurred': False,
        'repetition_move': None,
        'position_frequencies': [],  # 各局面の出現回数
        'action_patterns': [],  # 連続したアクションパターン
        'back_and_forth_count': 0,  # 往復パターンの回数
        'policy_entropy': [],  # ポリシーエントロピー（多様性指標）
        'top_action_prob': [],  # 最大確率アクションの確率
        'move_details': []  # 各手の詳細
    }
    
    # 初期局面を記録
    initial_key = board.get_position_key(Player.BLACK, hands[Player.BLACK], hands[Player.WHITE])
    position_history[initial_key] = 1
    
    model.eval()
    
    while move_count < max_moves:
        # ゲーム終了チェック
        is_over, winner = Rules.is_game_over(board)
        if is_over:
            diagnostics['game_result'] = winner.name if winner else 'DRAW'
            break
        
        # 合法手を取得
        my_hand = hands[current_player]
        opponent_hand = hands[current_player.opponent]
        legal_moves = Rules.get_legal_moves(board, current_player, my_hand)
        
        if not legal_moves:
            diagnostics['game_result'] = current_player.opponent.name
            break
        
        # 状態をエンコード
        state = state_encoder.encode(board, current_player, my_hand, opponent_hand)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # NN推論
        with torch.no_grad():
            log_policy, value = model(state_tensor)
            policy = torch.exp(log_policy).cpu().numpy().flatten()
        
        # 合法手マスクを適用
        legal_action_indices = []
        for move in legal_moves:
            action_idx = action_encoder.encode_move(move)
            if action_idx is not None:
                legal_action_indices.append(action_idx)
        
        legal_mask = np.zeros(7695)
        for idx in legal_action_indices:
            legal_mask[idx] = 1.0
        
        masked_policy = policy * legal_mask
        total = masked_policy.sum()
        if total > 0:
            masked_policy = masked_policy / total
        else:
            masked_policy = legal_mask / legal_mask.sum()
        
        # エントロピー計算（多様性指標）
        non_zero_probs = masked_policy[masked_policy > 0]
        entropy = -np.sum(non_zero_probs * np.log(non_zero_probs + 1e-10))
        diagnostics['policy_entropy'].append(entropy)
        
        # 最大確率
        top_prob = masked_policy.max()
        diagnostics['top_action_prob'].append(top_prob)
        
        # アクション選択（温度なし、最大確率）
        best_action = np.argmax(masked_policy)
        
        # アクション履歴に追加
        action_history.append(best_action)
        
        # 往復パターン検出
        if len(action_history) >= 4:
            if action_history[-1] == action_history[-3] and action_history[-2] == action_history[-4]:
                diagnostics['back_and_forth_count'] += 1
        
        # 手を適用
        best_move = action_encoder.decode_action(best_action, current_player, board)
        if best_move is None:
            # デコード失敗、ランダムに選択
            best_move = legal_moves[0]
        
        move_detail = {
            'move_num': move_count,
            'player': current_player.name,
            'action_idx': int(best_action),
            'entropy': float(entropy),
            'top_prob': float(top_prob),
            'value': float(value.cpu().numpy().flatten()[0])
        }
        
        success, _ = Rules.apply_move(board, best_move, my_hand)
        if not success:
            break
        
        # 局面記録
        position_key = board.get_position_key(current_player.opponent, opponent_hand, my_hand)
        position_history[position_key] = position_history.get(position_key, 0) + 1
        
        freq = position_history[position_key]
        diagnostics['position_frequencies'].append(freq)
        move_detail['position_frequency'] = freq
        
        if verbose:
            diagnostics['move_details'].append(move_detail)
        
        # 千日手チェック
        if freq >= 3:
            diagnostics['repetition_occurred'] = True
            diagnostics['repetition_move'] = move_count
            diagnostics['game_result'] = 'REPETITION'
            break
        
        current_player = current_player.opponent
        move_count += 1
    
    diagnostics['total_moves'] = move_count
    
    if 'game_result' not in diagnostics:
        diagnostics['game_result'] = 'MAX_MOVES'
    
    return diagnostics


def run_diagnosis(model_path: str, num_games: int = 20, verbose: bool = False):
    """診断を実行"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print("Gungi AI - Repetition Diagnosis")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Games to analyze: {num_games}")
    
    # モデルをロード
    model = create_model(device, test_mode=True)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model not found, using random weights")
    
    state_encoder = StateEncoder()
    action_encoder = ActionEncoder()
    
    # 全体統計
    results = {
        'BLACK': 0,
        'WHITE': 0,
        'REPETITION': 0,
        'MAX_MOVES': 0,
        'OTHER': 0
    }
    
    all_diagnostics = []
    total_back_and_forth = 0
    avg_entropy = []
    avg_top_prob = []
    repetition_moves = []
    
    print(f"\nRunning {num_games} diagnostic games...")
    for i in range(num_games):
        diag = analyze_game_for_repetition(
            model, state_encoder, action_encoder, device,
            verbose=verbose
        )
        all_diagnostics.append(diag)
        
        result = diag['game_result']
        if result in results:
            results[result] += 1
        else:
            results['OTHER'] += 1
        
        total_back_and_forth += diag['back_and_forth_count']
        
        if diag['policy_entropy']:
            avg_entropy.extend(diag['policy_entropy'])
        if diag['top_action_prob']:
            avg_top_prob.extend(diag['top_action_prob'])
        
        if diag['repetition_occurred']:
            repetition_moves.append(diag['repetition_move'])
        
        print(f"  Game {i+1}/{num_games}: {result} "
              f"(moves={diag['total_moves']}, back&forth={diag['back_and_forth_count']})")
    
    # サマリー
    print(f"\n{'='*60}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n【結果分布】")
    print(f"  BLACK wins:  {results['BLACK']:>3} ({100*results['BLACK']/num_games:.1f}%)")
    print(f"  WHITE wins:  {results['WHITE']:>3} ({100*results['WHITE']/num_games:.1f}%)")
    print(f"  REPETITION:  {results['REPETITION']:>3} ({100*results['REPETITION']/num_games:.1f}%) ⚠️ 問題")
    print(f"  MAX_MOVES:   {results['MAX_MOVES']:>3} ({100*results['MAX_MOVES']/num_games:.1f}%)")
    
    print(f"\n【千日手指標】")
    print(f"  千日手率:     {100*results['REPETITION']/num_games:.1f}%")
    print(f"  往復回数合計: {total_back_and_forth}")
    if repetition_moves:
        print(f"  千日手発生手数: 平均 {np.mean(repetition_moves):.1f}手")
    
    print(f"\n【ポリシー多様性】")
    if avg_entropy:
        print(f"  エントロピー:   平均 {np.mean(avg_entropy):.3f} (高いほど多様)")
        print(f"                 min={np.min(avg_entropy):.3f}, max={np.max(avg_entropy):.3f}")
    if avg_top_prob:
        print(f"  最大確率:       平均 {np.mean(avg_top_prob):.3f} (低いほど多様)")
        print(f"                 min={np.min(avg_top_prob):.3f}, max={np.max(avg_top_prob):.3f}")
    
    # 問題判定
    print(f"\n【診断結果】")
    rep_rate = results['REPETITION'] / num_games
    if rep_rate > 0.5:
        print("  🔴 重大: 千日手率が50%を超えています。学習が正常に機能していません。")
    elif rep_rate > 0.2:
        print("  🟡 警告: 千日手率が20%を超えています。調整が必要です。")
    elif rep_rate > 0.05:
        print("  🟢 許容: 千日手率は5-20%です。正常な範囲です。")
    else:
        print("  ✅ 良好: 千日手率は5%未満です。学習は正常です。")
    
    # 結果を保存
    output = {
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'num_games': num_games,
        'results': results,
        'total_back_and_forth': total_back_and_forth,
        'avg_entropy': float(np.mean(avg_entropy)) if avg_entropy else 0,
        'avg_top_prob': float(np.mean(avg_top_prob)) if avg_top_prob else 0,
        'repetition_rate': rep_rate,
        'diagnostics': all_diagnostics if verbose else []
    }
    
    output_path = './repetition_diagnosis.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
    
    return output


def visualize_diagnosis(diagnosis_path: str = './repetition_diagnosis.json'):
    """診断結果を可視化"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("matplotlib is required for visualization")
        return
    
    with open(diagnosis_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 結果分布（パイチャート）
    ax1 = axes[0, 0]
    results = data['results']
    labels = list(results.keys())
    sizes = list(results.values())
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#95a5a6']
    explode = [0.05 if k == 'REPETITION' else 0 for k in labels]
    ax1.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%')
    ax1.set_title('Game Results Distribution')
    
    # 2. 千日手発生統計
    ax2 = axes[0, 1]
    metrics = ['Repetition\nRate', 'Back&Forth\n(avg/game)', 'Top Prob\n(avg)']
    values = [
        data['repetition_rate'] * 100,
        data['total_back_and_forth'] / data['num_games'],
        data['avg_top_prob'] * 100
    ]
    colors2 = ['#e74c3c' if data['repetition_rate'] > 0.2 else '#2ecc71',
               '#f39c12', '#3498db']
    bars = ax2.bar(metrics, values, color=colors2)
    ax2.set_ylabel('Value')
    ax2.set_title('Repetition Metrics')
    
    # バーに値を表示
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 3. ポリシーエントロピーの分布
    ax3 = axes[1, 0]
    if data.get('diagnostics'):
        all_entropy = []
        for d in data['diagnostics']:
            all_entropy.extend(d.get('policy_entropy', []))
        if all_entropy:
            ax3.hist(all_entropy, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
            ax3.axvline(np.mean(all_entropy), color='red', linestyle='--', label=f'Mean: {np.mean(all_entropy):.2f}')
            ax3.legend()
    ax3.set_xlabel('Policy Entropy')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Policy Entropy Distribution (Higher = More Diverse)')
    
    # 4. 最大確率の分布
    ax4 = axes[1, 1]
    if data.get('diagnostics'):
        all_top_prob = []
        for d in data['diagnostics']:
            all_top_prob.extend(d.get('top_action_prob', []))
        if all_top_prob:
            ax4.hist(all_top_prob, bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
            ax4.axvline(np.mean(all_top_prob), color='blue', linestyle='--', label=f'Mean: {np.mean(all_top_prob):.2f}')
            ax4.legend()
    ax4.set_xlabel('Top Action Probability')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Top Action Probability Distribution (Lower = More Diverse)')
    
    plt.tight_layout()
    
    output_path = './repetition_diagnosis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Diagnose repetition issues in training')
    parser.add_argument('--model', type=str, default='./checkpoints/latest.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--games', type=int, default=20,
                        help='Number of games to analyze')
    parser.add_argument('--verbose', action='store_true',
                        help='Save detailed move information')
    parser.add_argument('--visualize-only', action='store_true',
                        help='Only visualize existing results')
    
    args = parser.parse_args()
    
    if args.visualize_only:
        visualize_diagnosis()
    else:
        run_diagnosis(args.model, args.games, args.verbose)
        visualize_diagnosis()


if __name__ == "__main__":
    main()
