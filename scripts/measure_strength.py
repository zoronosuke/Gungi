#!/usr/bin/env python
"""
モデルの強さを測定するスクリプト
対Random AI, 対Greedy AIでの勝率・引き分け率を測定
"""

import os
import sys
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.engine.board import Board
from src.engine.piece import Player, PieceType
from src.engine.rules import Rules
from src.engine.initial_setup import load_initial_board, get_initial_hand_pieces
from src.model.network import GungiNetwork
from src.model.mcts import MCTS


class RandomAI:
    """完全ランダムに手を選ぶAI"""
    
    def __init__(self):
        self.name = "Random"
    
    def get_move(self, board, player, hand, opponent_hand):
        legal_moves = Rules.get_legal_moves(board, player, hand)
        if not legal_moves:
            return None
        return np.random.choice(legal_moves)


class GreedyAI:
    """貪欲法AI - 駒を取れる手を優先"""
    
    def __init__(self):
        self.name = "Greedy"
        self.piece_values = {
            PieceType.SUI: 1000,
            PieceType.DAI: 50,
            PieceType.CHUU: 40,
            PieceType.SHO: 30,
            PieceType.SAMURAI: 25,
            PieceType.UMA: 20,
            PieceType.SHINOBI: 20,
            PieceType.YARI: 15,
            PieceType.YUMI: 15,
            PieceType.TORIDE: 10,
            PieceType.HYO: 5,
            PieceType.HOU: 15,
            PieceType.TSUTU: 15,
            PieceType.BOU: 20,
        }
    
    def get_move(self, board, player, hand, opponent_hand):
        legal_moves = Rules.get_legal_moves(board, player, hand)
        if not legal_moves:
            return None
        
        best_move = None
        best_score = -float('inf')
        
        for move in legal_moves:
            score = self._evaluate_move(board, player, move)
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move if best_move else np.random.choice(legal_moves)
    
    def _evaluate_move(self, board, player, move):
        score = 0.0
        if move.to_pos:
            target_stack = board.get_stack(move.to_pos)
            if target_stack and not target_stack.is_empty():
                top_piece = target_stack.get_top_piece()
                if top_piece and top_piece.owner != player:
                    score += self.piece_values.get(top_piece.piece_type, 10)
        score += np.random.uniform(0, 0.1)
        return score


class ModelAI:
    """学習済みモデルを使うAI"""
    
    def __init__(self, network, mcts_simulations=50, device='cuda'):
        self.name = f"Model-MCTS{mcts_simulations}"
        self.network = network
        self.device = device
        self.mcts = MCTS(
            network=network,
            num_simulations=mcts_simulations,
            device=device
        )
    
    def get_move(self, board, player, hand, opponent_hand):
        return self.mcts.get_best_move(board, player, hand, opponent_hand)


def play_game(ai1, ai2, max_moves=300):
    """
    2つのAI同士を対戦させる
    ai1 = BLACK, ai2 = WHITE
    
    Returns:
        (winner, num_moves, end_reason)
    """
    board = load_initial_board()
    hands = {
        Player.BLACK: get_initial_hand_pieces(Player.BLACK),
        Player.WHITE: get_initial_hand_pieces(Player.WHITE)
    }
    
    current_player = Player.BLACK
    move_count = 0
    position_history = {}
    
    while move_count < max_moves:
        is_over, winner = Rules.is_game_over(board)
        if is_over:
            return winner, move_count, "CHECKMATE"
        
        ai = ai1 if current_player == Player.BLACK else ai2
        my_hand = hands[current_player]
        opp_hand = hands[current_player.opponent]
        
        move = ai.get_move(board, current_player, my_hand, opp_hand)
        
        if move is None:
            return current_player.opponent, move_count, "NO_LEGAL_MOVES"
        
        success, _ = Rules.apply_move(board, move, my_hand)
        if not success:
            return current_player.opponent, move_count, "ILLEGAL_MOVE"
        
        position_key = board.get_position_key(current_player, my_hand, opp_hand)
        position_history[position_key] = position_history.get(position_key, 0) + 1
        if position_history[position_key] >= 4:
            return None, move_count, "REPETITION"
        
        current_player = current_player.opponent
        move_count += 1
    
    return None, max_moves, "MAX_MOVES"


def load_model_from_checkpoint(cp_path, device='cuda'):
    """チェックポイントからモデルをロード（構造を自動検出）"""
    checkpoint = torch.load(cp_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # 入力チャンネル数を推定（input_conv.weightから）
    first_conv_weight = state_dict['input_conv.weight']
    input_channels = first_conv_weight.shape[1]
    
    # フィルター数を推定
    num_filters = first_conv_weight.shape[0]
    
    # ResBlockの数を推定
    res_block_count = 0
    for key in state_dict.keys():
        if key.startswith('res_blocks.') and '.conv1.weight' in key:
            block_num = int(key.split('.')[1])
            res_block_count = max(res_block_count, block_num + 1)
    
    # アクション数を推定（policy_fc.weightから）
    policy_fc_weight = state_dict['policy_fc.weight']
    num_actions = policy_fc_weight.shape[0]
    
    print(f"  モデル構造: input_ch={input_channels}, filters={num_filters}, "
          f"res_blocks={res_block_count}, actions={num_actions}")
    
    # モデルを作成
    model = GungiNetwork(
        input_channels=input_channels,
        num_actions=num_actions,
        num_res_blocks=res_block_count,
        num_filters=num_filters
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


def run_evaluation(checkpoints, games_per_point=4, device='cuda'):
    """実際のモデルを使って評価を実行"""
    results = {
        'iterations': [],
        'vs_random_win': [],
        'vs_random_draw': [],
        'vs_greedy_win': [],
        'vs_greedy_draw': []
    }
    
    random_ai = RandomAI()
    greedy_ai = GreedyAI()
    
    print(f"\n{'='*60}")
    print(f"モデル評価を開始します（各チェックポイント {games_per_point * 2} 戦）")
    print(f"{'='*60}\n")
    
    for cp_path in checkpoints:
        # イテレーション数をファイル名から抽出
        try:
            iter_num = int(os.path.basename(cp_path).split('_')[-1].split('.')[0])
        except:
            continue
        
        print(f"\n--- Iteration {iter_num} ---")
        print(f"Loading: {os.path.basename(cp_path)}")
        
        try:
            model = load_model_from_checkpoint(cp_path, device)
        except Exception as e:
            print(f"  ロードエラー: {e}")
            continue
        
        model_ai = ModelAI(model, mcts_simulations=50, device=device)
        
        # vs Random
        print(f"  vs Random AI ({games_per_point * 2} games)...")
        wins, draws = 0, 0
        for i in tqdm(range(games_per_point), desc="    as Black"):
            winner, _, reason = play_game(model_ai, random_ai)
            if winner == Player.BLACK: wins += 1
            if winner is None: draws += 1
        
        for i in tqdm(range(games_per_point), desc="    as White"):
            winner, _, reason = play_game(random_ai, model_ai)
            if winner == Player.WHITE: wins += 1
            if winner is None: draws += 1
        
        total_games = games_per_point * 2
        win_rate = wins / total_games * 100
        draw_rate = draws / total_games * 100
        results['vs_random_win'].append(win_rate)
        results['vs_random_draw'].append(draw_rate)
        print(f"  → vs Random: Win={win_rate:.1f}%, Draw={draw_rate:.1f}%")
        
        # vs Greedy
        print(f"  vs Greedy AI ({games_per_point * 2} games)...")
        wins, draws = 0, 0
        for i in tqdm(range(games_per_point), desc="    as Black"):
            winner, _, reason = play_game(model_ai, greedy_ai)
            if winner == Player.BLACK: wins += 1
            if winner is None: draws += 1
        
        for i in tqdm(range(games_per_point), desc="    as White"):
            winner, _, reason = play_game(greedy_ai, model_ai)
            if winner == Player.WHITE: wins += 1
            if winner is None: draws += 1
        
        win_rate = wins / total_games * 100
        draw_rate = draws / total_games * 100
        results['vs_greedy_win'].append(win_rate)
        results['vs_greedy_draw'].append(draw_rate)
        print(f"  → vs Greedy: Win={win_rate:.1f}%, Draw={draw_rate:.1f}%")
        
        results['iterations'].append(iter_num)
    
    return results


def plot_strength(results, output_dir=None):
    """強さの推移をグラフ化"""
    if output_dir is None:
        output_dir = os.path.join(project_root, 'reports')
    os.makedirs(output_dir, exist_ok=True)
    
    iterations = results['iterations']
    
    plt.figure(figsize=(14, 6))
    
    # Win Rate
    plt.subplot(1, 2, 1)
    plt.plot(iterations, results['vs_random_win'], 'o-', label='vs Random AI', 
             color='blue', linewidth=2, markersize=8)
    plt.plot(iterations, results['vs_greedy_win'], 's-', label='vs Greedy AI', 
             color='green', linewidth=2, markersize=8)
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    plt.title('Win Rate vs Baseline AIs', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.ylim(-5, 105)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Draw Rate
    plt.subplot(1, 2, 2)
    plt.plot(iterations, results['vs_random_draw'], 'o--', label='Draw vs Random', 
             color='lightblue', linewidth=2, markersize=8)
    plt.plot(iterations, results['vs_greedy_draw'], 's--', label='Draw vs Greedy', 
             color='lightgreen', linewidth=2, markersize=8)
    plt.axhline(y=30, color='orange', linestyle='--', alpha=0.7, label='Warning (30%)')
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Critical (50%)')
    plt.title('Draw Rate (Repetition Behavior)', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Draw Rate (%)', fontsize=12)
    plt.ylim(-5, 105)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'report_strength_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nGraph Saved: {output_path}")
    
    # 結果をJSONで保存
    json_path = os.path.join(output_dir, 'strength_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results Saved: {json_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Measure model strength')
    parser.add_argument('--iterations', type=str, default='10,20,150',
                        help='Comma-separated iteration numbers to evaluate')
    parser.add_argument('--games', type=int, default=4,
                        help='Games per side per checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 指定されたイテレーションのチェックポイントを探す
    target_iters = [int(x.strip()) for x in args.iterations.split(',')]
    checkpoints = []
    for it in target_iters:
        cp_path = os.path.join(project_root, 'checkpoints', f'model_iter_{it:04d}.pt')
        if os.path.exists(cp_path):
            checkpoints.append(cp_path)
            print(f"Found: {cp_path}")
        else:
            print(f"Not found: {cp_path}")
    
    if not checkpoints:
        print("評価対象のチェックポイントが見つかりません。")
        sys.exit(1)
    
    results = run_evaluation(checkpoints, games_per_point=args.games, device=device)
    plot_strength(results)
    
    print("\n" + "="*60)
    print("評価完了！")
    print("="*60)
