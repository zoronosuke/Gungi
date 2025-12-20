#!/usr/bin/env python
"""
Elo Rating計算スクリプト
異なるイテレーションのモデル同士を対戦させてEloを算出

使用例:
    # 全イテレーションを評価
    python scripts/calculate_elo.py
    
    # 特定のイテレーションのみ
    python scripts/calculate_elo.py --iterations 1 5 10
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import glob

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.engine.board import Board
from src.engine.piece import Player, PieceType
from src.engine.rules import Rules
from src.engine.initial_setup import load_initial_board, get_initial_hand_pieces
from src.model.network import GungiNetwork, create_model
from src.model.mcts import MCTS


class EloRating:
    """Elo Rating システム"""
    
    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1500.0):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: Dict[str, float] = {}
    
    def get_rating(self, player_id: str) -> float:
        if player_id not in self.ratings:
            self.ratings[player_id] = self.initial_rating
        return self.ratings[player_id]
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """プレイヤーAの期待スコアを計算"""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    
    def update_ratings(self, player_a: str, player_b: str, 
                      score_a: float, score_b: float):
        """対戦結果でレーティングを更新
        
        score_a, score_b: 1.0=勝ち, 0.5=引き分け, 0.0=負け
        """
        rating_a = self.get_rating(player_a)
        rating_b = self.get_rating(player_b)
        
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = self.expected_score(rating_b, rating_a)
        
        self.ratings[player_a] = rating_a + self.k_factor * (score_a - expected_a)
        self.ratings[player_b] = rating_b + self.k_factor * (score_b - expected_b)


def load_model_for_iteration(iteration: int, device: str) -> Optional[GungiNetwork]:
    """特定イテレーションのモデルをロード"""
    checkpoint_path = f'./checkpoints/model_iter_{iteration:04d}.pt'
    
    if not os.path.exists(checkpoint_path):
        return None
    
    model = create_model(device, test_mode=True)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def play_game_between_models(model1: GungiNetwork, model2: GungiNetwork, 
                             mcts_sims: int, device: str) -> Tuple[Optional[Player], int]:
    """2つのモデル間で対戦"""
    mcts1 = MCTS(network=model1, num_simulations=mcts_sims, device=device)
    mcts2 = MCTS(network=model2, num_simulations=mcts_sims, device=device)
    
    board = load_initial_board()
    hands = {
        Player.BLACK: get_initial_hand_pieces(Player.BLACK),
        Player.WHITE: get_initial_hand_pieces(Player.WHITE)
    }
    
    current_player = Player.BLACK
    move_count = 0
    max_moves = 200
    position_history = {}
    
    while move_count < max_moves:
        is_over, winner = Rules.is_game_over(board)
        if is_over:
            return winner, move_count
        
        mcts = mcts1 if current_player == Player.BLACK else mcts2
        my_hand = hands[current_player]
        opp_hand = hands[current_player.opponent]
        
        move = mcts.get_best_move(board, current_player, my_hand, opp_hand)
        if move is None:
            return current_player.opponent, move_count
        
        success, _ = Rules.apply_move(board, move, my_hand)
        if not success:
            return current_player.opponent, move_count
        
        # 千日手チェック
        position_key = board.get_position_key(current_player, my_hand, opp_hand)
        position_history[position_key] = position_history.get(position_key, 0) + 1
        if position_history[position_key] >= 3:
            return None, move_count
        
        current_player = current_player.opponent
        move_count += 1
    
    return None, max_moves


def calculate_elo_ratings(iterations: List[int], games_per_match: int = 10,
                          mcts_sims: int = 50, device: str = 'cuda') -> Dict:
    """全イテレーション間のEloを計算"""
    
    elo = EloRating(k_factor=32.0, initial_rating=1500.0)
    results = {
        'iterations': iterations,
        'ratings': {},
        'match_results': []
    }
    
    # モデルをロード
    models = {}
    print("Loading models...")
    for it in tqdm(iterations):
        model = load_model_for_iteration(it, device)
        if model is not None:
            models[it] = model
        else:
            print(f"  Warning: Model for iteration {it} not found")
    
    available_iterations = list(models.keys())
    print(f"Loaded {len(available_iterations)} models")
    
    if len(available_iterations) < 2:
        print("Need at least 2 models for Elo calculation")
        return results
    
    # 隣接イテレーション間で対戦
    print("\nPlaying matches...")
    for i in range(len(available_iterations) - 1):
        it1 = available_iterations[i]
        it2 = available_iterations[i + 1]
        
        model1 = models[it1]
        model2 = models[it2]
        
        player1_id = f"iter_{it1:04d}"
        player2_id = f"iter_{it2:04d}"
        
        wins1, wins2, draws = 0, 0, 0
        
        for game in range(games_per_match):
            # 先手後手を交互に
            if game % 2 == 0:
                winner, _ = play_game_between_models(model1, model2, mcts_sims, device)
                if winner == Player.BLACK:
                    wins1 += 1
                elif winner == Player.WHITE:
                    wins2 += 1
                else:
                    draws += 1
            else:
                winner, _ = play_game_between_models(model2, model1, mcts_sims, device)
                if winner == Player.BLACK:
                    wins2 += 1
                elif winner == Player.WHITE:
                    wins1 += 1
                else:
                    draws += 1
        
        # Elo更新
        total = games_per_match
        score1 = (wins1 + 0.5 * draws) / total
        score2 = (wins2 + 0.5 * draws) / total
        
        for _ in range(games_per_match):
            elo.update_ratings(player1_id, player2_id, score1, score2)
        
        results['match_results'].append({
            'iter1': it1,
            'iter2': it2,
            'wins1': wins1,
            'wins2': wins2,
            'draws': draws
        })
        
        print(f"  Iter {it1} vs {it2}: {wins1}-{wins2}-{draws}")
    
    # 最終レーティング
    for it in available_iterations:
        player_id = f"iter_{it:04d}"
        results['ratings'][it] = elo.get_rating(player_id)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Calculate Elo ratings for model iterations')
    parser.add_argument('--iterations', type=int, nargs='+', default=None,
                        help='Specific iterations to evaluate')
    parser.add_argument('--games', type=int, default=6,
                        help='Games per match')
    parser.add_argument('--mcts-sims', type=int, default=50,
                        help='MCTS simulations')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--output', type=str, default='./elo_results.json',
                        help='Output file')
    
    args = parser.parse_args()
    
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print("Gungi AI Elo Rating Calculation")
    print(f"{'='*60}")
    print(f"Device: {args.device}")
    print(f"Games per match: {args.games}")
    print(f"MCTS simulations: {args.mcts_sims}")
    
    # イテレーションを自動検出または指定
    if args.iterations is None:
        checkpoints = glob.glob('./checkpoints/model_iter_*.pt')
        args.iterations = sorted([
            int(os.path.basename(c).replace('model_iter_', '').replace('.pt', ''))
            for c in checkpoints
        ])
    
    print(f"Iterations to evaluate: {args.iterations}")
    
    # Elo計算
    results = calculate_elo_ratings(
        args.iterations,
        games_per_match=args.games,
        mcts_sims=args.mcts_sims,
        device=args.device
    )
    
    results['timestamp'] = datetime.now().isoformat()
    results['config'] = {
        'games_per_match': args.games,
        'mcts_simulations': args.mcts_sims
    }
    
    # 結果保存
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 結果表示
    print(f"\n{'='*60}")
    print("Elo Ratings")
    print(f"{'='*60}")
    
    if results['ratings']:
        for it in sorted(results['ratings'].keys()):
            rating = results['ratings'][it]
            print(f"  Iteration {it:>4}: {rating:>7.1f}")
        
        # 改善を表示
        iterations = sorted(results['ratings'].keys())
        if len(iterations) >= 2:
            first = results['ratings'][iterations[0]]
            last = results['ratings'][iterations[-1]]
            improvement = last - first
            print(f"\nElo improvement: {improvement:+.1f} points")
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
