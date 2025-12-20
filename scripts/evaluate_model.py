#!/usr/bin/env python
"""
学習済みモデルの評価スクリプト

使用例:
    # 最新モデルを評価
    python scripts/evaluate_model.py
    
    # 特定のイテレーションを評価
    python scripts/evaluate_model.py --iteration 10
    
    # 対戦数を指定
    python scripts/evaluate_model.py --games 50
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
import copy

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.engine.board import Board
from src.engine.piece import Player, PieceType
from src.engine.move import Move
from src.engine.rules import Rules
from src.engine.initial_setup import load_initial_board, get_initial_hand_pieces
from src.model.network import GungiNetwork, create_model
from src.model.encoder import StateEncoder, ActionEncoder
from src.model.mcts import MCTS


class RandomAI:
    """完全ランダムに手を選ぶAI"""
    
    def __init__(self):
        self.name = "Random"
    
    def get_move(self, board: Board, player: Player, hand: Dict[PieceType, int], 
                 opponent_hand: Dict[PieceType, int]) -> Optional[Move]:
        legal_moves = Rules.get_legal_moves(board, player, hand)
        if not legal_moves:
            return None
        return np.random.choice(legal_moves)


class GreedyAI:
    """貪欲法AI - 駒を取れる手を優先"""
    
    def __init__(self):
        self.name = "Greedy"
        self.piece_values = {
            PieceType.SUI: 1000,   # 帥（王）
            PieceType.DAI: 50,     # 大
            PieceType.CHUU: 40,    # 中
            PieceType.SHO: 30,     # 小
            PieceType.SAMURAI: 25, # 侍
            PieceType.UMA: 20,     # 馬
            PieceType.SHINOBI: 20, # 忍
            PieceType.YARI: 15,    # 槍
            PieceType.YUMI: 15,    # 弓
            PieceType.TORIDE: 10,  # 砦
            PieceType.HYO: 5,      # 兵
            PieceType.HOU: 15,     # 砲
            PieceType.TSUTU: 15,   # 筒
            PieceType.BOU: 20,     # 謀
        }
    
    def get_move(self, board: Board, player: Player, hand: Dict[PieceType, int],
                 opponent_hand: Dict[PieceType, int]) -> Optional[Move]:
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
    
    def _evaluate_move(self, board: Board, player: Player, move: Move) -> float:
        """手を評価"""
        score = 0.0
        
        # 移動先に相手の駒があれば、その価値を加算
        if move.to_pos:
            target_stack = board.get_stack(move.to_pos)
            if target_stack:
                top_piece = target_stack[-1]
                if top_piece.player != player:
                    score += self.piece_values.get(top_piece.piece_type, 10)
        
        # ランダム要素を追加（同点時の多様性）
        score += np.random.uniform(0, 0.1)
        
        return score


class SimpleMCTSAI:
    """軽量MCTS AI（ニューラルネットなし、ランダムプレイアウト）"""
    
    def __init__(self, simulations: int = 50):
        self.name = f"SimpleMCTS-{simulations}"
        self.simulations = simulations
    
    def get_move(self, board: Board, player: Player, hand: Dict[PieceType, int],
                 opponent_hand: Dict[PieceType, int]) -> Optional[Move]:
        legal_moves = Rules.get_legal_moves(board, player, hand)
        if not legal_moves:
            return None
        
        # 手が多すぎる場合はサンプリング
        moves_to_evaluate = legal_moves[:min(50, len(legal_moves))]
        
        move_scores = {}
        for move in moves_to_evaluate:
            score = self._evaluate_move(board, player, hand, opponent_hand, move)
            move_scores[id(move)] = (move, score)
        
        # 最もスコアが高い手を選択
        best_move = max(move_scores.values(), key=lambda x: x[1])[0]
        return best_move
    
    def _evaluate_move(self, board: Board, player: Player, hand: Dict[PieceType, int],
                       opponent_hand: Dict[PieceType, int], move: Move) -> float:
        """手を評価（ランダムプレイアウト）"""
        sim_board = board.copy()
        sim_hand = copy.deepcopy(hand)
        sim_opp_hand = copy.deepcopy(opponent_hand)
        
        success, _ = Rules.apply_move(sim_board, move, sim_hand)
        if not success:
            return -100.0
        
        # 即座に勝ちなら高評価
        is_over, winner = Rules.is_game_over(sim_board)
        if is_over and winner == player:
            return 100.0
        if is_over and winner == player.opponent:
            return -100.0
        
        # ランダムプレイアウト
        wins = 0
        sims = min(10, self.simulations)
        for _ in range(sims):
            result = self._random_playout(
                sim_board.copy(), player.opponent, 
                copy.deepcopy(sim_opp_hand), copy.deepcopy(sim_hand)
            )
            if result == player:
                wins += 1
            elif result is None:
                wins += 0.5  # 引き分け
        
        return wins / sims
    
    def _random_playout(self, board: Board, current_player: Player, 
                       hand1: Dict[PieceType, int], hand2: Dict[PieceType, int],
                       max_moves: int = 30) -> Optional[Player]:
        """ランダムにゲームを進めて勝者を返す"""
        hands = {current_player: hand1, current_player.opponent: hand2}
        
        for _ in range(max_moves):
            is_over, winner = Rules.is_game_over(board)
            if is_over:
                return winner
            
            legal_moves = Rules.get_legal_moves(board, current_player, hands[current_player])
            if not legal_moves:
                return current_player.opponent
            
            move = np.random.choice(legal_moves)
            Rules.apply_move(board, move, hands[current_player])
            current_player = current_player.opponent
        
        return None  # 引き分け


class ModelAI:
    """学習済みモデルを使うAI"""
    
    def __init__(self, network: GungiNetwork, mcts_simulations: int = 200, device: str = 'cuda'):
        self.name = f"LearnedModel-MCTS{mcts_simulations}"
        self.network = network
        self.device = device
        self.mcts = MCTS(
            network=network,
            num_simulations=mcts_simulations,
            device=device
        )
    
    def get_move(self, board: Board, player: Player, hand: Dict[PieceType, int],
                 opponent_hand: Dict[PieceType, int]) -> Optional[Move]:
        return self.mcts.get_best_move(board, player, hand, opponent_hand)


def play_game(ai1, ai2, verbose: bool = False) -> Tuple[Optional[Player], int, str]:
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
    max_moves = 300
    position_history = {}
    
    while move_count < max_moves:
        # ゲーム終了チェック
        is_over, winner = Rules.is_game_over(board)
        if is_over:
            return winner, move_count, "CHECKMATE"
        
        # AIに手を選ばせる
        ai = ai1 if current_player == Player.BLACK else ai2
        my_hand = hands[current_player]
        opp_hand = hands[current_player.opponent]
        
        move = ai.get_move(board, current_player, my_hand, opp_hand)
        
        if move is None:
            # 合法手なし
            return current_player.opponent, move_count, "NO_LEGAL_MOVES"
        
        # 手を適用
        success, _ = Rules.apply_move(board, move, my_hand)
        if not success:
            return current_player.opponent, move_count, "ILLEGAL_MOVE"
        
        # 千日手チェック
        position_key = board.get_position_key(current_player, my_hand, opp_hand)
        position_history[position_key] = position_history.get(position_key, 0) + 1
        if position_history[position_key] >= 3:
            return None, move_count, "REPETITION"
        
        if verbose and move_count % 20 == 0:
            print(f"  Move {move_count}: {current_player.name}")
        
        current_player = current_player.opponent
        move_count += 1
    
    return None, max_moves, "MAX_MOVES"


def evaluate_vs_opponent(model_ai, opponent_ai, num_games: int, verbose: bool = True) -> Dict:
    """モデルを特定の相手と対戦させて評価"""
    results = {
        'model_wins': 0,
        'opponent_wins': 0,
        'draws': 0,
        'total_moves': [],
        'end_reasons': {'CHECKMATE': 0, 'REPETITION': 0, 'MAX_MOVES': 0, 'NO_LEGAL_MOVES': 0, 'ILLEGAL_MOVE': 0}
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_ai.name} vs {opponent_ai.name}")
        print(f"Games: {num_games}")
        print(f"{'='*60}")
    
    iterator = tqdm(range(num_games), desc=f"vs {opponent_ai.name}") if verbose else range(num_games)
    
    for i in iterator:
        # 先手後手を交互に
        if i % 2 == 0:
            ai_black = model_ai
            ai_white = opponent_ai
            model_is_black = True
        else:
            ai_black = opponent_ai
            ai_white = model_ai
            model_is_black = False
        
        winner, moves, reason = play_game(ai_black, ai_white, verbose=False)
        
        results['total_moves'].append(moves)
        if reason in results['end_reasons']:
            results['end_reasons'][reason] += 1
        
        if winner is None:
            results['draws'] += 1
        elif (winner == Player.BLACK and model_is_black) or (winner == Player.WHITE and not model_is_black):
            results['model_wins'] += 1
        else:
            results['opponent_wins'] += 1
    
    # 統計計算
    results['win_rate'] = results['model_wins'] / num_games if num_games > 0 else 0
    results['draw_rate'] = results['draws'] / num_games if num_games > 0 else 0
    results['avg_moves'] = float(np.mean(results['total_moves'])) if results['total_moves'] else 0
    results['std_moves'] = float(np.std(results['total_moves'])) if results['total_moves'] else 0
    
    if verbose:
        print(f"\nResults:")
        print(f"  Model wins: {results['model_wins']}/{num_games} ({results['win_rate']*100:.1f}%)")
        print(f"  Opponent wins: {results['opponent_wins']}/{num_games}")
        print(f"  Draws: {results['draws']}/{num_games} ({results['draw_rate']*100:.1f}%)")
        print(f"  Avg moves: {results['avg_moves']:.1f} ± {results['std_moves']:.1f}")
        print(f"  End reasons: {results['end_reasons']}")
    
    return results


def load_model(checkpoint_path: str, device: str = 'cuda') -> GungiNetwork:
    """チェックポイントからモデルをロード"""
    model = create_model(device, test_mode=True)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
        if 'iteration' in checkpoint:
            print(f"  Iteration: {checkpoint['iteration']}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using randomly initialized model")
    
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained Gungi AI model')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/latest.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--iteration', type=int, default=None,
                        help='Specific iteration to evaluate')
    parser.add_argument('--games', type=int, default=20,
                        help='Number of games to play against each opponent')
    parser.add_argument('--mcts-sims', type=int, default=100,
                        help='MCTS simulations for the model')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--output', type=str, default='./evaluation_results.json',
                        help='Output file for results')
    parser.add_argument('--quick', action='store_true',
                        help='Quick evaluation (fewer games, lower MCTS)')
    
    args = parser.parse_args()
    
    if args.quick:
        args.games = 10
        args.mcts_sims = 50
    
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print(f"Gungi AI Model Evaluation")
    print(f"{'='*60}")
    print(f"Device: {args.device}")
    print(f"MCTS simulations: {args.mcts_sims}")
    print(f"Games per opponent: {args.games}")
    
    # チェックポイントパス決定
    if args.iteration is not None:
        checkpoint_path = f'./checkpoints/model_iter_{args.iteration:04d}.pt'
    else:
        checkpoint_path = args.checkpoint
    
    # モデルロード
    print(f"\nLoading model from: {checkpoint_path}")
    model = load_model(checkpoint_path, args.device)
    
    # AI作成
    model_ai = ModelAI(model, mcts_simulations=args.mcts_sims, device=args.device)
    random_ai = RandomAI()
    greedy_ai = GreedyAI()
    simple_mcts_ai = SimpleMCTSAI(simulations=50)
    
    # 評価実行
    evaluation_results = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': checkpoint_path,
        'mcts_simulations': args.mcts_sims,
        'games_per_opponent': args.games,
        'opponents': {}
    }
    
    # vs Random
    print("\n" + "="*60)
    results_random = evaluate_vs_opponent(model_ai, random_ai, args.games)
    evaluation_results['opponents']['Random'] = {
        'win_rate': results_random['win_rate'],
        'draw_rate': results_random['draw_rate'],
        'avg_moves': results_random['avg_moves'],
        'model_wins': results_random['model_wins'],
        'opponent_wins': results_random['opponent_wins'],
        'draws': results_random['draws'],
    }
    
    # vs Greedy
    print("\n" + "="*60)
    results_greedy = evaluate_vs_opponent(model_ai, greedy_ai, args.games)
    evaluation_results['opponents']['Greedy'] = {
        'win_rate': results_greedy['win_rate'],
        'draw_rate': results_greedy['draw_rate'],
        'avg_moves': results_greedy['avg_moves'],
        'model_wins': results_greedy['model_wins'],
        'opponent_wins': results_greedy['opponent_wins'],
        'draws': results_greedy['draws'],
    }
    
    # vs SimpleMCTS
    print("\n" + "="*60)
    results_mcts = evaluate_vs_opponent(model_ai, simple_mcts_ai, args.games)
    evaluation_results['opponents']['SimpleMCTS-50'] = {
        'win_rate': results_mcts['win_rate'],
        'draw_rate': results_mcts['draw_rate'],
        'avg_moves': results_mcts['avg_moves'],
        'model_wins': results_mcts['model_wins'],
        'opponent_wins': results_mcts['opponent_wins'],
        'draws': results_mcts['draws'],
    }
    
    # 結果保存
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    # サマリー表示
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Opponent':<20} {'Win Rate':>10} {'Draws':>10} {'Avg Moves':>12}")
    print("-" * 55)
    print(f"{'Random':<20} {results_random['win_rate']*100:>9.1f}% {results_random['draw_rate']*100:>9.1f}% {results_random['avg_moves']:>11.1f}")
    print(f"{'Greedy':<20} {results_greedy['win_rate']*100:>9.1f}% {results_greedy['draw_rate']*100:>9.1f}% {results_greedy['avg_moves']:>11.1f}")
    print(f"{'SimpleMCTS-50':<20} {results_mcts['win_rate']*100:>9.1f}% {results_mcts['draw_rate']*100:>9.1f}% {results_mcts['avg_moves']:>11.1f}")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
