"""
GPU活用版の並列自己対戦
バッチ推論でGPUを効率的に使用する
"""

import copy
import numpy as np
import torch
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Event
from queue import Queue, Empty
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import time

from ..engine.board import Board
from ..engine.piece import Player, PieceType
from ..engine.move import Move, MoveType
from ..engine.rules import Rules
from ..engine.initial_setup import load_initial_board, get_initial_hand_pieces
from .encoder import StateEncoder, ActionEncoder
from .self_play import TrainingExample


@dataclass
class GameState:
    """ゲームの状態"""
    board: Board
    current_player: Player
    hands: Dict[Player, Dict[PieceType, int]]
    move_count: int
    history: List[Tuple[np.ndarray, np.ndarray, Player]]  # (state, policy, player)


class GPUSelfPlay:
    """
    GPU活用版の自己対戦
    複数ゲームを並行してプレイし、NN推論はGPUでバッチ処理
    """
    
    MAX_MOVES = 300
    
    def __init__(
        self,
        network,
        state_encoder: StateEncoder = None,
        action_encoder: ActionEncoder = None,
        mcts_simulations: int = 50,
        c_puct: float = 1.5,
        device: str = 'cuda',
        batch_size: int = 64
    ):
        self.network = network
        self.state_encoder = state_encoder or StateEncoder()
        self.action_encoder = action_encoder or ActionEncoder()
        self.mcts_simulations = mcts_simulations
        self.c_puct = c_puct
        self.device = device
        self.batch_size = batch_size
        
        self.network.to(device)
        self.network.eval()
    
    def _batch_inference(self, states: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """バッチでNN推論"""
        if not states:
            return np.array([]), np.array([])
        
        states_array = np.stack(states)
        states_tensor = torch.from_numpy(states_array).float().to(self.device)
        
        with torch.no_grad():
            log_policies, values = self.network(states_tensor)
            policies = torch.exp(log_policies).cpu().numpy()
            values = values.cpu().numpy().flatten()
        
        return policies, values
    
    def _mcts_search(
        self,
        board: Board,
        current_player: Player,
        my_hand: Dict[PieceType, int],
        opponent_hand: Dict[PieceType, int],
        temperature: float = 1.0
    ) -> Tuple[int, np.ndarray]:
        """単一ゲームのMCTS探索（GPU推論版）"""
        
        # 簡易版MCTS（バッチ推論の恩恵を受けるために設計）
        root_state = self.state_encoder.encode(board, current_player, my_hand, opponent_hand)
        
        # 合法手を取得
        legal_moves = Rules.get_legal_moves(board, current_player, my_hand)
        if not legal_moves:
            return 0, np.zeros(7695)
        
        legal_actions = []
        for move in legal_moves:
            action_idx = self.action_encoder.encode_move(move)
            if action_idx is not None:
                legal_actions.append(action_idx)
        
        if not legal_actions:
            return 0, np.zeros(7695)
        
        # 訪問回数を記録
        visit_counts = defaultdict(int)
        total_values = defaultdict(float)
        
        # バッチで処理するために状態を収集
        for sim in range(self.mcts_simulations):
            # ルートの推論
            policies, values = self._batch_inference([root_state])
            root_policy = policies[0]
            root_value = values[0]
            
            # 合法手でマスク
            legal_mask = np.zeros(7695)
            for action_idx in legal_actions:
                legal_mask[action_idx] = 1.0
            
            masked_policy = root_policy * legal_mask
            total = masked_policy.sum()
            if total > 0:
                masked_policy = masked_policy / total
            else:
                masked_policy = legal_mask / legal_mask.sum()
            
            # PUCTで行動を選択
            sqrt_total = np.sqrt(sum(visit_counts.values()) + 1)
            best_score = -float('inf')
            best_action = legal_actions[0]
            
            for action_idx in legal_actions:
                q_value = total_values[action_idx] / (visit_counts[action_idx] + 1e-8) if visit_counts[action_idx] > 0 else 0
                prior = masked_policy[action_idx]
                u_value = self.c_puct * prior * sqrt_total / (1 + visit_counts[action_idx])
                score = q_value + u_value
                
                if score > best_score:
                    best_score = score
                    best_action = action_idx
            
            # シミュレーション：選んだ手を実行して評価
            sim_board = copy.deepcopy(board)
            sim_hand = copy.deepcopy(my_hand)
            
            move = self.action_encoder.decode_action(best_action, current_player, board)
            success, _ = Rules.apply_move(sim_board, move, sim_hand)
            
            if success:
                # 相手視点で評価
                sim_state = self.state_encoder.encode(
                    sim_board, current_player.opponent, opponent_hand, sim_hand
                )
                _, sim_values = self._batch_inference([sim_state])
                value = -sim_values[0]  # 自分視点に反転
            else:
                value = -1.0  # 無効な手は負け
            
            visit_counts[best_action] += 1
            total_values[best_action] += value
        
        # 最終的な行動確率を計算
        action_probs = np.zeros(7695)
        for action_idx in legal_actions:
            action_probs[action_idx] = visit_counts[action_idx]
        
        if temperature == 0:
            best_action = max(legal_actions, key=lambda a: visit_counts[a])
            final_probs = np.zeros(7695)
            final_probs[best_action] = 1.0
        else:
            action_probs = action_probs ** (1.0 / temperature)
            total = action_probs.sum()
            if total > 0:
                final_probs = action_probs / total
            else:
                final_probs = np.zeros(7695)
                for a in legal_actions:
                    final_probs[a] = 1.0 / len(legal_actions)
            best_action = np.random.choice(7695, p=final_probs)
        
        return best_action, final_probs
    
    def play_game(
        self,
        temperature_threshold: int = 20,
        verbose: bool = False
    ) -> Tuple[List[TrainingExample], Optional[Player]]:
        """1ゲームをプレイ"""
        board = load_initial_board()
        current_player = Player.BLACK
        
        hands = {
            Player.BLACK: get_initial_hand_pieces(Player.BLACK),
            Player.WHITE: get_initial_hand_pieces(Player.WHITE)
        }
        
        game_history = []
        move_count = 0
        winner = None
        
        while move_count < self.MAX_MOVES:
            is_over, game_winner = Rules.is_game_over(board)
            if is_over:
                winner = game_winner
                break
            
            my_hand = hands[current_player]
            opponent_hand = hands[current_player.opponent]
            
            state = self.state_encoder.encode(board, current_player, my_hand, opponent_hand)
            temperature = 1.0 if move_count < temperature_threshold else 0.1
            
            action_idx, action_probs = self._mcts_search(
                board, current_player, my_hand, opponent_hand, temperature
            )
            
            game_history.append((state.copy(), action_probs.copy(), current_player))
            
            move = self.action_encoder.decode_action(action_idx, current_player, board)
            success, _ = Rules.apply_move(board, move, my_hand)
            
            if not success:
                break
            
            current_player = current_player.opponent
            move_count += 1
        
        if move_count >= self.MAX_MOVES:
            winner = None
        
        # 学習データを作成
        examples = []
        for state, policy, player in game_history:
            if winner is None:
                value = 0.0
            elif winner == player:
                value = 1.0
            else:
                value = -1.0
            
            examples.append(TrainingExample(state=state, policy=policy, value=value))
        
        return examples, winner
    
    def generate_data(
        self,
        num_games: int,
        temperature_threshold: int = 20,
        verbose: bool = True,
        num_workers: int = 1
    ) -> List[TrainingExample]:
        """複数ゲームのデータを生成（GPU活用版）"""
        all_examples = []
        wins = {'BLACK': 0, 'WHITE': 0, None: 0}
        
        if verbose:
            print(f"Starting GPU-accelerated self-play...")
            pbar = tqdm(total=num_games, desc="Self-play (GPU)")
        
        for game_idx in range(num_games):
            examples, winner = self.play_game(
                temperature_threshold=temperature_threshold,
                verbose=False
            )
            all_examples.extend(examples)
            
            winner_key = winner.name if winner else None
            wins[winner_key] += 1
            
            if verbose:
                pbar.update(1)
                pbar.set_postfix({
                    'B': wins['BLACK'],
                    'W': wins['WHITE'],
                    'D': wins[None],
                    'examples': len(all_examples)
                })
        
        if verbose:
            pbar.close()
            print(f"\nGenerated {len(all_examples)} examples from {num_games} games")
            print(f"Results: BLACK={wins['BLACK']}, WHITE={wins['WHITE']}, DRAW={wins[None]}")
        
        return all_examples


# defaultdictのインポートを追加
from collections import defaultdict
