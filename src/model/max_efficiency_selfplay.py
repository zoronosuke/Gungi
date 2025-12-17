"""
最大効率版の並列自己対戦
複数ゲームを並行して進行し、GPUをフル活用する
"""

import copy
import numpy as np
import torch
from threading import Thread, Lock, Event
from queue import Queue, Empty
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
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
class GameContext:
    """並行ゲームのコンテキスト"""
    game_id: int
    board: Board
    current_player: Player
    hands: Dict[Player, Dict[PieceType, int]]
    move_count: int
    history: List[Tuple[np.ndarray, np.ndarray, Player]]
    finished: bool = False
    winner: Optional[Player] = None
    
    # MCTS用の一時状態
    mcts_visit_counts: Dict[int, int] = field(default_factory=dict)
    mcts_total_values: Dict[int, float] = field(default_factory=dict)
    mcts_simulation: int = 0
    legal_actions: List[int] = field(default_factory=list)
    current_state: Optional[np.ndarray] = None
    temperature: float = 1.0


class MaxEfficiencySelfPlay:
    """
    最大効率版の自己対戦
    複数ゲームを並行して進行し、NN推論をバッチ処理
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
        num_parallel_games: int = 16  # 並行ゲーム数
    ):
        self.network = network
        self.state_encoder = state_encoder or StateEncoder()
        self.action_encoder = action_encoder or ActionEncoder()
        self.mcts_simulations = mcts_simulations
        self.c_puct = c_puct
        self.device = device
        self.num_parallel_games = num_parallel_games
        
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
    
    def _create_game_context(self, game_id: int) -> GameContext:
        """新しいゲームコンテキストを作成"""
        board = load_initial_board()
        hands = {
            Player.BLACK: get_initial_hand_pieces(Player.BLACK),
            Player.WHITE: get_initial_hand_pieces(Player.WHITE)
        }
        return GameContext(
            game_id=game_id,
            board=board,
            current_player=Player.BLACK,
            hands=hands,
            move_count=0,
            history=[]
        )
    
    def _get_legal_actions(self, ctx: GameContext) -> List[int]:
        """合法手のアクションインデックスを取得"""
        legal_moves = Rules.get_legal_moves(
            ctx.board, ctx.current_player, ctx.hands[ctx.current_player]
        )
        
        action_indices = []
        for move in legal_moves:
            action_idx = self.action_encoder.encode_move(move)
            if action_idx is not None:
                action_indices.append(action_idx)
        
        return action_indices
    
    def _select_action_puct(
        self, 
        ctx: GameContext, 
        policy: np.ndarray
    ) -> int:
        """PUCTでアクションを選択"""
        # 合法手でマスク
        legal_mask = np.zeros(7695)
        for action_idx in ctx.legal_actions:
            legal_mask[action_idx] = 1.0
        
        masked_policy = policy * legal_mask
        total = masked_policy.sum()
        if total > 0:
            masked_policy = masked_policy / total
        else:
            masked_policy = legal_mask / legal_mask.sum()
        
        sqrt_total = np.sqrt(sum(ctx.mcts_visit_counts.values()) + 1)
        best_score = -float('inf')
        best_action = ctx.legal_actions[0] if ctx.legal_actions else 0
        
        for action_idx in ctx.legal_actions:
            q_value = ctx.mcts_total_values.get(action_idx, 0) / (ctx.mcts_visit_counts.get(action_idx, 0) + 1e-8)
            prior = masked_policy[action_idx]
            u_value = self.c_puct * prior * sqrt_total / (1 + ctx.mcts_visit_counts.get(action_idx, 0))
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = action_idx
        
        return best_action
    
    def _finalize_action(self, ctx: GameContext) -> Tuple[int, np.ndarray]:
        """MCTSの結果から最終的な行動を選択"""
        action_probs = np.zeros(7695)
        for action_idx in ctx.legal_actions:
            action_probs[action_idx] = ctx.mcts_visit_counts.get(action_idx, 0)
        
        if ctx.temperature == 0 or ctx.temperature < 0.1:
            best_action = max(ctx.legal_actions, key=lambda a: ctx.mcts_visit_counts.get(a, 0))
            final_probs = np.zeros(7695)
            final_probs[best_action] = 1.0
        else:
            action_probs = action_probs ** (1.0 / ctx.temperature)
            total = action_probs.sum()
            if total > 0:
                final_probs = action_probs / total
            else:
                final_probs = np.zeros(7695)
                for a in ctx.legal_actions:
                    final_probs[a] = 1.0 / len(ctx.legal_actions)
            best_action = np.random.choice(7695, p=final_probs)
        
        return best_action, final_probs
    
    def _apply_action(self, ctx: GameContext, action_idx: int, action_probs: np.ndarray):
        """アクションを適用"""
        # 履歴に記録
        ctx.history.append((ctx.current_state.copy(), action_probs.copy(), ctx.current_player))
        
        # 手を適用
        move = self.action_encoder.decode_action(action_idx, ctx.current_player, ctx.board)
        my_hand = ctx.hands[ctx.current_player]
        success, _ = Rules.apply_move(ctx.board, move, my_hand)
        
        if not success:
            ctx.finished = True
            ctx.winner = ctx.current_player.opponent
            return
        
        # ゲーム終了チェック
        is_over, winner = Rules.is_game_over(ctx.board)
        if is_over:
            ctx.finished = True
            ctx.winner = winner
            return
        
        # 手番交代
        ctx.current_player = ctx.current_player.opponent
        ctx.move_count += 1
        
        # 最大手数チェック
        if ctx.move_count >= self.MAX_MOVES:
            ctx.finished = True
            ctx.winner = None
    
    def _reset_mcts_state(self, ctx: GameContext, temperature_threshold: int):
        """MCTSの状態をリセット"""
        ctx.mcts_visit_counts = defaultdict(int)
        ctx.mcts_total_values = defaultdict(float)
        ctx.mcts_simulation = 0
        ctx.legal_actions = self._get_legal_actions(ctx)
        ctx.temperature = 1.0 if ctx.move_count < temperature_threshold else 0.1
        
        # 現在の状態をエンコード
        my_hand = ctx.hands[ctx.current_player]
        opponent_hand = ctx.hands[ctx.current_player.opponent]
        ctx.current_state = self.state_encoder.encode(
            ctx.board, ctx.current_player, my_hand, opponent_hand
        )
    
    def generate_data(
        self,
        num_games: int,
        temperature_threshold: int = 20,
        verbose: bool = True,
        num_workers: int = 1  # 互換性のため（使用しない）
    ) -> List[TrainingExample]:
        """複数ゲームを並行してデータを生成"""
        all_examples = []
        wins = {'BLACK': 0, 'WHITE': 0, None: 0}
        completed_games = 0
        
        if verbose:
            print(f"Starting max-efficiency self-play ({self.num_parallel_games} parallel games)...")
            pbar = tqdm(total=num_games, desc="Self-play (max-eff)")
        
        # アクティブなゲーム
        active_games: List[GameContext] = []
        next_game_id = 0
        
        # 初期ゲームを作成
        while len(active_games) < self.num_parallel_games and next_game_id < num_games:
            ctx = self._create_game_context(next_game_id)
            self._reset_mcts_state(ctx, temperature_threshold)
            active_games.append(ctx)
            next_game_id += 1
        
        while active_games:
            # フェーズ1: ルートノード評価のためのバッチ推論
            states_to_eval = []
            games_needing_eval = []
            
            for ctx in active_games:
                if not ctx.finished and ctx.legal_actions:
                    states_to_eval.append(ctx.current_state)
                    games_needing_eval.append(ctx)
            
            if states_to_eval:
                policies, values = self._batch_inference(states_to_eval)
                
                # フェーズ2: 各ゲームのMCTSを1ステップ進める
                sim_states = []
                sim_games = []
                sim_actions = []
                
                for i, ctx in enumerate(games_needing_eval):
                    policy = policies[i]
                    
                    # PUCTでアクションを選択
                    action = self._select_action_puct(ctx, policy)
                    
                    # シミュレーション用の状態を作成
                    sim_board = copy.deepcopy(ctx.board)
                    sim_hand = copy.deepcopy(ctx.hands[ctx.current_player])
                    
                    move = self.action_encoder.decode_action(action, ctx.current_player, ctx.board)
                    success, _ = Rules.apply_move(sim_board, move, sim_hand)
                    
                    if success:
                        # 相手視点での状態をエンコード
                        opponent_hand = ctx.hands[ctx.current_player.opponent]
                        sim_state = self.state_encoder.encode(
                            sim_board, ctx.current_player.opponent, opponent_hand, sim_hand
                        )
                        sim_states.append(sim_state)
                        sim_games.append(ctx)
                        sim_actions.append(action)
                    else:
                        # 無効な手
                        ctx.mcts_visit_counts[action] += 1
                        ctx.mcts_total_values[action] -= 1.0
                        ctx.mcts_simulation += 1
                
                # フェーズ3: シミュレーション結果のバッチ推論
                if sim_states:
                    _, sim_values = self._batch_inference(sim_states)
                    
                    for j, ctx in enumerate(sim_games):
                        action = sim_actions[j]
                        value = -sim_values[j]  # 自分視点に反転
                        
                        ctx.mcts_visit_counts[action] += 1
                        ctx.mcts_total_values[action] += value
                        ctx.mcts_simulation += 1
            
            # フェーズ4: MCTS完了したゲームの処理
            games_to_remove = []
            
            for ctx in active_games:
                if ctx.finished:
                    games_to_remove.append(ctx)
                    continue
                
                if not ctx.legal_actions:
                    # 合法手がない = 負け
                    ctx.finished = True
                    ctx.winner = ctx.current_player.opponent
                    games_to_remove.append(ctx)
                    continue
                
                if ctx.mcts_simulation >= self.mcts_simulations:
                    # MCTS完了、手を選択して適用
                    action, action_probs = self._finalize_action(ctx)
                    self._apply_action(ctx, action, action_probs)
                    
                    if ctx.finished:
                        games_to_remove.append(ctx)
                    else:
                        # 次の手のためにMCTSをリセット
                        self._reset_mcts_state(ctx, temperature_threshold)
            
            # 完了したゲームを処理
            for ctx in games_to_remove:
                active_games.remove(ctx)
                completed_games += 1
                
                # 学習データを作成
                for state, policy, player in ctx.history:
                    if ctx.winner is None:
                        value = 0.0
                    elif ctx.winner == player:
                        value = 1.0
                    else:
                        value = -1.0
                    
                    all_examples.append(TrainingExample(state=state, policy=policy, value=value))
                
                winner_key = ctx.winner.name if ctx.winner else None
                wins[winner_key] += 1
                
                if verbose:
                    pbar.update(1)
                    pbar.set_postfix({
                        'B': wins['BLACK'],
                        'W': wins['WHITE'],
                        'D': wins[None],
                        'ex': len(all_examples),
                        'active': len(active_games)
                    })
                
                # 新しいゲームを追加
                if next_game_id < num_games:
                    new_ctx = self._create_game_context(next_game_id)
                    self._reset_mcts_state(new_ctx, temperature_threshold)
                    active_games.append(new_ctx)
                    next_game_id += 1
        
        if verbose:
            pbar.close()
            print(f"\nGenerated {len(all_examples)} examples from {num_games} games")
            print(f"Results: BLACK={wins['BLACK']}, WHITE={wins['WHITE']}, DRAW={wins[None]}")
        
        return all_examples
