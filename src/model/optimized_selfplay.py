"""
最適化版の並列自己対戦 - GPU使用率を最大化
RTX 3060 Ti (8GB VRAM)向けに最適化
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F
from threading import Thread
from queue import Queue
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
    draw_reason: Optional[str] = None  # 引き分けの理由
    
    # 千日手検出用
    position_history: Dict[str, int] = field(default_factory=dict)
    
    # MCTS用の一時状態
    mcts_visit_counts: Dict[int, int] = field(default_factory=dict)
    mcts_total_values: Dict[int, float] = field(default_factory=dict)
    mcts_simulation: int = 0
    legal_actions: List[int] = field(default_factory=list)
    current_state: Optional[np.ndarray] = None
    temperature: float = 1.0
    cached_policy: Optional[np.ndarray] = None


class OptimizedSelfPlay:
    """
    GPU最大効率版の自己対戦
    
    最適化ポイント:
    1. より大きなバッチサイズでGPU使用率向上
    2. Virtual Lossで並列MCTS
    3. TensorCoreを活用するため半精度（FP16）対応
    4. メモリ効率の良い状態管理
    5. Dirichletノイズで探索の多様性確保
    """
    
    MAX_MOVES = 300  # 軍儀は複雑なので300手まで許容
    REPETITION_THRESHOLD = 3  # 千日手判定を3回に（早めに検出）
    
    # 引き分けの評価値（千日手と最大手数で区別）
    DRAW_VALUE_REPETITION = -0.9  # 千日手は強いペナルティ（同じ手の繰り返しは悪い）
    DRAW_VALUE_MAX_MOVES = -0.2   # 最大手数到達は中程度のペナルティ（積極的に勝ちを目指す）
    
    # Dirichletノイズ（AlphaZeroスタイル）
    DIRICHLET_ALPHA = 0.15  # より小さく（将棋と同じ）
    DIRICHLET_EPSILON = 0.25
    
    def __init__(
        self,
        network,
        state_encoder: StateEncoder = None,
        action_encoder: ActionEncoder = None,
        mcts_simulations: int = 100,  # より多くのシミュレーション
        c_puct: float = 1.5,
        device: str = 'cuda',
        num_parallel_games: int = 64,  # より多くの並行ゲーム
        virtual_loss: float = 3.0,  # Virtual Loss
        use_fp16: bool = True,  # 半精度
    ):
        self.network = network
        self.state_encoder = state_encoder or StateEncoder()
        self.action_encoder = action_encoder or ActionEncoder()
        self.mcts_simulations = mcts_simulations
        self.c_puct = c_puct
        self.device = device
        self.num_parallel_games = num_parallel_games
        self.virtual_loss = virtual_loss
        self.use_fp16 = use_fp16 and device == 'cuda'
        
        self.network.to(device)
        self.network.eval()
        
        # FP16用のスケーラー
        if self.use_fp16:
            print("Using FP16 (half precision) for faster inference")
    
    @torch.no_grad()
    def _batch_inference(self, states: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """最適化されたバッチNN推論"""
        if not states:
            return np.array([]), np.array([])
        
        states_array = np.stack(states)
        states_tensor = torch.from_numpy(states_array).to(self.device)
        
        if self.use_fp16:
            states_tensor = states_tensor.half()
            with torch.amp.autocast('cuda'):
                log_policies, values = self.network(states_tensor)
        else:
            states_tensor = states_tensor.float()
            log_policies, values = self.network(states_tensor)
        
        # Softmaxはlog_policyから直接
        policies = torch.exp(log_policies).float().cpu().numpy()
        values = values.float().cpu().numpy().flatten()
        
        return policies, values
    
    @torch.no_grad()
    def _batch_inference_large(self, states: List[np.ndarray], max_batch: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """
        大きなバッチを分割して処理（VRAM不足対策）
        """
        if not states:
            return np.array([]), np.array([])
        
        all_policies = []
        all_values = []
        
        for i in range(0, len(states), max_batch):
            batch = states[i:i + max_batch]
            policies, values = self._batch_inference(batch)
            all_policies.append(policies)
            all_values.append(values)
        
        return np.concatenate(all_policies), np.concatenate(all_values)
    
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
    
    def _select_action_puct_with_virtual_loss(
        self, 
        ctx: GameContext, 
        policy: np.ndarray,
        num_selections: int = 1,
        add_noise: bool = False
    ) -> List[int]:
        """
        Virtual Lossを使用して複数のアクションを同時に選択
        これにより、同じゲームで複数のMCTSシミュレーションを並列実行可能
        """
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
        
        # Dirichletノイズを追加（ルートノードのみ）
        if add_noise and len(ctx.legal_actions) > 0:
            noise = np.random.dirichlet([self.DIRICHLET_ALPHA] * len(ctx.legal_actions))
            noise_full = np.zeros(7695)
            for i, action_idx in enumerate(ctx.legal_actions):
                noise_full[action_idx] = noise[i]
            masked_policy = (1 - self.DIRICHLET_EPSILON) * masked_policy + self.DIRICHLET_EPSILON * noise_full
        
        selected_actions = []
        temp_visit_counts = defaultdict(int, ctx.mcts_visit_counts)
        
        for _ in range(num_selections):
            sqrt_total = np.sqrt(sum(temp_visit_counts.values()) + 1)
            best_score = -float('inf')
            best_action = ctx.legal_actions[0] if ctx.legal_actions else 0
            
            for action_idx in ctx.legal_actions:
                n_visits = temp_visit_counts.get(action_idx, 0)
                q_value = ctx.mcts_total_values.get(action_idx, 0) / (n_visits + 1e-8)
                prior = masked_policy[action_idx]
                u_value = self.c_puct * prior * sqrt_total / (1 + n_visits)
                score = q_value + u_value
                
                if score > best_score:
                    best_score = score
                    best_action = action_idx
            
            selected_actions.append(best_action)
            # Virtual Lossを適用（同じアクションが連続で選ばれないように）
            temp_visit_counts[best_action] += 1
        
        return selected_actions
    
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
            ctx.draw_reason = "MOVE_FAILED"
            return
        
        # ゲーム終了チェック
        is_over, winner = Rules.is_game_over(ctx.board)
        if is_over:
            ctx.finished = True
            ctx.winner = winner
            return
        
        # 千日手チェック（盤面のハッシュを使用）
        position_key = self._get_position_hash(ctx)
        ctx.position_history[position_key] = ctx.position_history.get(position_key, 0) + 1
        
        if ctx.position_history[position_key] >= self.REPETITION_THRESHOLD:
            # 同じ局面が閾値回出現したら千日手
            ctx.finished = True
            ctx.winner = None
            ctx.draw_reason = "REPETITION"
            return
        
        # 手番交代
        ctx.current_player = ctx.current_player.opponent
        ctx.move_count += 1
        
        # 最大手数チェック
        if ctx.move_count >= self.MAX_MOVES:
            ctx.finished = True
            ctx.winner = None
            ctx.draw_reason = "MAX_MOVES"
    
    def _get_position_hash(self, ctx: GameContext) -> str:
        """局面のハッシュを生成（千日手検出用）"""
        board_str = ""
        for r in range(9):
            for c in range(9):
                stack = ctx.board.get_stack((r, c))
                if stack.is_empty():
                    board_str += "."
                else:
                    for piece in stack.pieces:
                        board_str += f"{piece.piece_type.name[0]}{piece.owner.name[0]}"
                    board_str += "|"
        board_str += ctx.current_player.name
        return board_str
    
    def _reset_mcts_state(self, ctx: GameContext, temperature_threshold: int):
        """MCTSの状態をリセット"""
        ctx.mcts_visit_counts = defaultdict(int)
        ctx.mcts_total_values = defaultdict(float)
        ctx.mcts_simulation = 0
        ctx.legal_actions = self._get_legal_actions(ctx)
        ctx.temperature = 1.0 if ctx.move_count < temperature_threshold else 0.1
        ctx.cached_policy = None
        
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
        """
        複数ゲームを並行してデータを生成
        最適化: バッチサイズを大きくし、Virtual Lossで並列MCTS
        """
        all_examples = []
        wins = {'BLACK': 0, 'WHITE': 0, None: 0}
        draw_reasons = {'REPETITION': 0, 'MAX_MOVES': 0, 'OTHER': 0}
        completed_games = 0
        
        if verbose:
            print(f"Starting optimized self-play ({self.num_parallel_games} parallel games, FP16={self.use_fp16})...")
            pbar = tqdm(total=num_games, desc="Self-play (optimized)")
        
        # アクティブなゲーム
        active_games: List[GameContext] = []
        next_game_id = 0
        
        # 初期ゲームを作成
        while len(active_games) < self.num_parallel_games and next_game_id < num_games:
            ctx = self._create_game_context(next_game_id)
            self._reset_mcts_state(ctx, temperature_threshold)
            active_games.append(ctx)
            next_game_id += 1
        
        # 各MCTSシミュレーションで複数のアクションを並列評価
        sims_per_batch = 4  # 1回のバッチで複数のシミュレーションを進める
        
        while active_games:
            # フェーズ1: ルートノードのポリシーを取得（まだ持っていないゲームのみ）
            states_for_policy = []
            games_for_policy = []
            
            for ctx in active_games:
                if not ctx.finished and ctx.legal_actions and ctx.cached_policy is None:
                    states_for_policy.append(ctx.current_state)
                    games_for_policy.append(ctx)
            
            if states_for_policy:
                policies, _ = self._batch_inference_large(states_for_policy)
                for i, ctx in enumerate(games_for_policy):
                    ctx.cached_policy = policies[i]
            
            # フェーズ2: 各ゲームのMCTSを複数ステップ進める（Virtual Loss）
            all_sim_states = []
            all_sim_contexts = []  # (ctx, action)のペア
            
            for ctx in active_games:
                if ctx.finished or not ctx.legal_actions:
                    continue
                
                if ctx.cached_policy is None:
                    continue
                
                # 残りのシミュレーション数
                remaining_sims = self.mcts_simulations - ctx.mcts_simulation
                batch_sims = min(sims_per_batch, remaining_sims)
                
                if batch_sims <= 0:
                    continue
                
                # Virtual Lossで複数アクションを選択
                # 最初のシミュレーションではDirichletノイズを追加
                add_noise = (ctx.mcts_simulation == 0)
                selected_actions = self._select_action_puct_with_virtual_loss(
                    ctx, ctx.cached_policy, batch_sims, add_noise=add_noise
                )
                
                for action in selected_actions:
                    # シミュレーション用の状態を作成
                    sim_board = copy.deepcopy(ctx.board)
                    sim_hand = copy.deepcopy(ctx.hands[ctx.current_player])
                    
                    move = self.action_encoder.decode_action(action, ctx.current_player, ctx.board)
                    success, _ = Rules.apply_move(sim_board, move, sim_hand)
                    
                    if success:
                        opponent_hand = ctx.hands[ctx.current_player.opponent]
                        sim_state = self.state_encoder.encode(
                            sim_board, ctx.current_player.opponent, opponent_hand, sim_hand
                        )
                        all_sim_states.append(sim_state)
                        all_sim_contexts.append((ctx, action))
                    else:
                        # 無効な手
                        ctx.mcts_visit_counts[action] += 1
                        ctx.mcts_total_values[action] -= 1.0
                        ctx.mcts_simulation += 1
            
            # フェーズ3: 大きなバッチでシミュレーション結果を推論
            if all_sim_states:
                _, sim_values = self._batch_inference_large(all_sim_states)
                
                for j, (ctx, action) in enumerate(all_sim_contexts):
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
                
                # 学習データを作成（千日手と最大手数到達で異なるペナルティ）
                for state, policy, player in ctx.history:
                    if ctx.winner is None:
                        # 引き分けの理由によってペナルティを変える
                        if ctx.draw_reason == "REPETITION":
                            value = self.DRAW_VALUE_REPETITION  # 千日手は強いペナルティ
                        else:
                            value = self.DRAW_VALUE_MAX_MOVES   # 最大手数は軽いペナルティ
                    elif ctx.winner == player:
                        value = 1.0
                    else:
                        value = -1.0
                    
                    all_examples.append(TrainingExample(state=state, policy=policy, value=value))
                
                winner_key = ctx.winner.name if ctx.winner else None
                wins[winner_key] += 1
                
                # 引き分け理由を記録
                if ctx.winner is None and ctx.draw_reason:
                    if ctx.draw_reason in draw_reasons:
                        draw_reasons[ctx.draw_reason] += 1
                    else:
                        draw_reasons['OTHER'] += 1
                
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
            if wins[None] > 0:
                print(f"Draw reasons: REPETITION={draw_reasons['REPETITION']}, MAX_MOVES={draw_reasons['MAX_MOVES']}, OTHER={draw_reasons['OTHER']}")
            avg_moves = len(all_examples) / num_games if num_games > 0 else 0
            print(f"Average moves per game: {avg_moves:.1f}")
        
        return all_examples


# 推奨設定
def get_recommended_settings(vram_gb: float = 8.0) -> dict:
    """
    VRAM量に応じた推奨設定を返す
    
    RTX 3060 Ti (8GB): 
    - parallel_games: 64
    - mcts_sims: 100
    - batch_size: 512
    
    より強いAIを目指す場合:
    - mcts_sims: 200-400
    - games_per_iteration: 50-100
    """
    if vram_gb >= 8:
        return {
            'num_parallel_games': 64,
            'mcts_simulations': 100,
            'batch_size': 512,
            'use_fp16': True,
            'games_per_iteration': 50,
        }
    elif vram_gb >= 6:
        return {
            'num_parallel_games': 32,
            'mcts_simulations': 100,
            'batch_size': 256,
            'use_fp16': True,
            'games_per_iteration': 30,
        }
    else:
        return {
            'num_parallel_games': 16,
            'mcts_simulations': 50,
            'batch_size': 128,
            'use_fp16': True,
            'games_per_iteration': 20,
        }
