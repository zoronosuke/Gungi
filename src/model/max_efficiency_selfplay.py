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
    draw_reason: Optional[str] = None  # 引き分けの理由
    
    # 千日手検出用
    position_history: Dict[str, int] = field(default_factory=dict)
    
    # 往復検出用（直前4手のアクションを記録）
    last_actions: List[int] = field(default_factory=list)
    
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
    AlphaZero/将棋AIの手法を参考に改良
    """
    
    MAX_MOVES = 300  # 軍儀は複雑なので300手まで許容
    REPETITION_THRESHOLD = 3  # 千日手判定を3回に（早期検出）
    
    # Dirichletノイズのパラメータ（探索の多様性を強化）
    DIRICHLET_ALPHA = 0.5   # さらに大きくして多様性を最大化（初期学習向け）
    DIRICHLET_EPSILON = 0.6  # ノイズの混合率を60%に（学習初期は探索重視）
    
    # 引き分けの評価値（千日手は中立、勝敗のみで学習）
    DRAW_VALUE_REPETITION = 0.0  # 千日手は中立（ペナルティなし）
    DRAW_VALUE_MAX_MOVES = 0.0   # 最大手数も中立
    
    def __init__(
        self,
        network,
        state_encoder: StateEncoder = None,
        action_encoder: ActionEncoder = None,
        mcts_simulations: int = 100,  # 将棋AIは800、最低でも100推奨
        c_puct: float = 1.5,
        device: str = 'cuda',
        num_parallel_games: int = 16,  # 並行ゲーム数
        use_dirichlet_noise: bool = True  # 探索の多様性
    ):
        self.network = network
        self.state_encoder = state_encoder or StateEncoder()
        self.action_encoder = action_encoder or ActionEncoder()
        self.mcts_simulations = mcts_simulations
        self.c_puct = c_puct
        self.device = device
        self.num_parallel_games = num_parallel_games
        self.use_dirichlet_noise = use_dirichlet_noise
        
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
        ctx = GameContext(
            game_id=game_id,
            board=board,
            current_player=Player.BLACK,
            hands=hands,
            move_count=0,
            history=[]
        )
        # 初期局面をposition_historyに登録
        initial_key = board.get_position_key(
            Player.BLACK, hands[Player.BLACK], hands[Player.WHITE]
        )
        ctx.position_history[initial_key] = 1
        return ctx
    
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
    
    def _is_back_and_forth(self, ctx: GameContext, action_idx: int) -> bool:
        """往復パターン（A→B→A→B）および長いサイクルを検出"""
        # 2手前（同じプレイヤーの前の手）と同じならTrue
        if len(ctx.last_actions) >= 2:
            if ctx.last_actions[-2] == action_idx:
                return True
        # 4手前（2手前と同じパターンの繰り返し）もチェック
        if len(ctx.last_actions) >= 4:
            if ctx.last_actions[-4] == action_idx:
                return True
        return False
    
    def _would_cause_repetition(self, ctx: GameContext, action_idx: int) -> bool:
        """この手を打つと既出局面に戻る、または往復パターンになるかチェック"""
        # 1. 往復パターンをチェック（最も頻繁）
        if self._is_back_and_forth(ctx, action_idx):
            return True
        
        # 2. 局面の繰り返しをチェック（計算コストが高いので後）
        sim_board = copy.deepcopy(ctx.board)
        sim_hand = copy.deepcopy(ctx.hands[ctx.current_player])
        opponent_hand = ctx.hands[ctx.current_player.opponent]
        
        move = self.action_encoder.decode_action(action_idx, ctx.current_player, ctx.board)
        if move is None:
            return False
        success, _ = Rules.apply_move(sim_board, move, sim_hand)
        
        if not success:
            return False
        
        # 手を打った後の局面キーを計算（手番交代後の相手視点）
        next_key = sim_board.get_position_key(
            ctx.current_player.opponent, opponent_hand, sim_hand
        )
        
        # 既に2回出現した局面に行くと千日手になる（REPETITION_THRESHOLD - 1）
        return ctx.position_history.get(next_key, 0) >= (self.REPETITION_THRESHOLD - 1)
    
    def _select_action_puct(
        self, 
        ctx: GameContext, 
        policy: np.ndarray,
        is_root: bool = False
    ) -> int:
        """PUCTでアクションを選択（AlphaZeroスタイル + 循環回避）"""
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
        
        # ルートノードにDirichletノイズを追加（探索の多様性）
        if is_root and self.use_dirichlet_noise and len(ctx.legal_actions) > 0:
            noise = np.random.dirichlet([self.DIRICHLET_ALPHA] * len(ctx.legal_actions))
            noise_full = np.zeros(7695)
            for i, action_idx in enumerate(ctx.legal_actions):
                noise_full[action_idx] = noise[i]
            
            masked_policy = (1 - self.DIRICHLET_EPSILON) * masked_policy + self.DIRICHLET_EPSILON * noise_full
        
        sqrt_total = np.sqrt(sum(ctx.mcts_visit_counts.values()) + 1)
        best_score = -float('inf')
        best_action = ctx.legal_actions[0] if ctx.legal_actions else 0
        
        # 循環する手を検出して除外（常にチェック、学習中は特に重要）
        repetition_actions = set()
        for action_idx in ctx.legal_actions:
            if self._would_cause_repetition(ctx, action_idx):
                repetition_actions.add(action_idx)
        
        # 非循環手のみを候補にする（循環手しかない場合のみ全て候補）
        candidate_actions = [a for a in ctx.legal_actions if a not in repetition_actions]
        if not candidate_actions:
            # 本当に循環手しかない場合のみ許可（稀なケース）
            candidate_actions = ctx.legal_actions
        
        for action_idx in candidate_actions:
            q_value = ctx.mcts_total_values.get(action_idx, 0) / (ctx.mcts_visit_counts.get(action_idx, 0) + 1e-8)
            prior = masked_policy[action_idx]
            u_value = self.c_puct * prior * sqrt_total / (1 + ctx.mcts_visit_counts.get(action_idx, 0))
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = action_idx
        
        return best_action
    
    def _finalize_action(self, ctx: GameContext) -> Tuple[int, np.ndarray]:
        """MCTSの結果から最終的な行動を選択（循環回避）"""
        action_probs = np.zeros(7695)
        
        # 循環する手を検出
        repetition_actions = set()
        for action_idx in ctx.legal_actions:
            if self._would_cause_repetition(ctx, action_idx):
                repetition_actions.add(action_idx)
        
        # 非循環手のみを候補にする
        non_rep_actions = [a for a in ctx.legal_actions if a not in repetition_actions]
        candidate_actions = non_rep_actions if non_rep_actions else ctx.legal_actions
        
        # 候補手のみで確率を計算
        for action_idx in candidate_actions:
            count = ctx.mcts_visit_counts.get(action_idx, 0)
            action_probs[action_idx] = count
        
        if ctx.temperature == 0 or ctx.temperature < 0.1:
            # 最も訪問回数が多い手を選ぶ
            best_action = max(candidate_actions, key=lambda a: ctx.mcts_visit_counts.get(a, 0))
            final_probs = np.zeros(7695)
            final_probs[best_action] = 1.0
        else:
            # 温度を適用
            if action_probs.sum() > 0:
                action_probs = action_probs ** (1.0 / ctx.temperature)
                total = action_probs.sum()
                if total > 0:
                    final_probs = action_probs / total
                else:
                    final_probs = np.zeros(7695)
                    for a in candidate_actions:
                        final_probs[a] = 1.0 / len(candidate_actions)
            else:
                # 訪問回数がない場合は均等
                final_probs = np.zeros(7695)
                for a in candidate_actions:
                    final_probs[a] = 1.0 / len(candidate_actions)
            
            # 確率に従って選択
            best_action = np.random.choice(7695, p=final_probs)
        
        return best_action, final_probs
    
    def _apply_action(self, ctx: GameContext, action_idx: int, action_probs: np.ndarray):
        """アクションを適用"""
        # 履歴に記録
        ctx.history.append((ctx.current_state.copy(), action_probs.copy(), ctx.current_player))
        
        # last_actionsに記録（往復検出用、最新4手を保持）
        ctx.last_actions.append(action_idx)
        if len(ctx.last_actions) > 4:
            ctx.last_actions.pop(0)
        
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
        """局面のハッシュを生成（千日手検出用）- 持ち駒を含む完全版"""
        my_hand = ctx.hands[ctx.current_player]
        opponent_hand = ctx.hands[ctx.current_player.opponent]
        return ctx.board.get_position_key(ctx.current_player, my_hand, opponent_hand)
    
    def _reset_mcts_state(self, ctx: GameContext, temperature_threshold: int):
        """MCTSの状態をリセット"""
        ctx.mcts_visit_counts = defaultdict(int)
        ctx.mcts_total_values = defaultdict(float)
        ctx.mcts_simulation = 0
        ctx.legal_actions = self._get_legal_actions(ctx)
        
        # 温度スケジューリング（初期学習向けに高温を維持）
        # 序盤: 高温(2.0) - 非常に多様な手を探索（学習初期は特に重要）
        # 中盤: 高温維持(1.5)
        # 終盤: 中温(1.0) - 多様性を維持しつつ少し収束
        if ctx.move_count < temperature_threshold:
            ctx.temperature = 2.0  # 序盤は最大温度
        elif ctx.move_count < temperature_threshold * 3:
            ctx.temperature = 1.5  # 中盤も高温維持
        else:
            ctx.temperature = 1.0  # 終盤でも十分な多様性
        
        # 現在の状態をエンコード（局面履歴を含む）
        my_hand = ctx.hands[ctx.current_player]
        opponent_hand = ctx.hands[ctx.current_player.opponent]
        ctx.current_state = self.state_encoder.encode(
            ctx.board, ctx.current_player, my_hand, opponent_hand,
            position_history=ctx.position_history
        )
    
    def generate_data(
        self,
        num_games: int,
        temperature_threshold: int = 30,
        verbose: bool = True,
        num_workers: int = 1  # 互換性のため（使用しない）
    ) -> List[TrainingExample]:
        """複数ゲームを並行してデータを生成"""
        all_examples = []
        wins = {'BLACK': 0, 'WHITE': 0, None: 0}
        draw_reasons = {'REPETITION': 0, 'MAX_MOVES': 0, 'NO_LEGAL_MOVES': 0, 'OTHER': 0}
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
                    
                    # PUCTでアクションを選択（最初のシミュレーションはルート）
                    is_root = (ctx.mcts_simulation == 0)
                    action = self._select_action_puct(ctx, policy, is_root=is_root)
                    
                    # シミュレーション用の状態を作成
                    sim_board = copy.deepcopy(ctx.board)
                    sim_hand = copy.deepcopy(ctx.hands[ctx.current_player])
                    
                    move = self.action_encoder.decode_action(action, ctx.current_player, ctx.board)
                    success, _ = Rules.apply_move(sim_board, move, sim_hand)
                    
                    if success:
                        # 相手視点での状態をエンコード（局面履歴を含む）
                        opponent_hand = ctx.hands[ctx.current_player.opponent]
                        sim_state = self.state_encoder.encode(
                            sim_board, ctx.current_player.opponent, opponent_hand, sim_hand,
                            position_history=ctx.position_history
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
                
                # 引き分けの理由を記録
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
                print(f"Draw reasons: REPETITION={draw_reasons['REPETITION']}, MAX_MOVES={draw_reasons['MAX_MOVES']}, OTHER={draw_reasons['NO_LEGAL_MOVES'] + draw_reasons['OTHER']}")
        
        return all_examples
