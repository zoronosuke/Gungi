"""
モンテカルロ木探索（MCTS）の実装
AlphaZero型の探索アルゴリズム
"""

import math
import copy
import numpy as np
import torch
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from ..engine.board import Board
from ..engine.piece import Player, PieceType
from ..engine.move import Move
from ..engine.rules import Rules
from .encoder import StateEncoder, ActionEncoder


@dataclass
class GameState:
    """ゲームの状態を保持するクラス"""
    board: Board
    player: Player
    my_hand: Dict[PieceType, int]
    opponent_hand: Dict[PieceType, int]
    position_history: Dict[str, int] = None  # 局面履歴（千日手対策用）
    
    def __post_init__(self):
        """position_historyの初期化"""
        if self.position_history is None:
            self.position_history = {}
    
    def copy(self) -> 'GameState':
        """ディープコピーを作成"""
        return GameState(
            board=self.board.copy(),
            player=self.player,
            my_hand=copy.deepcopy(self.my_hand),
            opponent_hand=copy.deepcopy(self.opponent_hand),
            position_history=copy.deepcopy(self.position_history)
        )
    
    def switch_player(self):
        """手番を交代（持ち駒も入れ替え）"""
        self.player = self.player.opponent
        self.my_hand, self.opponent_hand = self.opponent_hand, self.my_hand
    
    def update_position_history(self):
        """現在の局面をposition_historyに追加"""
        position_key = self.board.get_position_key(
            self.player, self.my_hand, self.opponent_hand
        )
        self.position_history[position_key] = self.position_history.get(position_key, 0) + 1


class MCTSNode:
    """MCTSの探索木のノード"""
    
    def __init__(
        self,
        state: GameState,
        parent: Optional['MCTSNode'] = None,
        action: Optional[int] = None,  # 行動インデックス
        prior: float = 0.0
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        
        self.children: Dict[int, 'MCTSNode'] = {}  # action_idx -> child
        self.visit_count: int = 0
        self.value_sum: float = 0.0
    
    @property
    def mean_value(self) -> float:
        """平均価値"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, c_puct: float = 1.5) -> float:
        """
        PUCTスコアを計算
        
        PUCT = Q + c_puct * P * sqrt(N_parent) / (1 + N)
        """
        if self.parent is None:
            return 0.0
        
        # Q値（平均価値）- 親視点なので符号反転
        q_value = -self.mean_value
        
        # U値（探索ボーナス）
        u_value = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        
        return q_value + u_value
    
    def is_expanded(self) -> bool:
        """展開済みかどうか"""
        return len(self.children) > 0
    
    def select_child(self, c_puct: float = 1.5) -> 'MCTSNode':
        """UCBスコアが最大の子ノードを選択"""
        best_score = -float('inf')
        best_child = None
        
        for child in self.children.values():
            score = child.ucb_score(c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand(self, policy: np.ndarray, legal_mask: np.ndarray,
               action_encoder: ActionEncoder):
        """
        ノードを展開
        
        Args:
            policy: (7695,) 各行動の確率
            legal_mask: (7695,) 合法手マスク
            action_encoder: ActionEncoderインスタンス
        """
        # 合法手のみを展開
        legal_actions = np.where(legal_mask > 0)[0]
        
        for action_idx in legal_actions:
            prior = policy[action_idx]
            
            # 手を適用した新しい状態を作成
            new_state = self.state.copy()
            move = action_encoder.decode_action(
                action_idx, 
                new_state.player,
                new_state.board
            )
            
            # 手を適用
            success, captured = Rules.apply_move(
                new_state.board, 
                move, 
                new_state.my_hand
            )
            
            if success:
                # 局面履歴を更新（千日手検出用）
                new_state.update_position_history()
                
                # 手番を交代
                new_state.switch_player()
                
                # 子ノードを作成
                child = MCTSNode(
                    state=new_state,
                    parent=self,
                    action=action_idx,
                    prior=prior
                )
                self.children[action_idx] = child
    
    def backpropagate(self, value: float):
        """価値をルートまで逆伝播"""
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # 手番が変わるので符号反転
            node = node.parent


class MCTS:
    """モンテカルロ木探索
    
    Value予測の0収束問題対策:
    - Dirichlet noiseを追加して探索多様性を維持
    - ルートノードの事前確率にノイズを混ぜることで、
      Policy Networkが特定の手に過度に集中することを防ぐ
    - 局面ハッシュベースの千日手検出
    """
    
    # Dirichlet noiseパラメータ（探索多様性を大幅強化）
    DIRICHLET_ALPHA = 0.3    # ノイズの集中度（将棋は0.3が標準）
    DIRICHLET_EPSILON = 0.5  # ノイズの混合比率（50%に強化 - 千日手対策）
    
    # 千日手ペナルティ（局面繰り返し検出時）
    REPETITION_PENALTY = -1.0   # 千日手は完全敗北扱い（-0.99から強化）
    
    def __init__(
        self,
        network,
        state_encoder: StateEncoder = None,
        action_encoder: ActionEncoder = None,
        c_puct: float = 3.0,  # 探索幅を拡大（1.5→3.0、千日手対策）
        num_simulations: int = 50,
        device: str = 'cuda',
        dirichlet_alpha: float = None,
        dirichlet_epsilon: float = None,
        default_temperature: float = 2.0  # 確率分布平滑化（1.0→2.0）
    ):
        self.network = network
        self.state_encoder = state_encoder or StateEncoder()
        self.action_encoder = action_encoder or ActionEncoder()
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.device = device
        self.default_temperature = default_temperature  # デフォルト温度（千日手対策で高めに設定）
        
        # Dirichlet noiseパラメータ（オーバーライド可能）
        self.dirichlet_alpha = dirichlet_alpha if dirichlet_alpha is not None else self.DIRICHLET_ALPHA
        self.dirichlet_epsilon = dirichlet_epsilon if dirichlet_epsilon is not None else self.DIRICHLET_EPSILON
    
    def search(
        self,
        board: Board,
        player: Player,
        my_hand: Dict[PieceType, int],
        opponent_hand: Dict[PieceType, int],
        temperature: float = 2.0,  # 確率分布平滑化（千日手対策で高めに設定）
        position_history: Dict[str, int] = None
    ) -> Tuple[int, np.ndarray]:
        """
        MCTSで探索を行う
        
        Args:
            board: 現在の盤面
            player: 現在のプレイヤー
            my_hand: 自分の持ち駒
            opponent_hand: 相手の持ち駒
            temperature: 行動選択の温度パラメータ
                        1.0: 訪問回数に比例した確率で選択
                        0.0(または0に近い値): 最も訪問回数が多い手を選択
            position_history: 局面履歴 {position_key: 出現回数}（千日手対策用）
        
        Returns:
            best_action: 選択された行動インデックス
            action_probs: (7695,) 各行動の選択確率（学習データ用）
        """
        # ルートノードを作成
        root_state = GameState(
            board=board.copy(),
            player=player,
            my_hand=copy.deepcopy(my_hand),
            opponent_hand=copy.deepcopy(opponent_hand),
            position_history=copy.deepcopy(position_history) if position_history else {}
        )
        root = MCTSNode(state=root_state)
        
        # ルートノードを展開
        value = self._evaluate_and_expand(root)
        
        # ルートノードにDirichlet noiseを追加（探索多様性の強化）
        self._add_dirichlet_noise(root)
        
        # シミュレーションを実行
        for _ in range(self.num_simulations):
            node = root
            visited_in_path = set()  # このシミュレーション内で訪問した局面
            is_cycle = False
            
            # 1. Selection: 葉ノードまで降下
            while node.is_expanded():
                # 循環検出: 局面キーを取得
                position_key = node.state.board.get_position_key(
                    node.state.player,
                    node.state.my_hand,
                    node.state.opponent_hand
                )
                
                # 【重要】対局履歴からの千日手チェック（3回以上同一局面 = 千日手）
                history_count = node.state.position_history.get(position_key, 0)
                
                if position_key in visited_in_path or history_count >= 3:
                    # 千日手検出：
                    # 1) この探索パス内でループした場合
                    # 2) 対局全体で既に3回以上この局面が出現している場合
                    # → 千日手ペナルティで終端（-0.99 = ほぼ負け扱い）
                    node.backpropagate(self.REPETITION_PENALTY)
                    is_cycle = True
                    break
                
                visited_in_path.add(position_key)
                
                child = node.select_child(self.c_puct)
                if child is None:
                    break
                node = child
            
            if is_cycle:
                continue  # 次のシミュレーションへ
            
            # 2. ゲーム終了チェック
            is_over, winner = Rules.is_game_over(node.state.board)
            
            if is_over:
                # 終了局面の価値（現在のノードのプレイヤー視点）
                if winner is None:
                    value = 0.0  # 引き分けは中立
                elif winner == node.state.player:
                    value = 1.0
                else:
                    value = -1.0
            else:
                # 3. Expansion & Evaluation
                value = self._evaluate_and_expand(node)
            
            # 4. Backpropagation
            node.backpropagate(value)
        
        # 訪問回数から方策を計算
        action_probs = np.zeros(self.action_encoder.ACTION_SIZE, dtype=np.float32)
        visit_counts = {}
        
        for action_idx, child in root.children.items():
            visit_counts[action_idx] = child.visit_count
            action_probs[action_idx] = child.visit_count
        
        # 温度を適用
        if temperature < 0.01:
            # 温度が0に近い場合は最大訪問回数の手を選択
            best_action = max(visit_counts.keys(), key=lambda a: visit_counts[a])
            action_probs = np.zeros_like(action_probs)
            action_probs[best_action] = 1.0
        else:
            # 温度を適用して確率分布を計算
            if action_probs.sum() > 0:
                action_probs = action_probs ** (1.0 / temperature)
                action_probs = action_probs / action_probs.sum()
            
            # 確率に従って選択
            if action_probs.sum() > 0:
                best_action = np.random.choice(
                    len(action_probs), 
                    p=action_probs
                )
            else:
                # 合法手がない（通常は起きない）
                best_action = 0
        
        return best_action, action_probs
    
    def _add_dirichlet_noise(self, node: MCTSNode):
        """
        ルートノードにDirichlet noiseを追加
        
        Value予測の0収束問題対策:
        事前確率にノイズを混ぜることで、Policy Networkが
        特定の「安全な手」に過度に集中することを防ぎ、
        探索の多様性を維持する。
        
        AlphaZero論文に基づく実装:
        P(a) = (1-ε) * P(a) + ε * Dir(α)
        """
        if not node.is_expanded() or len(node.children) == 0:
            return
        
        # 子ノードの数だけDirichlet分布からサンプリング
        num_actions = len(node.children)
        noise = np.random.dirichlet([self.dirichlet_alpha] * num_actions)
        
        # 各子ノードのpriorにノイズを混合
        for i, child in enumerate(node.children.values()):
            child.prior = (1 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * noise[i]
    
    def _evaluate_and_expand(self, node: MCTSNode) -> float:
        """
        ノードをニューラルネットワークで評価し、展開する
        
        千日手検出:
        葉ノードが千日手局面（3回以上同一局面）の場合、
        ネットワーク評価を行わず即座にペナルティを返す。
        
        Returns:
            value: 評価値（現在のプレイヤー視点）
        """
        state = node.state
        
        # 千日手チェック: 同一局面が3回以上出現していれば即座にペナルティ
        position_key = state.board.get_position_key(
            state.player, state.my_hand, state.opponent_hand
        )
        history_count = state.position_history.get(position_key, 0)
        if history_count >= 3:
            # 千日手局面 → ネットワーク評価せず敗北扱い
            return self.REPETITION_PENALTY
        
        # 合法手を取得
        legal_moves = Rules.get_legal_moves(
            state.board, 
            state.player, 
            state.my_hand
        )
        
        if not legal_moves:
            # 合法手がない = 負け
            return -1.0
        
        # 状態をエンコード（局面履歴を含む）
        state_tensor = self.state_encoder.encode(
            state.board,
            state.player,
            state.my_hand,
            state.opponent_hand,
            position_history=state.position_history
        )
        state_tensor = torch.from_numpy(state_tensor).unsqueeze(0).to(self.device)
        
        # 合法手マスクを生成
        legal_mask = self.action_encoder.get_legal_mask(
            state.board,
            state.player,
            state.my_hand,
            legal_moves
        )
        
        # ニューラルネットで評価
        policy, value = self.network.predict(state_tensor, legal_mask)
        
        # ノードを展開
        node.expand(policy, legal_mask, self.action_encoder)
        
        return value
    
    def get_best_move(
        self,
        board: Board,
        player: Player,
        my_hand: Dict[PieceType, int],
        opponent_hand: Dict[PieceType, int]
    ) -> Move:
        """
        最善手をMoveオブジェクトで返す（推論用）
        """
        action_idx, _ = self.search(
            board, player, my_hand, opponent_hand,
            temperature=0.1  # 推論時は低温度
        )
        
        return self.action_encoder.decode_action(action_idx, player, board)


if __name__ == "__main__":
    # テスト
    from ..engine.initial_setup import load_initial_board, get_initial_hand_pieces
    from .network import create_model
    
    print("=== MCTS Test ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # モデルを作成
    model = create_model(device, test_mode=True)
    
    # MCTSを作成
    mcts = MCTS(
        network=model,
        num_simulations=10,  # テスト用に少なく
        device=device
    )
    
    # 初期盤面を作成
    board = load_initial_board()
    player = Player.BLACK
    my_hand = get_initial_hand_pieces(player)
    opponent_hand = get_initial_hand_pieces(player.opponent)
    
    print(f"Initial legal moves: {len(Rules.get_legal_moves(board, player, my_hand))}")
    
    # 探索を実行
    best_action, action_probs = mcts.search(
        board, player, my_hand, opponent_hand,
        temperature=1.0
    )
    
    print(f"Best action index: {best_action}")
    print(f"Action probs sum: {action_probs.sum():.4f}")
    print(f"Non-zero probs: {np.sum(action_probs > 0)}")
    
    # Moveに変換
    best_move = mcts.action_encoder.decode_action(best_action, player, board)
    print(f"Best move: {best_move.move_type}, {best_move.from_pos} -> {best_move.to_pos}")
