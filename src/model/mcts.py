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
    
    def copy(self) -> 'GameState':
        """ディープコピーを作成"""
        return GameState(
            board=self.board.copy(),
            player=self.player,
            my_hand=copy.deepcopy(self.my_hand),
            opponent_hand=copy.deepcopy(self.opponent_hand)
        )
    
    def switch_player(self):
        """手番を交代（持ち駒も入れ替え）"""
        self.player = self.player.opponent
        self.my_hand, self.opponent_hand = self.opponent_hand, self.my_hand


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
    """モンテカルロ木探索"""
    
    def __init__(
        self,
        network,
        state_encoder: StateEncoder = None,
        action_encoder: ActionEncoder = None,
        c_puct: float = 1.5,
        num_simulations: int = 50,
        device: str = 'cuda'
    ):
        self.network = network
        self.state_encoder = state_encoder or StateEncoder()
        self.action_encoder = action_encoder or ActionEncoder()
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.device = device
    
    def search(
        self,
        board: Board,
        player: Player,
        my_hand: Dict[PieceType, int],
        opponent_hand: Dict[PieceType, int],
        temperature: float = 1.0
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
        
        Returns:
            best_action: 選択された行動インデックス
            action_probs: (7695,) 各行動の選択確率（学習データ用）
        """
        # ルートノードを作成
        root_state = GameState(
            board=board.copy(),
            player=player,
            my_hand=copy.deepcopy(my_hand),
            opponent_hand=copy.deepcopy(opponent_hand)
        )
        root = MCTSNode(state=root_state)
        
        # ルートノードを展開
        value = self._evaluate_and_expand(root)
        
        # シミュレーションを実行
        for _ in range(self.num_simulations):
            node = root
            
            # 1. Selection: 葉ノードまで降下
            while node.is_expanded():
                child = node.select_child(self.c_puct)
                if child is None:
                    break
                node = child
            
            # 2. ゲーム終了チェック
            is_over, winner = Rules.is_game_over(node.state.board)
            
            if is_over:
                # 終了局面の価値（現在のノードのプレイヤー視点）
                if winner is None:
                    value = 0.0
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
    
    def _evaluate_and_expand(self, node: MCTSNode) -> float:
        """
        ノードをニューラルネットワークで評価し、展開する
        
        Returns:
            value: 評価値（現在のプレイヤー視点）
        """
        state = node.state
        
        # 合法手を取得
        legal_moves = Rules.get_legal_moves(
            state.board, 
            state.player, 
            state.my_hand
        )
        
        if not legal_moves:
            # 合法手がない = 負け
            return -1.0
        
        # 状態をエンコード
        state_tensor = self.state_encoder.encode(
            state.board,
            state.player,
            state.my_hand,
            state.opponent_hand
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
