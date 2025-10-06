"""
モンテカルロ木探索（MCTS）の実装
AlphaZero型の探索アルゴリズム
"""

import math
import numpy as np
from typing import Optional, List, Dict, Tuple
from ..engine import Board, Player, Move, Rules


class MCTSNode:
    """MCTSの探索木のノード"""
    
    def __init__(
        self,
        board: Board,
        player: Player,
        parent: Optional['MCTSNode'] = None,
        move: Optional[Move] = None,
        prior_prob: float = 0.0
    ):
        self.board = board
        self.player = player
        self.parent = parent
        self.move = move  # このノードに至った手
        self.prior_prob = prior_prob  # ニューラルネットが予測した事前確率
        
        self.children: Dict[str, 'MCTSNode'] = {}  # 子ノード
        self.visit_count = 0  # 訪問回数
        self.total_value = 0.0  # 累積価値
        self.mean_value = 0.0  # 平均価値
    
    def is_leaf(self) -> bool:
        """葉ノードかどうか"""
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """ルートノードかどうか"""
        return self.parent is None
    
    def expand(self, action_probs: Dict[Move, float]):
        """
        ノードを展開（子ノードを作成）
        
        Args:
            action_probs: 各手の事前確率
        """
        for move, prob in action_probs.items():
            if str(move) not in self.children:
                # 手を適用した新しい盤面を作成
                new_board = self.board.copy()
                Rules.apply_move(new_board, move)
                
                # 子ノードを作成
                child = MCTSNode(
                    board=new_board,
                    player=self.player.opponent,
                    parent=self,
                    move=move,
                    prior_prob=prob
                )
                self.children[str(move)] = child
    
    def select_child(self, c_puct: float = 1.0) -> 'MCTSNode':
        """
        UCB1アルゴリズムで最良の子ノードを選択
        
        Args:
            c_puct: 探索と活用のバランスパラメータ
        
        Returns:
            選択された子ノード
        """
        best_score = -float('inf')
        best_child = None
        
        for child in self.children.values():
            # UCB1スコアを計算
            if child.visit_count == 0:
                ucb_score = float('inf')
            else:
                # Q値（平均価値）
                q_value = child.mean_value
                
                # U値（探索ボーナス）
                u_value = c_puct * child.prior_prob * math.sqrt(self.visit_count) / (1 + child.visit_count)
                
                ucb_score = q_value + u_value
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        
        return best_child
    
    def update(self, value: float):
        """
        ノードの統計情報を更新
        
        Args:
            value: バックアップする価値
        """
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count
    
    def backpropagate(self, value: float):
        """
        価値をルートまでバックプロパゲート
        
        Args:
            value: リーフノードでの評価値
        """
        node = self
        while node is not None:
            node.update(value)
            value = -value  # 相手視点では符号反転
            node = node.parent


class MCTS:
    """モンテカルロ木探索"""
    
    def __init__(
        self,
        network,
        c_puct: float = 1.0,
        num_simulations: int = 100,
        device: str = 'cpu'
    ):
        self.network = network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.device = device
    
    def search(self, board: Board, player: Player) -> Tuple[Move, Dict[Move, float]]:
        """
        MCTSで最良の手を探索
        
        Args:
            board: 現在の盤面
            player: 現在のプレイヤー
        
        Returns:
            (最良の手, 各手の訪問確率)
        """
        root = MCTSNode(board=board, player=player)
        
        # シミュレーションを実行
        for _ in range(self.num_simulations):
            node = root
            
            # 1. Selection: 葉ノードまで降下
            while not node.is_leaf():
                node = node.select_child(self.c_puct)
            
            # 2. Evaluation: ニューラルネットで評価
            value, action_probs = self._evaluate(node.board, node.player)
            
            # ゲーム終了チェック
            is_over, winner = Rules.is_game_over(node.board)
            if is_over:
                # 終了局面の価値
                if winner == player:
                    value = 1.0
                elif winner == player.opponent:
                    value = -1.0
                else:
                    value = 0.0
            else:
                # 3. Expansion: ノードを展開
                node.expand(action_probs)
            
            # 4. Backpropagation: 価値を逆伝播
            node.backpropagate(value)
        
        # 最も訪問回数の多い手を選択
        best_move = None
        best_visits = -1
        visit_counts = {}
        
        for move_str, child in root.children.items():
            visit_counts[child.move] = child.visit_count
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_move = child.move
        
        # 訪問確率を計算
        total_visits = sum(visit_counts.values())
        visit_probs = {
            move: count / total_visits
            for move, count in visit_counts.items()
        } if total_visits > 0 else {}
        
        return best_move, visit_probs
    
    def _evaluate(self, board: Board, player: Player) -> Tuple[float, Dict[Move, float]]:
        """
        ニューラルネットで盤面を評価
        
        Args:
            board: 評価する盤面
            player: 現在のプレイヤー
        
        Returns:
            (価値, 各手の事前確率)
        """
        import torch
        from .network import encode_board_state
        
        # 盤面をエンコード
        state = encode_board_state(board, player).to(self.device)
        
        # ニューラルネットで予測
        with torch.no_grad():
            policy_logits, value = self.network(state)
        
        value = value.item()
        policy_probs = torch.exp(policy_logits).cpu().numpy()[0]
        
        # 合法手を取得
        legal_moves = Rules.get_legal_moves(board, player)
        
        # 合法手のみに確率を割り当て
        action_probs = {}
        total_prob = 0.0
        
        for move in legal_moves:
            # 手をアクションインデックスに変換（簡易版）
            if move.from_pos and move.to_pos:
                from_row, from_col = move.from_pos
                to_row, to_col = move.to_pos
                action_idx = (from_row * 9 + from_col) * 81 + to_row * 9 + to_col
                
                if action_idx < len(policy_probs):
                    prob = policy_probs[action_idx]
                else:
                    prob = 1.0 / len(legal_moves)
            else:
                prob = 1.0 / len(legal_moves)
            
            action_probs[move] = prob
            total_prob += prob
        
        # 正規化
        if total_prob > 0:
            action_probs = {
                move: prob / total_prob
                for move, prob in action_probs.items()
            }
        
        return value, action_probs


if __name__ == "__main__":
    # テスト
    from ..engine.initial_setup import load_initial_board
    from .network import create_model
    
    board = load_initial_board()
    model = create_model('cpu')
    
    mcts = MCTS(model, num_simulations=10)
    best_move, visit_probs = mcts.search(board, Player.BLACK)
    
    print(f"Best move: {best_move}")
    print(f"Visit probs: {len(visit_probs)}")
