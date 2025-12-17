"""
GPU活用版の並列MCTS
バッチ推論でGPUを効率的に使用する
"""

import copy
import numpy as np
import torch
import torch.multiprocessing as mp
from queue import Empty
from threading import Thread
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from ..engine.board import Board
from ..engine.piece import Player, PieceType
from ..engine.move import Move, MoveType
from ..engine.rules import Rules
from ..engine.initial_setup import load_initial_board, get_initial_hand_pieces
from .encoder import StateEncoder, ActionEncoder


@dataclass
class InferenceRequest:
    """推論リクエスト"""
    worker_id: int
    request_id: int
    state: np.ndarray  # (91, 9, 9)


@dataclass
class InferenceResult:
    """推論結果"""
    worker_id: int
    request_id: int
    policy: np.ndarray  # (7695,)
    value: float


class BatchInferenceServer:
    """
    GPUでバッチ推論を行うサーバー
    複数ワーカーからのリクエストをまとめて処理
    """
    
    def __init__(
        self,
        network,
        device: str = 'cuda',
        batch_size: int = 32,
        timeout: float = 0.01
    ):
        self.network = network
        self.device = device
        self.batch_size = batch_size
        self.timeout = timeout
        
        self.network.to(device)
        self.network.eval()
        
        # マルチプロセス用のキュー
        self.request_queue = mp.Queue()
        self.result_queues: Dict[int, mp.Queue] = {}
        
        self.running = False
        self.server_thread = None
    
    def start(self, num_workers: int):
        """サーバーを開始"""
        # 各ワーカー用の結果キューを作成
        for i in range(num_workers):
            self.result_queues[i] = mp.Queue()
        
        self.running = True
        self.server_thread = Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()
    
    def stop(self):
        """サーバーを停止"""
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=2.0)
    
    def _server_loop(self):
        """推論サーバーのメインループ"""
        while self.running:
            requests = []
            
            # リクエストを収集（バッチサイズまで、またはタイムアウトまで）
            try:
                # 最初の1つは待つ
                req = self.request_queue.get(timeout=0.1)
                requests.append(req)
                
                # 追加のリクエストを非ブロッキングで収集
                while len(requests) < self.batch_size:
                    try:
                        req = self.request_queue.get_nowait()
                        requests.append(req)
                    except Empty:
                        break
            except Empty:
                continue
            
            if not requests:
                continue
            
            # バッチ推論を実行
            try:
                states = np.stack([r.state for r in requests])
                states_tensor = torch.from_numpy(states).float().to(self.device)
                
                with torch.no_grad():
                    log_policies, values = self.network(states_tensor)
                    policies = torch.exp(log_policies).cpu().numpy()
                    values = values.cpu().numpy().flatten()
                
                # 結果を各ワーカーに返す
                for i, req in enumerate(requests):
                    result = InferenceResult(
                        worker_id=req.worker_id,
                        request_id=req.request_id,
                        policy=policies[i],
                        value=values[i]
                    )
                    self.result_queues[req.worker_id].put(result)
            
            except Exception as e:
                print(f"Inference error: {e}")
                # エラー時はダミー結果を返す
                for req in requests:
                    result = InferenceResult(
                        worker_id=req.worker_id,
                        request_id=req.request_id,
                        policy=np.ones(7695) / 7695,
                        value=0.0
                    )
                    self.result_queues[req.worker_id].put(result)
    
    def get_request_queue(self):
        return self.request_queue
    
    def get_result_queue(self, worker_id: int):
        return self.result_queues[worker_id]


class ParallelMCTSWorker:
    """
    並列MCTS用のワーカー
    推論はサーバーにリクエストして結果を待つ
    """
    
    def __init__(
        self,
        worker_id: int,
        request_queue: mp.Queue,
        result_queue: mp.Queue,
        state_encoder: StateEncoder,
        action_encoder: ActionEncoder,
        c_puct: float = 1.5,
        num_simulations: int = 50
    ):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.request_counter = 0
    
    def _request_inference(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """推論をリクエストして結果を待つ"""
        request_id = self.request_counter
        self.request_counter += 1
        
        request = InferenceRequest(
            worker_id=self.worker_id,
            request_id=request_id,
            state=state
        )
        self.request_queue.put(request)
        
        # 結果を待つ
        while True:
            result = self.result_queue.get()
            if result.request_id == request_id:
                return result.policy, result.value
    
    def search(
        self,
        board: Board,
        current_player: Player,
        my_hand: Dict[PieceType, int],
        opponent_hand: Dict[PieceType, int],
        temperature: float = 1.0
    ) -> Tuple[int, np.ndarray]:
        """MCTSで最善手を探索"""
        
        # ルートノードを作成
        root = MCTSNode(
            board=copy.deepcopy(board),
            current_player=current_player,
            my_hand=copy.deepcopy(my_hand),
            opponent_hand=copy.deepcopy(opponent_hand),
            parent=None,
            action_idx=None
        )
        
        # ルートを展開
        self._expand_node(root)
        
        # シミュレーションを実行
        for _ in range(self.num_simulations):
            node = root
            
            # 選択: リーフノードまで降りる
            while node.is_expanded and not node.is_terminal:
                node = self._select_child(node)
            
            # 展開と評価
            if not node.is_terminal:
                self._expand_node(node)
            
            value = node.value
            
            # バックアップ
            self._backup(node, value)
        
        # 行動確率を計算
        visit_counts = np.zeros(7695)
        for action_idx, child in root.children.items():
            visit_counts[action_idx] = child.visit_count
        
        if temperature == 0:
            # 最も訪問回数が多い手を選択
            action_idx = np.argmax(visit_counts)
            probs = np.zeros(7695)
            probs[action_idx] = 1.0
        else:
            # 温度に基づいて確率的に選択
            visit_counts = visit_counts ** (1.0 / temperature)
            total = visit_counts.sum()
            if total > 0:
                probs = visit_counts / total
            else:
                probs = np.ones(7695) / 7695
            action_idx = np.random.choice(7695, p=probs)
        
        return action_idx, probs
    
    def _expand_node(self, node: 'MCTSNode'):
        """ノードを展開"""
        # 状態をエンコード
        state = self.state_encoder.encode(
            node.board, node.current_player, node.my_hand, node.opponent_hand
        )
        
        # 推論をリクエスト
        policy, value = self._request_inference(state)
        node.value = value
        
        # 合法手を取得
        legal_actions = self._get_legal_actions(node)
        
        if not legal_actions:
            node.is_terminal = True
            # 合法手がない = 負け
            node.value = -1.0
            return
        
        # 合法手でマスク
        legal_mask = np.zeros(7695)
        for action_idx in legal_actions:
            legal_mask[action_idx] = 1.0
        
        masked_policy = policy * legal_mask
        total = masked_policy.sum()
        if total > 0:
            masked_policy = masked_policy / total
        else:
            masked_policy = legal_mask / legal_mask.sum()
        
        node.prior_probs = masked_policy
        node.legal_actions = legal_actions
        node.is_expanded = True
    
    def _get_legal_actions(self, node: 'MCTSNode') -> List[int]:
        """合法手のアクションインデックスを取得"""
        legal_moves = Rules.get_legal_moves(
            node.board, node.current_player, node.my_hand
        )
        
        action_indices = []
        for move in legal_moves:
            action_idx = self.action_encoder.encode_move(move)
            if action_idx is not None:
                action_indices.append(action_idx)
        
        return action_indices
    
    def _select_child(self, node: 'MCTSNode') -> 'MCTSNode':
        """PUCTで子ノードを選択"""
        best_score = -float('inf')
        best_action = None
        
        sqrt_total = np.sqrt(node.visit_count + 1)
        
        for action_idx in node.legal_actions:
            if action_idx in node.children:
                child = node.children[action_idx]
                q_value = child.total_value / (child.visit_count + 1e-8)
                # 子ノードの価値は相手視点なので反転
                q_value = -q_value
            else:
                q_value = 0.0
            
            prior = node.prior_probs[action_idx]
            
            if action_idx in node.children:
                visit = node.children[action_idx].visit_count
            else:
                visit = 0
            
            u_value = self.c_puct * prior * sqrt_total / (1 + visit)
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = action_idx
        
        # 子ノードがなければ作成
        if best_action not in node.children:
            node.children[best_action] = self._create_child(node, best_action)
        
        return node.children[best_action]
    
    def _create_child(self, parent: 'MCTSNode', action_idx: int) -> 'MCTSNode':
        """子ノードを作成"""
        new_board = copy.deepcopy(parent.board)
        new_my_hand = copy.deepcopy(parent.my_hand)
        new_opponent_hand = copy.deepcopy(parent.opponent_hand)
        
        move = self.action_encoder.decode_action(
            action_idx, parent.current_player, parent.board
        )
        
        success, _ = Rules.apply_move(new_board, move, new_my_hand)
        
        if not success:
            # 無効な手の場合
            child = MCTSNode(
                board=new_board,
                current_player=parent.current_player.opponent,
                my_hand=new_opponent_hand,
                opponent_hand=new_my_hand,
                parent=parent,
                action_idx=action_idx
            )
            child.is_terminal = True
            child.value = 1.0  # 親から見て勝ち
            return child
        
        # ゲーム終了チェック
        is_over, winner = Rules.is_game_over(new_board)
        
        child = MCTSNode(
            board=new_board,
            current_player=parent.current_player.opponent,
            my_hand=new_opponent_hand,
            opponent_hand=new_my_hand,
            parent=parent,
            action_idx=action_idx
        )
        
        if is_over:
            child.is_terminal = True
            if winner == parent.current_player:
                child.value = 1.0
            elif winner == parent.current_player.opponent:
                child.value = -1.0
            else:
                child.value = 0.0
        
        return child
    
    def _backup(self, node: 'MCTSNode', value: float):
        """結果をバックアップ"""
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_value += value
            value = -value  # 相手視点に反転
            current = current.parent


class MCTSNode:
    """MCTSのノード"""
    
    def __init__(
        self,
        board: Board,
        current_player: Player,
        my_hand: Dict[PieceType, int],
        opponent_hand: Dict[PieceType, int],
        parent: Optional['MCTSNode'] = None,
        action_idx: Optional[int] = None
    ):
        self.board = board
        self.current_player = current_player
        self.my_hand = my_hand
        self.opponent_hand = opponent_hand
        self.parent = parent
        self.action_idx = action_idx
        
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.value = 0.0
        
        self.prior_probs = None
        self.legal_actions = []
        self.is_expanded = False
        self.is_terminal = False
