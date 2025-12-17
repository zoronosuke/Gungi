"""
ウェブAPI用 AI推論モジュール
学習済みモデルを使ってゲームプレイを行う
"""

import os
import torch
import numpy as np
from typing import Optional, Tuple, List, Dict
from pathlib import Path

from ..engine.board import Board
from ..engine.piece import Player, PieceType, Piece
from ..engine.move import Move, MoveType
from ..engine.rules import Rules
from ..model.network import GungiNetwork, create_model
from ..model.encoder import StateEncoder, ActionEncoder


class GungiAI:
    """
    軍儀AI - ニューラルネットワーク + MCTS
    
    難易度レベル:
    - easy: MCTS 10回、ランダム要素多め
    - medium: MCTS 30回
    - hard: MCTS 100回
    - expert: MCTS 200回
    """
    
    DIFFICULTY_SETTINGS = {
        'easy': {'mcts_sims': 10, 'temperature': 1.5},
        'medium': {'mcts_sims': 30, 'temperature': 1.0},
        'hard': {'mcts_sims': 100, 'temperature': 0.5},
        'expert': {'mcts_sims': 200, 'temperature': 0.1},
    }
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = 'auto',
        test_mode: bool = True
    ):
        """
        Args:
            checkpoint_path: チェックポイントファイルのパス（Noneなら最新を自動検索）
            device: 'cuda', 'cpu', または 'auto'
            test_mode: テストモードのモデルを使うか
        """
        # デバイス設定
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"AI Device: {self.device}")
        
        # エンコーダー
        self.state_encoder = StateEncoder()
        self.action_encoder = ActionEncoder()
        
        # モデル作成
        self.network = create_model(self.device, test_mode=test_mode)
        self.network.eval()
        
        # チェックポイント読み込み
        self.model_loaded = False
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        else:
            self._load_latest_checkpoint()
    
    def _load_latest_checkpoint(self):
        """最新のチェックポイントを自動で読み込む"""
        checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints'
        
        if not checkpoint_dir.exists():
            print("Warning: No checkpoint directory found. Using random weights.")
            return
        
        # latest.ptを探す
        latest_path = checkpoint_dir / 'latest.pt'
        if latest_path.exists():
            self.load_checkpoint(str(latest_path))
            return
        
        # なければmodel_iter_*.ptから最新を探す
        checkpoints = list(checkpoint_dir.glob('model_iter_*.pt'))
        if checkpoints:
            # ファイル名からイテレーション番号を抽出してソート
            checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
            self.load_checkpoint(str(checkpoints[-1]))
            return
        
        print("Warning: No checkpoint files found. Using random weights.")
    
    def load_checkpoint(self, path: str):
        """チェックポイントを読み込む"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # モデルの重みを読み込み
            if 'model_state_dict' in checkpoint:
                self.network.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.network.load_state_dict(checkpoint)
            
            self.network.eval()
            self.model_loaded = True
            
            # 学習情報を表示
            if 'iteration' in checkpoint:
                print(f"Loaded checkpoint: iteration {checkpoint['iteration']}")
            if 'total_games' in checkpoint:
                print(f"Total games trained: {checkpoint['total_games']}")
            
            print(f"Model loaded from: {path}")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Using random weights.")
    
    def get_best_move(
        self,
        board: Board,
        current_player: Player,
        hand_pieces: Dict[PieceType, int],
        opponent_hand: Dict[PieceType, int],
        difficulty: str = 'medium'
    ) -> Tuple[Move, float]:
        """
        最善手を取得
        
        Args:
            board: 現在の盤面
            current_player: 現在のプレイヤー
            hand_pieces: 現在のプレイヤーの持ち駒
            opponent_hand: 相手の持ち駒
            difficulty: 難易度 ('easy', 'medium', 'hard', 'expert')
        
        Returns:
            (最善手, 評価値)
        """
        settings = self.DIFFICULTY_SETTINGS.get(difficulty, self.DIFFICULTY_SETTINGS['medium'])
        mcts_sims = settings['mcts_sims']
        temperature = settings['temperature']
        
        # 合法手を取得
        legal_moves = Rules.get_legal_moves(board, current_player, hand_pieces)
        
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # 合法手が1つなら即座に返す
        if len(legal_moves) == 1:
            return legal_moves[0], 0.0
        
        # 状態をエンコード
        state = self.state_encoder.encode(board, current_player, hand_pieces, opponent_hand)
        
        # ニューラルネットワークで評価
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            log_policy, value = self.network(state_tensor)
            policy = torch.exp(log_policy).cpu().numpy()[0]
            value = value.cpu().numpy()[0, 0]
        
        # MCTSを実行
        move_probs = self._run_mcts(
            board, current_player, hand_pieces, opponent_hand,
            policy, mcts_sims
        )
        
        # 温度を適用して手を選択
        if temperature < 0.1:
            # 最善手を選択
            best_idx = np.argmax(move_probs)
            best_move = legal_moves[best_idx]
        else:
            # 確率的に選択
            probs = move_probs ** (1.0 / temperature)
            probs = probs / probs.sum()
            
            # 安全のためクリップ
            probs = np.clip(probs, 1e-10, 1.0)
            probs = probs / probs.sum()
            
            try:
                selected_idx = np.random.choice(len(legal_moves), p=probs)
                best_move = legal_moves[selected_idx]
            except:
                # フォールバック
                best_idx = np.argmax(move_probs)
                best_move = legal_moves[best_idx]
        
        return best_move, float(value)
    
    def _run_mcts(
        self,
        board: Board,
        current_player: Player,
        hand_pieces: Dict[PieceType, int],
        opponent_hand: Dict[PieceType, int],
        prior_policy: np.ndarray,
        num_simulations: int
    ) -> np.ndarray:
        """
        簡易MCTS（ウェブAPI用に軽量化）
        
        Returns:
            各合法手の選択確率
        """
        legal_moves = Rules.get_legal_moves(board, current_player, hand_pieces)
        num_moves = len(legal_moves)
        
        if num_moves == 0:
            return np.array([])
        
        # 合法手のpriorを取得
        move_priors = np.zeros(num_moves)
        for i, move in enumerate(legal_moves):
            action_idx = self.action_encoder.encode_move(move)
            if action_idx is not None and action_idx < len(prior_policy):
                move_priors[i] = prior_policy[action_idx]
            else:
                move_priors[i] = 1e-6
        
        # 正規化
        if move_priors.sum() > 0:
            move_priors = move_priors / move_priors.sum()
        else:
            move_priors = np.ones(num_moves) / num_moves
        
        # MCTS統計
        visit_counts = np.zeros(num_moves)
        total_values = np.zeros(num_moves)
        c_puct = 1.5
        
        for _ in range(num_simulations):
            # PUCT公式でアクションを選択
            sqrt_total = np.sqrt(visit_counts.sum() + 1)
            q_values = total_values / (visit_counts + 1e-8)
            u_values = c_puct * move_priors * sqrt_total / (1 + visit_counts)
            scores = q_values + u_values
            
            action_idx = np.argmax(scores)
            move = legal_moves[action_idx]
            
            # シミュレーション（1手先を評価）
            sim_board = board.copy()
            sim_hand = hand_pieces.copy()
            
            success, _ = Rules.apply_move(sim_board, move, sim_hand)
            
            if success:
                # 相手視点で評価
                sim_state = self.state_encoder.encode(
                    sim_board, current_player.opponent, opponent_hand, sim_hand
                )
                
                with torch.no_grad():
                    state_tensor = torch.from_numpy(sim_state).float().unsqueeze(0).to(self.device)
                    _, sim_value = self.network(state_tensor)
                    value = -sim_value.cpu().numpy()[0, 0]  # 自分視点に反転
            else:
                value = -1.0  # 無効な手はペナルティ
            
            # 統計を更新
            visit_counts[action_idx] += 1
            total_values[action_idx] += value
        
        # 訪問回数に基づいて確率を計算
        if visit_counts.sum() > 0:
            return visit_counts / visit_counts.sum()
        else:
            return move_priors
    
    def evaluate_position(
        self,
        board: Board,
        current_player: Player,
        hand_pieces: Dict[PieceType, int],
        opponent_hand: Dict[PieceType, int]
    ) -> float:
        """
        局面を評価（-1.0 ~ 1.0）
        
        Returns:
            正の値: 現在のプレイヤー有利
            負の値: 相手有利
        """
        state = self.state_encoder.encode(board, current_player, hand_pieces, opponent_hand)
        
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            _, value = self.network(state_tensor)
            return float(value.cpu().numpy()[0, 0])


# シングルトンインスタンス（サーバー起動時に1回だけ初期化）
_ai_instance: Optional[GungiAI] = None


def get_ai() -> GungiAI:
    """AIインスタンスを取得（遅延初期化）"""
    global _ai_instance
    if _ai_instance is None:
        _ai_instance = GungiAI()
    return _ai_instance


def reload_ai(checkpoint_path: Optional[str] = None):
    """AIモデルを再読み込み"""
    global _ai_instance
    _ai_instance = GungiAI(checkpoint_path=checkpoint_path)
    return _ai_instance
