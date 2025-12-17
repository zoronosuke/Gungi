"""
自己対戦によるデータ生成
AlphaZero方式の学習データを生成する
"""

import copy
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

from ..engine.board import Board
from ..engine.piece import Player, PieceType
from ..engine.move import Move
from ..engine.rules import Rules
from ..engine.initial_setup import load_initial_board, get_initial_hand_pieces
from .encoder import StateEncoder, ActionEncoder
from .mcts import MCTS, GameState


@dataclass
class TrainingExample:
    """1つの学習サンプル"""
    state: np.ndarray      # (91, 9, 9) エンコードされた盤面
    policy: np.ndarray     # (7695,) MCTSが出した方策
    value: float           # 最終結果（勝ち=1, 負け=-1, 引き分け=0）


class SelfPlay:
    """自己対戦によるデータ生成"""
    
    # 最大手数（千日手対策）
    MAX_MOVES = 300
    
    def __init__(
        self,
        network,
        state_encoder: StateEncoder = None,
        action_encoder: ActionEncoder = None,
        mcts_simulations: int = 50,
        c_puct: float = 1.5,
        device: str = 'cuda'
    ):
        self.network = network
        self.state_encoder = state_encoder or StateEncoder()
        self.action_encoder = action_encoder or ActionEncoder()
        self.device = device
        
        # MCTSを作成
        self.mcts = MCTS(
            network=network,
            state_encoder=self.state_encoder,
            action_encoder=self.action_encoder,
            c_puct=c_puct,
            num_simulations=mcts_simulations,
            device=device
        )
    
    def play_game(
        self, 
        temperature_threshold: int = 20,
        verbose: bool = False
    ) -> Tuple[List[TrainingExample], Optional[Player]]:
        """
        1ゲームをプレイしてデータを生成
        
        Args:
            temperature_threshold: この手数まではtemperature=1.0、
                                  それ以降はtemperature=0.1
            verbose: 詳細出力するかどうか
        
        Returns:
            (学習データのリスト, 勝者(引き分けはNone))
        """
        # 初期盤面を設定
        board = load_initial_board()
        current_player = Player.BLACK
        
        # 各プレイヤーの持ち駒
        hands = {
            Player.BLACK: get_initial_hand_pieces(Player.BLACK),
            Player.WHITE: get_initial_hand_pieces(Player.WHITE)
        }
        
        # ゲームデータを記録
        game_history = []  # [(state, policy, player), ...]
        
        move_count = 0
        winner = None
        
        while move_count < self.MAX_MOVES:
            # ゲーム終了チェック
            is_over, game_winner = Rules.is_game_over(board)
            if is_over:
                winner = game_winner
                break
            
            # 現在の状態をエンコード
            my_hand = hands[current_player]
            opponent_hand = hands[current_player.opponent]
            
            state = self.state_encoder.encode(
                board, current_player, my_hand, opponent_hand
            )
            
            # 温度を決定
            temperature = 1.0 if move_count < temperature_threshold else 0.1
            
            # MCTSで探索
            action_idx, action_probs = self.mcts.search(
                board, current_player, my_hand, opponent_hand,
                temperature=temperature
            )
            
            # データを記録
            game_history.append((state.copy(), action_probs.copy(), current_player))
            
            # 手を適用
            move = self.action_encoder.decode_action(action_idx, current_player, board)
            success, captured = Rules.apply_move(board, move, my_hand)
            
            if not success:
                # 手が適用できない場合（バグ）
                if verbose:
                    print(f"Warning: Move failed at turn {move_count}")
                break
            
            if verbose and move_count % 20 == 0:
                print(f"Turn {move_count}: {current_player.name} - {move.move_type.name}")
            
            # 手番を交代
            current_player = current_player.opponent
            move_count += 1
        
        # 最大手数に達した場合は引き分け
        if move_count >= self.MAX_MOVES:
            winner = None
        
        if verbose:
            if winner:
                print(f"Game ended: {winner.name} wins after {move_count} moves")
            else:
                print(f"Game ended: Draw after {move_count} moves")
        
        # 学習データを作成（結果を付与）
        examples = []
        for state, policy, player in game_history:
            # 勝者に基づいて価値を決定
            if winner is None:
                value = 0.0
            elif winner == player:
                value = 1.0
            else:
                value = -1.0
            
            examples.append(TrainingExample(
                state=state,
                policy=policy,
                value=value
            ))
        
        return examples, winner
    
    def generate_data(
        self, 
        num_games: int,
        temperature_threshold: int = 20,
        verbose: bool = True
    ) -> List[TrainingExample]:
        """
        複数ゲームのデータを生成
        
        Args:
            num_games: 生成するゲーム数
            temperature_threshold: 温度を下げる閾値
            verbose: プログレスバーを表示するか
        
        Returns:
            全ゲームの学習データ
        """
        all_examples = []
        wins = {Player.BLACK: 0, Player.WHITE: 0, None: 0}
        
        iterator = tqdm(range(num_games), desc="Self-play") if verbose else range(num_games)
        
        for game_idx in iterator:
            examples, winner = self.play_game(
                temperature_threshold=temperature_threshold,
                verbose=False
            )
            all_examples.extend(examples)
            wins[winner] += 1
            
            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix({
                    'B': wins[Player.BLACK],
                    'W': wins[Player.WHITE],
                    'D': wins[None],
                    'examples': len(all_examples)
                })
        
        if verbose:
            print(f"\nGenerated {len(all_examples)} examples from {num_games} games")
            print(f"Results: BLACK={wins[Player.BLACK]}, WHITE={wins[Player.WHITE]}, DRAW={wins[None]}")
        
        return all_examples


class ReplayBuffer:
    """学習データを保持するリプレイバッファ"""
    
    def __init__(self, max_size: int = 50000):
        self.max_size = max_size
        self.buffer: List[TrainingExample] = []
    
    def add(self, examples: List[TrainingExample]):
        """データを追加"""
        self.buffer.extend(examples)
        
        # 最大サイズを超えたら古いデータを削除
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]
    
    def sample(self, batch_size: int) -> List[TrainingExample]:
        """ランダムにサンプリング"""
        if len(self.buffer) < batch_size:
            return self.buffer.copy()
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def save(self, path: str):
        """バッファをファイルに保存"""
        data = {
            'states': np.array([ex.state for ex in self.buffer]),
            'policies': np.array([ex.policy for ex in self.buffer]),
            'values': np.array([ex.value for ex in self.buffer])
        }
        np.savez_compressed(path, **data)
    
    def load(self, path: str):
        """バッファをファイルから読み込み"""
        data = np.load(path)
        states = data['states']
        policies = data['policies']
        values = data['values']
        
        self.buffer = [
            TrainingExample(state=s, policy=p, value=v)
            for s, p, v in zip(states, policies, values)
        ]


if __name__ == "__main__":
    # テスト
    from .network import create_model
    
    print("=== SelfPlay Test ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # モデルを作成
    model = create_model(device, test_mode=True)
    
    # SelfPlayを作成
    self_play = SelfPlay(
        network=model,
        mcts_simulations=10,  # テスト用に少なく
        device=device
    )
    
    # 1ゲームをプレイ
    print("\nPlaying 1 game...")
    examples, winner = self_play.play_game(
        temperature_threshold=10,
        verbose=True
    )
    
    print(f"\nGenerated {len(examples)} training examples")
    if examples:
        print(f"First example state shape: {examples[0].state.shape}")
        print(f"First example policy shape: {examples[0].policy.shape}")
        print(f"First example value: {examples[0].value}")
