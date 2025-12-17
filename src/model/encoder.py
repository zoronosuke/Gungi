"""
状態・行動のエンコーダー
盤面状態をニューラルネットワークの入力形式に変換し、
行動をインデックスに相互変換する
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

from ..engine.piece import PieceType, Player
from ..engine.board import Board, BOARD_SIZE
from ..engine.move import Move, MoveType


# 駒の種類一覧（ソースコードの定義順）
PIECE_TYPES = [
    PieceType.SUI,      # 帥 (0)
    PieceType.DAI,      # 大 (1)
    PieceType.CHUU,     # 中 (2)
    PieceType.SHO,      # 小 (3)
    PieceType.SAMURAI,  # 侍 (4)
    PieceType.HYO,      # 兵 (5)
    PieceType.UMA,      # 馬 (6)
    PieceType.SHINOBI,  # 忍 (7)
    PieceType.YARI,     # 槍 (8)
    PieceType.TORIDE,   # 砦 (9)
    PieceType.YUMI,     # 弓 (10)
    PieceType.TSUTU,    # 筒 (11)
    PieceType.HOU,      # 砲 (12)
    PieceType.BOU,      # 謀 (13)
]

# 駒種類からインデックスへのマッピング
PIECE_TYPE_TO_INDEX = {pt: i for i, pt in enumerate(PIECE_TYPES)}
NUM_PIECE_TYPES = len(PIECE_TYPES)  # 14

# 持ち駒のカテゴリ分け（3カテゴリ）
# カテゴリ1: 大駒（大、中、小）
HAND_CATEGORY_LARGE = [PieceType.DAI, PieceType.CHUU, PieceType.SHO]
# カテゴリ2: 中駒（侍、馬、忍、槍、弓）
HAND_CATEGORY_MEDIUM = [PieceType.SAMURAI, PieceType.UMA, PieceType.SHINOBI, 
                         PieceType.YARI, PieceType.YUMI]
# カテゴリ3: 小駒（兵、砦、筒、砲、謀）
HAND_CATEGORY_SMALL = [PieceType.HYO, PieceType.TORIDE, PieceType.TSUTU, 
                        PieceType.HOU, PieceType.BOU]


class StateEncoder:
    """
    盤面状態をニューラルネットワークの入力テンソルに変換するクラス
    
    入力テンソルの形状: (91, 9, 9)
    
    チャンネル構成:
    - ch 0-41:   自分の駒（14種類 × 3段）
    - ch 42-83:  相手の駒（14種類 × 3段）
    - ch 84-86:  自分の持ち駒（3カテゴリ）
    - ch 87-89:  相手の持ち駒（3カテゴリ）
    - ch 90:     手番（自分の番なら全て1）
    """
    
    CHANNELS = 91
    BOARD_SIZE = BOARD_SIZE  # 9
    MAX_STACK = 3
    
    def __init__(self):
        pass
    
    def encode(
        self,
        board: Board,
        current_player: Player,
        my_hand: Dict[PieceType, int],
        opponent_hand: Dict[PieceType, int]
    ) -> np.ndarray:
        """
        盤面状態をニューラルネットワークの入力に変換
        
        Args:
            board: Boardオブジェクト
            current_player: 現在の手番のプレイヤー
            my_hand: 自分の持ち駒 {PieceType: count}
            opponent_hand: 相手の持ち駒 {PieceType: count}
        
        Returns:
            np.ndarray: shape (91, 9, 9)
        """
        state = np.zeros((self.CHANNELS, self.BOARD_SIZE, self.BOARD_SIZE), 
                         dtype=np.float32)
        
        opponent_player = current_player.opponent
        
        # 盤面上の駒をエンコード
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                stack = board.get_stack((row, col))
                if stack is None:
                    continue
                    
                for level in range(min(len(stack), self.MAX_STACK)):
                    piece = stack.get_piece_at_level(level)
                    if piece is None:
                        continue
                    
                    piece_idx = PIECE_TYPE_TO_INDEX.get(piece.piece_type)
                    if piece_idx is None:
                        continue
                    
                    # 自分の駒か相手の駒か
                    is_opponent = (piece.owner != current_player)
                    channel = self._get_piece_channel(piece_idx, level, is_opponent)
                    state[channel, row, col] = 1.0
        
        # 持ち駒をエンコード（3カテゴリに集約）
        my_hand_counts = self._categorize_hand(my_hand)
        opponent_hand_counts = self._categorize_hand(opponent_hand)
        
        # 自分の持ち駒（ch 84-86）
        for cat_idx, count in enumerate(my_hand_counts):
            # 正規化: 最大10枚程度なので10で割る
            normalized = min(count / 10.0, 1.0)
            state[84 + cat_idx, :, :] = normalized
        
        # 相手の持ち駒（ch 87-89）
        for cat_idx, count in enumerate(opponent_hand_counts):
            normalized = min(count / 10.0, 1.0)
            state[87 + cat_idx, :, :] = normalized
        
        # 手番（ch 90）- 常に自分の番なので1
        state[90, :, :] = 1.0
        
        return state
    
    def _get_piece_channel(self, piece_type_idx: int, stack_level: int, 
                           is_opponent: bool) -> int:
        """
        駒のチャンネルインデックスを計算
        
        Args:
            piece_type_idx: 駒種類のインデックス (0-13)
            stack_level: スタックレベル (0, 1, 2)
            is_opponent: 相手の駒かどうか
        
        Returns:
            チャンネルインデックス
        """
        base = 42 if is_opponent else 0
        return base + piece_type_idx * 3 + stack_level
    
    def _categorize_hand(self, hand: Dict[PieceType, int]) -> List[int]:
        """
        持ち駒を3カテゴリに分類
        
        Returns:
            [大駒の数, 中駒の数, 小駒の数]
        """
        if hand is None:
            return [0, 0, 0]
        
        large = sum(hand.get(pt, 0) for pt in HAND_CATEGORY_LARGE)
        medium = sum(hand.get(pt, 0) for pt in HAND_CATEGORY_MEDIUM)
        small = sum(hand.get(pt, 0) for pt in HAND_CATEGORY_SMALL)
        
        return [large, medium, small]
    
    def encode_batch(
        self,
        boards: List[Board],
        players: List[Player],
        my_hands: List[Dict[PieceType, int]],
        opponent_hands: List[Dict[PieceType, int]]
    ) -> np.ndarray:
        """
        複数の盤面をバッチエンコード
        
        Returns:
            np.ndarray: shape (batch, 91, 9, 9)
        """
        batch = [
            self.encode(board, player, my_hand, opp_hand)
            for board, player, my_hand, opp_hand 
            in zip(boards, players, my_hands, opponent_hands)
        ]
        return np.stack(batch, axis=0)


class ActionEncoder:
    """
    行動とインデックスを相互変換するクラス
    
    総行動数: 7,695
    
    行動インデックスの構成:
    - 0-6560:      移動/スタック/取る（81 × 81 = 6,561通り）
    - 6561-7694:   持ち駒を打つ（81 × 14 = 1,134通り）
    """
    
    ACTION_SIZE = 7695
    BOARD_SIZE = BOARD_SIZE  # 9
    NUM_SQUARES = BOARD_SIZE * BOARD_SIZE  # 81
    MOVE_ACTIONS = NUM_SQUARES * NUM_SQUARES  # 6561
    DROP_ACTIONS = NUM_SQUARES * NUM_PIECE_TYPES  # 1134
    
    def __init__(self):
        pass
    
    def encode_move(self, move: Move) -> int:
        """
        Moveオブジェクトを行動インデックスに変換
        
        Args:
            move: Moveオブジェクト
        
        Returns:
            行動インデックス (0 <= index < 7695)
        """
        if move.move_type == MoveType.DROP:
            # 持ち駒を打つ場合
            to_row, to_col = move.to_pos
            piece_idx = PIECE_TYPE_TO_INDEX.get(move.piece_type, 0)
            to_square = to_row * self.BOARD_SIZE + to_col
            return self.MOVE_ACTIONS + to_square * NUM_PIECE_TYPES + piece_idx
        else:
            # 移動/スタック/取る場合
            from_row, from_col = move.from_pos
            to_row, to_col = move.to_pos
            from_square = from_row * self.BOARD_SIZE + from_col
            to_square = to_row * self.BOARD_SIZE + to_col
            return from_square * self.NUM_SQUARES + to_square
    
    def decode_action(self, action_idx: int, player: Player,
                      board: Board = None) -> Move:
        """
        行動インデックスをMoveオブジェクトに変換
        
        Args:
            action_idx: 行動インデックス
            player: プレイヤー
            board: 盤面（MoveTypeの判定に使用、Noneの場合はMOVEとする）
        
        Returns:
            Moveオブジェクト
        """
        if action_idx >= self.MOVE_ACTIONS:
            # DROP
            drop_idx = action_idx - self.MOVE_ACTIONS
            to_square = drop_idx // NUM_PIECE_TYPES
            piece_idx = drop_idx % NUM_PIECE_TYPES
            
            to_row = to_square // self.BOARD_SIZE
            to_col = to_square % self.BOARD_SIZE
            piece_type = PIECE_TYPES[piece_idx]
            
            return Move(
                move_type=MoveType.DROP,
                from_pos=None,
                to_pos=(to_row, to_col),
                piece_type=piece_type,
                player=player
            )
        else:
            # MOVE/STACK/CAPTURE
            from_square = action_idx // self.NUM_SQUARES
            to_square = action_idx % self.NUM_SQUARES
            
            from_row = from_square // self.BOARD_SIZE
            from_col = from_square % self.BOARD_SIZE
            to_row = to_square // self.BOARD_SIZE
            to_col = to_square % self.BOARD_SIZE
            
            # MoveTypeの判定
            move_type = MoveType.NORMAL
            if board is not None:
                to_stack = board.get_stack((to_row, to_col))
                if to_stack and len(to_stack) > 0:
                    top_piece = to_stack.get_top_piece()
                    if top_piece:
                        if top_piece.owner == player:
                            move_type = MoveType.STACK
                        else:
                            move_type = MoveType.CAPTURE
            
            return Move(
                move_type=move_type,
                from_pos=(from_row, from_col),
                to_pos=(to_row, to_col),
                player=player
            )
    
    def get_legal_mask(
        self,
        board: Board,
        player: Player,
        hand_pieces: Dict[PieceType, int],
        legal_moves: List[Move] = None
    ) -> np.ndarray:
        """
        合法手のマスクを生成
        
        Args:
            board: 盤面
            player: プレイヤー
            hand_pieces: 持ち駒
            legal_moves: 合法手のリスト（Noneの場合は自動取得）
        
        Returns:
            np.ndarray: shape (7695,), 合法手は1、非合法手は0
        """
        from ..engine.rules import Rules
        
        mask = np.zeros(self.ACTION_SIZE, dtype=np.float32)
        
        if legal_moves is None:
            legal_moves = Rules.get_legal_moves(board, player, hand_pieces)
        
        for move in legal_moves:
            action_idx = self.encode_move(move)
            if 0 <= action_idx < self.ACTION_SIZE:
                mask[action_idx] = 1.0
        
        return mask
    
    def moves_to_policy(
        self,
        moves: List[Move],
        visit_counts: Dict[int, int]
    ) -> np.ndarray:
        """
        訪問回数から方策分布を生成
        
        Args:
            moves: 手のリスト
            visit_counts: 各手の訪問回数 {action_idx: count}
        
        Returns:
            np.ndarray: shape (7695,), 確率分布
        """
        policy = np.zeros(self.ACTION_SIZE, dtype=np.float32)
        
        total_visits = sum(visit_counts.values())
        if total_visits == 0:
            return policy
        
        for action_idx, count in visit_counts.items():
            if 0 <= action_idx < self.ACTION_SIZE:
                policy[action_idx] = count / total_visits
        
        return policy


# モジュールレベルのインスタンス（便利用）
state_encoder = StateEncoder()
action_encoder = ActionEncoder()


if __name__ == "__main__":
    # テスト
    from ..engine.initial_setup import load_initial_board, get_initial_hand_pieces
    
    print("=== StateEncoder Test ===")
    board = load_initial_board()
    player = Player.BLACK
    my_hand = get_initial_hand_pieces(player)
    opponent_hand = get_initial_hand_pieces(player.opponent)
    
    encoder = StateEncoder()
    state = encoder.encode(board, player, my_hand, opponent_hand)
    print(f"State shape: {state.shape}")
    print(f"Non-zero channels: {np.sum(np.any(state != 0, axis=(1, 2)))}")
    
    print("\n=== ActionEncoder Test ===")
    action_enc = ActionEncoder()
    
    # 移動手のテスト
    move1 = Move(MoveType.MOVE, from_pos=(6, 0), to_pos=(5, 0), player=Player.BLACK)
    idx1 = action_enc.encode_move(move1)
    decoded1 = action_enc.decode_action(idx1, Player.BLACK)
    print(f"Move: {move1.from_pos} -> {move1.to_pos}")
    print(f"  Index: {idx1}")
    print(f"  Decoded: {decoded1.from_pos} -> {decoded1.to_pos}")
    
    # DROP手のテスト
    move2 = Move(MoveType.DROP, from_pos=None, to_pos=(5, 4), 
                 piece_type=PieceType.SHO, player=Player.BLACK)
    idx2 = action_enc.encode_move(move2)
    decoded2 = action_enc.decode_action(idx2, Player.BLACK)
    print(f"\nDrop: {move2.piece_type} -> {move2.to_pos}")
    print(f"  Index: {idx2}")
    print(f"  Decoded: {decoded2.piece_type} -> {decoded2.to_pos}")
    
    # 合法手マスクのテスト
    from ..engine.rules import Rules
    legal_moves = Rules.get_legal_moves(board, player, my_hand)
    mask = action_enc.get_legal_mask(board, player, my_hand, legal_moves)
    print(f"\nLegal mask: {np.sum(mask)} legal moves out of {action_enc.ACTION_SIZE}")
