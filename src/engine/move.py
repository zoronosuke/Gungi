"""
軍儀の手（Move）を表現するモジュール
"""

from enum import Enum, auto
from typing import Tuple, Optional
from .piece import PieceType, Player


class MoveType(Enum):
    """手の種類"""
    NORMAL = auto()    # 通常の移動
    CAPTURE = auto()   # 駒を取る
    STACK = auto()     # 駒を重ねる（ツケ）
    DROP = auto()      # 持ち駒を打つ（新）
    SETUP = auto()     # 初期配置


class Move:
    """軍儀の一手を表すクラス"""
    
    def __init__(
        self,
        move_type: MoveType,
        from_pos: Optional[Tuple[int, int]] = None,
        to_pos: Optional[Tuple[int, int]] = None,
        piece_type: Optional[PieceType] = None,
        player: Optional[Player] = None
    ):
        self.move_type = move_type
        self.from_pos = from_pos  # 移動元（Noneの場合は持ち駒）
        self.to_pos = to_pos      # 移動先
        self.piece_type = piece_type  # 駒の種類
        self.player = player      # プレイヤー

    def __str__(self):
        if self.move_type == MoveType.DROP or self.move_type == MoveType.SETUP:
            return f"{self.player.name} {self.piece_type.name} -> {self.to_pos}"
        else:
            return f"{self.player.name} {self.from_pos} -> {self.to_pos} ({self.move_type.name})"

    def __repr__(self):
        return (
            f"Move(type={self.move_type.name}, "
            f"from={self.from_pos}, to={self.to_pos}, "
            f"piece={self.piece_type.name if self.piece_type else None}, "
            f"player={self.player.name if self.player else None})"
        )

    def to_dict(self) -> dict:
        """手を辞書形式に変換（API用）"""
        return {
            "type": self.move_type.name,
            "from": self.from_pos,
            "to": self.to_pos,
            "piece_type": self.piece_type.name if self.piece_type else None,
            "player": self.player.name if self.player else None
        }

    @staticmethod
    def from_dict(data: dict) -> 'Move':
        """辞書形式から手を復元（API用）"""
        move_type = MoveType[data["type"]]
        from_pos = tuple(data["from"]) if data.get("from") else None
        to_pos = tuple(data["to"]) if data.get("to") else None
        piece_type = PieceType[data["piece_type"]] if data.get("piece_type") else None
        player = Player[data["player"]] if data.get("player") else None
        
        return Move(
            move_type=move_type,
            from_pos=from_pos,
            to_pos=to_pos,
            piece_type=piece_type,
            player=player
        )

    @staticmethod
    def create_normal_move(
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        player: Player
    ) -> 'Move':
        """通常の移動手を作成"""
        return Move(
            move_type=MoveType.NORMAL,
            from_pos=from_pos,
            to_pos=to_pos,
            player=player
        )

    @staticmethod
    def create_capture_move(
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        player: Player
    ) -> 'Move':
        """駒を取る手を作成"""
        return Move(
            move_type=MoveType.CAPTURE,
            from_pos=from_pos,
            to_pos=to_pos,
            player=player
        )

    @staticmethod
    def create_stack_move(
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        player: Player
    ) -> 'Move':
        """駒を重ねる手を作成"""
        return Move(
            move_type=MoveType.STACK,
            from_pos=from_pos,
            to_pos=to_pos,
            player=player
        )

    @staticmethod
    def create_drop_move(
        to_pos: Tuple[int, int],
        piece_type: PieceType,
        player: Player
    ) -> 'Move':
        """持ち駒を打つ手を作成"""
        return Move(
            move_type=MoveType.DROP,
            to_pos=to_pos,
            piece_type=piece_type,
            player=player
        )

    @staticmethod
    def create_setup_move(
        to_pos: Tuple[int, int],
        piece_type: PieceType,
        player: Player
    ) -> 'Move':
        """初期配置の手を作成"""
        return Move(
            move_type=MoveType.SETUP,
            to_pos=to_pos,
            piece_type=piece_type,
            player=player
        )
