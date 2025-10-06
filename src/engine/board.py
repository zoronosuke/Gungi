"""
軍儀の盤面を管理するモジュール
"""

from typing import List, Tuple, Optional, Dict
from .piece import Piece, Player, PieceType

# 盤面サイズ
BOARD_SIZE = 9
# 最大スタック高さ
MAX_STACK_HEIGHT = 3


class Stack:
    """一つのマスに積まれた駒を管理するクラス"""
    
    def __init__(self):
        self.pieces: List[Piece] = []  # 下から上への駒のリスト

    def __len__(self):
        return len(self.pieces)

    def is_empty(self) -> bool:
        return len(self.pieces) == 0

    def add_piece(self, piece: Piece) -> bool:
        """
        スタックの一番上に駒を追加
        返り値: 成功したらTrue、失敗したらFalse
        """
        if len(self.pieces) >= MAX_STACK_HEIGHT:
            return False
        self.pieces.append(piece)
        return True

    def remove_top_piece(self) -> Optional[Piece]:
        """スタックから一番上の駒を取り除いて返す"""
        if self.is_empty():
            return None
        return self.pieces.pop()

    def get_top_piece(self) -> Optional[Piece]:
        """一番上の駒を取得（削除はしない）"""
        if self.is_empty():
            return None
        return self.pieces[-1]

    def get_piece_at_level(self, level: int) -> Optional[Piece]:
        """
        指定されたレベルの駒を取得
        level: 0から始まるインデックス（0=最下層、len-1=最上層）
        """
        if level < 0 or level >= len(self.pieces):
            return None
        return self.pieces[level]

    def get_height(self) -> int:
        """スタックの高さを返す"""
        return len(self.pieces)
    
    def get_top_piece_stack_level(self) -> int:
        """
        一番上の駒のスタックレベルを返す（1, 2, 3）
        スタックが空の場合は0を返す
        """
        return len(self.pieces)

    def __str__(self):
        if self.is_empty():
            return "   "
        return "/".join(str(piece) for piece in self.pieces)

    def to_dict(self) -> List[dict]:
        """スタックを辞書形式に変換（API用）"""
        return [
            {
                "type": piece.piece_type.name,
                "owner": piece.owner.name
            }
            for piece in self.pieces
        ]


class Board:
    """軍儀のゲームボードを表すクラス"""
    
    def __init__(self):
        # 9x9の盤面を初期化
        self.stacks: List[List[Stack]] = [
            [Stack() for _ in range(BOARD_SIZE)]
            for _ in range(BOARD_SIZE)
        ]
        # 帥の位置を記録
        self.sui_positions: Dict[Player, Optional[Tuple[int, int]]] = {
            Player.BLACK: None,
            Player.WHITE: None
        }

    def get_stack(self, position: Tuple[int, int]) -> Stack:
        """指定位置のスタックを取得"""
        row, col = position
        if not self.is_valid_position(position):
            raise ValueError(f"Invalid position: {position}")
        return self.stacks[row][col]

    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        """位置が盤面内か確認"""
        row, col = position
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def is_occupied(self, position: Tuple[int, int]) -> bool:
        """指定位置に駒があるか確認"""
        return not self.get_stack(position).is_empty()

    def get_top_piece(self, position: Tuple[int, int]) -> Optional[Piece]:
        """指定位置の一番上の駒を取得"""
        return self.get_stack(position).get_top_piece()

    def get_top_piece_owner(self, position: Tuple[int, int]) -> Optional[Player]:
        """指定位置の一番上の駒の所有者を取得"""
        piece = self.get_top_piece(position)
        return piece.owner if piece else None

    def get_stack_height(self, position: Tuple[int, int]) -> int:
        """指定位置のスタック高さを取得"""
        return self.get_stack(position).get_height()
    
    def get_top_piece_stack_level(self, position: Tuple[int, int]) -> int:
        """
        指定位置の一番上の駒のスタックレベルを取得（1, 2, 3）
        駒がない場合は0を返す
        """
        return self.get_stack(position).get_top_piece_stack_level()

    def add_piece(self, position: Tuple[int, int], piece: Piece) -> bool:
        """指定位置に駒を追加"""
        if not self.is_valid_position(position):
            return False
        
        stack = self.get_stack(position)
        
        # スタック高さチェック
        if stack.get_height() >= MAX_STACK_HEIGHT:
            return False
        
        # 帥の上には乗せられない
        if stack.get_height() > 0:
            bottom_piece = stack.get_top_piece()
            if bottom_piece and not bottom_piece.can_be_stacked_on():
                return False
        
        # 砦は他の駒の上に乗れない
        if stack.get_height() > 0 and not piece.can_stack_on_other():
            return False
        
        success = stack.add_piece(piece)
        
        # 帥の位置を記録
        if success and piece.piece_type == PieceType.SUI:
            self.sui_positions[piece.owner] = position
        
        return success

    def remove_piece(self, position: Tuple[int, int]) -> Optional[Piece]:
        """指定位置から一番上の駒を削除"""
        piece = self.get_stack(position).remove_top_piece()
        
        # 帥が削除された場合、位置情報をクリア
        if piece and piece.piece_type == PieceType.SUI:
            if self.sui_positions[piece.owner] == position:
                self.sui_positions[piece.owner] = None
        
        return piece

    def move_piece(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """
        駒を移動する
        返り値: 成功したらTrue
        """
        if not self.is_valid_position(from_pos) or not self.is_valid_position(to_pos):
            return False
        
        if not self.is_occupied(from_pos):
            return False
        
        # 移動元から駒を取得
        piece = self.remove_piece(from_pos)
        if piece is None:
            return False
        
        # 移動先に追加（失敗した場合は元に戻す）
        if not self.add_piece(to_pos, piece):
            self.add_piece(from_pos, piece)
            return False
        
        return True

    def capture_piece(self, position: Tuple[int, int]) -> Optional[Piece]:
        """指定位置の駒を取る（盤面から除去）"""
        return self.remove_piece(position)

    def get_sui_position(self, player: Player) -> Optional[Tuple[int, int]]:
        """指定プレイヤーの帥の位置を取得"""
        return self.sui_positions[player]

    def is_in_territory(self, position: Tuple[int, int], player: Player) -> bool:
        """
        指定位置が指定プレイヤーの陣地内か確認
        黒の陣地: 6-8行（下3段）
        白の陣地: 0-2行（上3段）
        """
        row, col = position
        if player == Player.BLACK:
            return 6 <= row <= 8
        else:  # Player.WHITE
            return 0 <= row <= 2

    def copy(self) -> 'Board':
        """盤面のコピーを作成"""
        new_board = Board()
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                stack = self.stacks[row][col]
                for piece in stack.pieces:
                    new_piece = Piece(piece.piece_type, piece.owner)
                    new_board.add_piece((row, col), new_piece)
        return new_board

    def __str__(self):
        """盤面の文字列表現を返す"""
        cell_width = 6
        separator_length = BOARD_SIZE * (cell_width + 1) + 1

        result = []

        # 列インデックスヘッダー
        header = "   "
        for i in range(BOARD_SIZE):
            header += f"{i:^{cell_width}}|"
        result.append(header)

        # 区切り線
        result.append("  " + "-" * separator_length)

        for row in range(BOARD_SIZE):
            row_str = f"{row} |"
            for col in range(BOARD_SIZE):
                stack = self.stacks[row][col]
                if stack.is_empty():
                    row_str += " " * cell_width + "|"
                else:
                    piece_str = str(stack)
                    row_str += f"{piece_str:^{cell_width}}|"
            result.append(row_str)
            result.append("  " + "-" * separator_length)

        return "\n".join(result)

    def to_dict(self) -> dict:
        """盤面を辞書形式に変換（API用）"""
        board_data = []
        for row in range(BOARD_SIZE):
            row_data = []
            for col in range(BOARD_SIZE):
                stack = self.stacks[row][col]
                row_data.append(stack.to_dict())
            board_data.append(row_data)
        
        return {
            "board": board_data,
            "sui_positions": {
                "BLACK": self.sui_positions[Player.BLACK],
                "WHITE": self.sui_positions[Player.WHITE]
            }
        }

    def load_from_initial_setup(self, setup_text: str):
        """
        初期盤面テキストから盤面を読み込む
        初期盤面.txtの形式を想定
        """
        # この機能は後で実装
        pass
