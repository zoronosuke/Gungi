"""
初期盤面の設定とユーティリティ
"""

from typing import Dict, List, Tuple
from .board import Board
from .piece import Piece, Player, PieceType, PIECE_NAMES


def load_initial_board() -> Board:
    """
    公式の初期盤面を読み込む
    初期盤面.txtに記載されている配置を使用
    """
    board = Board()
    
    # 後手（白）の配置（上側: 1-3段目）
    white_setup = [
        # 1段目
        (0, 3, PieceType.CHUU),   # 中
        (0, 4, PieceType.SUI),    # 帥
        (0, 5, PieceType.DAI),    # 大
        # 2段目
        (1, 1, PieceType.SHINOBI),
        (1, 2, PieceType.YUMI),
        (1, 4, PieceType.YARI),
        (1, 6, PieceType.YUMI),
        (1, 7, PieceType.UMA),
        # 3段目
        (2, 0, PieceType.HYO),
        (2, 2, PieceType.TORIDE),
        (2, 3, PieceType.SAMURAI),
        (2, 4, PieceType.HYO),
        (2, 5, PieceType.SAMURAI),
        (2, 6, PieceType.TORIDE),
        (2, 8, PieceType.HYO),
    ]
    
    # 先手（黒）の配置（下側: 7-9段目、インデックスは6-8）
    black_setup = [
        # 7段目（インデックス6）
        (6, 0, PieceType.HYO),
        (6, 2, PieceType.TORIDE),
        (6, 3, PieceType.SAMURAI),
        (6, 4, PieceType.HYO),
        (6, 5, PieceType.SAMURAI),
        (6, 6, PieceType.TORIDE),
        (6, 8, PieceType.HYO),
        # 8段目（インデックス7）
        (7, 1, PieceType.UMA),
        (7, 2, PieceType.YUMI),
        (7, 4, PieceType.YARI),
        (7, 6, PieceType.YUMI),
        (7, 7, PieceType.SHINOBI),
        # 9段目（インデックス8）
        (8, 3, PieceType.DAI),    # 大
        (8, 4, PieceType.SUI),    # 帥
        (8, 5, PieceType.CHUU),   # 中
    ]
    
    # 白の駒を配置
    for row, col, piece_type in white_setup:
        piece = Piece(piece_type, Player.WHITE)
        board.add_piece((row, col), piece)
    
    # 黒の駒を配置
    for row, col, piece_type in black_setup:
        piece = Piece(piece_type, Player.BLACK)
        board.add_piece((row, col), piece)
    
    return board


def get_initial_hand_pieces(player: Player) -> Dict[PieceType, int]:
    """
    初期盤面で使用していない持ち駒を返す
    """
    # 公式初期盤面では以下の駒が持ち駒として残る
    hand_pieces = {
        PieceType.SHO: 2,      # 小将 x2
        PieceType.YARI: 2,     # 槍 x2（3本中1本は使用）
        PieceType.UMA: 1,      # 馬 x1（2頭中1頭は使用）
        PieceType.SHINOBI: 1,  # 忍 x1（2人中1人は使用）
        PieceType.HYO: 1,      # 兵 x1（4人中3人は使用）
        PieceType.HOU: 1,      # 砲 x1
        PieceType.TSUTU: 1,    # 筒 x1
        PieceType.BOU: 1,      # 謀 x1
    }
    
    return hand_pieces


def parse_piece_from_text(text: str) -> Tuple[PieceType, Player]:
    """
    テキストから駒の種類とプレイヤーを解析
    例: 'b帥' -> (PieceType.SUI, Player.BLACK)
    """
    if len(text) < 2:
        raise ValueError(f"Invalid piece text: {text}")
    
    player_char = text[0]
    piece_char = text[1]
    
    # プレイヤーの判定
    if player_char == 'b':
        player = Player.BLACK
    elif player_char == 'w':
        player = Player.WHITE
    else:
        raise ValueError(f"Invalid player character: {player_char}")
    
    # 駒の種類を判定
    piece_type = None
    for pt, name in PIECE_NAMES.items():
        if name == piece_char:
            piece_type = pt
            break
    
    if piece_type is None:
        raise ValueError(f"Invalid piece character: {piece_char}")
    
    return piece_type, player


def format_position(row: int, col: int) -> str:
    """
    盤面の位置を文字列に変換
    例: (0, 0) -> "a1", (8, 8) -> "i9"
    """
    col_char = chr(ord('a') + col)
    row_num = row + 1
    return f"{col_char}{row_num}"


def parse_position(pos_str: str) -> Tuple[int, int]:
    """
    文字列を盤面の位置に変換
    例: "a1" -> (0, 0), "i9" -> (8, 8)
    """
    if len(pos_str) < 2:
        raise ValueError(f"Invalid position string: {pos_str}")
    
    col_char = pos_str[0].lower()
    row_num = int(pos_str[1:])
    
    col = ord(col_char) - ord('a')
    row = row_num - 1
    
    return row, col
