"""
軍儀の駒の種類と動きを定義するモジュール
"""

from enum import Enum, auto
from typing import List, Tuple, Optional


class Player(Enum):
    """プレイヤーの定義"""
    BLACK = 0  # 先手（黒）
    WHITE = 1  # 後手（白）

    @property
    def opponent(self):
        """相手プレイヤーを返す"""
        return Player.WHITE if self == Player.BLACK else Player.BLACK


class PieceType(Enum):
    """駒の種類"""
    SUI = auto()      # 帥 - 王将に相当
    DAI = auto()      # 大 - 大将（飛車+斜め1マス）
    CHUU = auto()     # 中 - 中将（角行+直線1マス）
    SHO = auto()      # 小 - 小将（金将）
    SAMURAI = auto()  # 侍 - 前方・斜め前・後方1マス
    HYO = auto()      # 兵 - 前後1マス
    UMA = auto()      # 馬 - 直線2マス
    SHINOBI = auto()  # 忍 - 斜め1～2マス
    YARI = auto()     # 槍 - 侍+前方2マス
    TORIDE = auto()   # 砦 - 防御駒
    YUMI = auto()     # 弓 - 飛び越え可能
    TSUTU = auto()    # 筒 - 飛び越え可能
    HOU = auto()      # 砲 - 飛び越え可能
    BOU = auto()      # 謀 - 斜め（寝返り能力）


# 各プレイヤーが持つ駒の初期数
PIECE_COUNTS = {
    PieceType.SUI: 1,
    PieceType.DAI: 1,
    PieceType.CHUU: 1,
    PieceType.SHO: 2,
    PieceType.SAMURAI: 2,
    PieceType.HYO: 4,
    PieceType.UMA: 2,
    PieceType.SHINOBI: 2,
    PieceType.YARI: 3,
    PieceType.TORIDE: 2,
    PieceType.YUMI: 2,
    PieceType.TSUTU: 1,
    PieceType.HOU: 1,
    PieceType.BOU: 1
}

# 駒の表示名（漢字）
PIECE_NAMES = {
    PieceType.SUI: "帥",
    PieceType.DAI: "大",
    PieceType.CHUU: "中",
    PieceType.SHO: "小",
    PieceType.SAMURAI: "侍",
    PieceType.HYO: "兵",
    PieceType.UMA: "馬",
    PieceType.SHINOBI: "忍",
    PieceType.YARI: "槍",
    PieceType.TORIDE: "砦",
    PieceType.YUMI: "弓",
    PieceType.TSUTU: "筒",
    PieceType.HOU: "砲",
    PieceType.BOU: "謀"
}

# 駒の動きパターン定義（軍儀本家ルールに準拠）
# [row, col]で表現: 正の値は前方向（黒プレイヤー基準で上方向）
PIECE_MOVE_PATTERNS = {
    PieceType.SUI: {  # 帥
        'base': {
            'moves': [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]],
            'maxSteps': 1,
            'canJump': False
        },
        'evolved': {
            'moves': [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1],
                     [-2, -2], [-2, 0], [-2, 2], [0, -2], [0, 2], [2, -2], [2, 0], [2, 2]],
            'maxSteps': 1,
            'canJump': False
        },
        'mastered': {
            'moves': [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1],
                     [-2, -2], [-2, 0], [-2, 2], [0, -2], [0, 2], [2, -2], [2, 0], [2, 2],
                     [-3, -3], [-3, 0], [-3, 3], [0, -3], [0, 3], [3, -3], [3, 0], [3, 3]],
            'maxSteps': 1,
            'canJump': False
        }
    },
    PieceType.DAI: {  # 大
        'base': {
            'moves': [[-1, -1], [-1, 1], [1, -1], [1, 1]],
            'maxSteps': 1,
            'canJump': False
        },
        'evolved': {
            'moves': [[-1, -1], [-1, 1], [1, -1], [1, 1],
                     [-2, -2], [-2, 2], [2, -2], [2, 2]],
            'maxSteps': 1,
            'canJump': False
        },
        'mastered': {
            'moves': [[-1, -1], [-1, 1], [1, -1], [1, 1],
                     [-2, -2], [-2, 2], [2, -2], [2, 2],
                     [-3, -3], [-3, 3], [3, -3], [3, 3]],
            'maxSteps': 1,
            'canJump': False
        }
    },
    PieceType.CHUU: {  # 中
        'base': {
            'moves': [[-1, 0], [0, -1], [0, 1], [1, 0]],
            'maxSteps': 1,
            'canJump': False
        },
        'evolved': {
            'moves': [[-1, 0], [0, -1], [0, 1], [1, 0],
                     [-2, 0], [0, -2], [0, 2], [2, 0]],
            'maxSteps': 1,
            'canJump': False
        },
        'mastered': {
            'moves': [[-1, 0], [0, -1], [0, 1], [1, 0],
                     [-2, 0], [0, -2], [0, 2], [2, 0],
                     [-3, 0], [0, -3], [0, 3], [3, 0]],
            'maxSteps': 1,
            'canJump': False
        }
    },
    PieceType.SHO: {  # 小
        'base': {
            'moves': [[-1, 0], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]],
            'maxSteps': 1,
            'canJump': False
        },
        'evolved': {
            'moves': [[-1, 0], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1],
                     [2, 2], [2, 0], [2, -2], [0, 2], [0, -2], [-2, 0]],
            'maxSteps': 1,
            'canJump': False
        },
        'mastered': {
            'moves': [[-1, 0], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1],
                     [2, 2], [2, 0], [2, -2], [0, 2], [0, -2], [-2, 0],
                     [3, 3], [3, 0], [3, -3], [0, 3], [0, -3], [-3, 0]],
            'maxSteps': 1,
            'canJump': False
        }
    },
    PieceType.SAMURAI: {  # 侍
        'base': {
            'moves': [[1, 1], [1, 0], [1, -1], [-1, 0]],
            'maxSteps': 1,
            'canJump': False
        },
        'evolved': {
            'moves': [[1, 1], [1, 0], [1, -1], [-1, 0],
                     [2, 2], [2, 0], [2, -2], [-2, 0]],
            'maxSteps': 1,
            'canJump': False
        },
        'mastered': {
            'moves': [[1, 1], [1, 0], [1, -1], [-1, 0],
                     [2, 2], [2, 0], [2, -2], [-2, 0],
                     [3, 3], [3, 0], [3, -3], [-3, 0]],
            'maxSteps': 1,
            'canJump': False
        }
    },
    PieceType.HYO: {  # 兵
        'base': {
            'moves': [[1, 0], [-1, 0]],
            'maxSteps': 1,
            'canJump': False
        },
        'evolved': {
            'moves': [[1, 0], [-1, 0], [2, 0], [-2, 0]],
            'maxSteps': 1,
            'canJump': False
        },
        'mastered': {
            'moves': [[1, 0], [-1, 0], [2, 0], [-2, 0], [3, 0], [-3, 0]],
            'maxSteps': 1,
            'canJump': True
        }
    },
    PieceType.UMA: {  # 馬 - 上下2マス、左右1マス
        'base': {
            'moves': [[1, 0], [2, 0], [-1, 0], [-2, 0], [0, 1], [0, -1]],
            'maxSteps': 1,
            'canJump': False
        },
        'evolved': {
            'moves': [[1, 0], [2, 0], [3, 0], [-1, 0], [-2, 0], [-3, 0], [0, 1], [0, 2], [0, -1], [0, -2]],
            'maxSteps': 1,
            'canJump': False
        },
        'mastered': {
            'moves': [[1, 0], [2, 0], [3, 0], [4, 0], [-1, 0], [-2, 0], [-3, 0], [-4, 0], [0, 1], [0, 2], [0, 3], [0, -1], [0, -2], [0, -3]],
            'maxSteps': 1,
            'canJump': True
        }
    },
    PieceType.SHINOBI: {  # 忍
        'base': {
            'moves': [[2, 2], [2, -2], [1, 1], [1, -1], [-1, 1], [-1, -1], [-2, 2], [-2, -2]],
            'maxSteps': 1,
            'canJump': False
        },
        'evolved': {
            'moves': [[2, 2], [2, -2], [1, 1], [1, -1], [-1, 1], [-1, -1], [-2, 2], [-2, -2],
                     [3, 3], [3, -3], [-3, 3], [-3, -3]],
            'maxSteps': 1,
            'canJump': False
        },
        'mastered': {
            'moves': [[2, 2], [2, -2], [1, 1], [1, -1], [-1, 1], [-1, -1], [-2, 2], [-2, -2],
                     [3, 3], [3, -3], [-3, 3], [-3, -3],
                     [4, 4], [4, -4], [-4, 4], [-4, -4]],
            'maxSteps': 1,
            'canJump': True
        }
    },
    PieceType.YARI: {  # 槍
        'base': {
            'moves': [[2, 0], [1, 1], [1, 0], [1, -1], [-1, 0]],
            'maxSteps': 1,
            'canJump': False
        },
        'evolved': {
            'moves': [[2, 0], [1, 1], [1, 0], [1, -1], [-1, 0],
                     [3, 0], [2, 2], [2, -2], [-2, 0]],
            'maxSteps': 1,
            'canJump': False
        },
        'mastered': {
            'moves': [[2, 0], [1, 1], [1, 0], [1, -1], [-1, 0],
                     [3, 0], [2, 2], [2, -2], [-2, 0],
                     [4, 0], [3, 3], [3, -3], [-3, 0]],
            'maxSteps': 1,
            'canJump': False
        }
    },
    PieceType.TORIDE: {  # 砦
        'base': {
            'moves': [[1, 0], [0, 1], [0, -1], [-1, 1], [-1, -1]],
            'maxSteps': 1,
            'canJump': False
        },
        'evolved': {
            'moves': [[1, 0], [0, 1], [0, -1], [-1, 1], [-1, -1],
                     [2, 0], [0, 2], [0, -2], [-2, 2], [-2, -2]],
            'maxSteps': 1,
            'canJump': False
        },
        'mastered': {
            'moves': [[1, 0], [0, 1], [0, -1], [-1, 1], [-1, -1],
                     [2, 0], [0, 2], [0, -2], [-2, 2], [-2, -2],
                     [3, 0], [0, 3], [0, -3], [-3, 3], [-3, -3]],
            'maxSteps': 1,
            'canJump': True
        }
    },
    PieceType.YUMI: {  # 弓
        'base': {
            'moves': [[2, -1], [2, 0], [2, 1], [-1, 0]],
            'maxSteps': 1,
            'canJump': False
        },
        'evolved': {
            'moves': [[2, -1], [2, 0], [2, 1], [-1, 0],
                     [3, -1], [3, 0], [3, 1], [-2, 0]],
            'maxSteps': 1,
            'canJump': False
        },
        'mastered': {
            'moves': [[2, -1], [2, 0], [2, 1], [-1, 0],
                     [3, -1], [3, 0], [3, 1], [-2, 0],
                     [4, -1], [4, 0], [4, 1], [-3, 0]],
            'maxSteps': 1,
            'canJump': True
        }
    },
    PieceType.TSUTU: {  # 筒
        'base': {
            'moves': [[2, 0], [-1, 1], [-1, -1]],
            'maxSteps': 1,
            'canJump': False
        },
        'evolved': {
            'moves': [[2, 0], [-1, 1], [-1, -1],
                     [-2, 2], [-2, -2]],
            'maxSteps': 1,
            'canJump': False
        },
        'mastered': {
            'moves': [[2, 0], [-1, 1], [-1, -1],
                     [-2, 2], [-2, -2],
                     [4, 0], [-3, 3], [-3, -3]],
            'maxSteps': 1,
            'canJump': True
        }
    },
    PieceType.HOU: {  # 砲
        'base': {
            'moves': [[3, 0], [0, 1], [0, -1], [-1, 0]],
            'maxSteps': 1,
            'canJump': False
        },
        'evolved': {
            'moves': [[3, 0], [0, 1], [0, -1], [-1, 0],
                     [4, 0], [0, 2], [0, -2], [-2, 0]],
            'maxSteps': 1,
            'canJump': False
        },
        'mastered': {
            'moves': [[3, 0], [0, 1], [0, -1], [-1, 0],
                     [4, 0], [0, 2], [0, -2], [-2, 0],
                     [5, 0], [0, 3], [0, -3], [-3, 0]],
            'maxSteps': 1,
            'canJump': True
        }
    },
    PieceType.BOU: {  # 謀
        'base': {
            'moves': [[1, 1], [1, -1], [-1, 0]],
            'maxSteps': 1,
            'canJump': False
        },
        'evolved': {
            'moves': [[1, 1], [1, -1], [-1, 0],
                     [2, 2], [2, -2], [-2, 0]],
            'maxSteps': 1,
            'canJump': False
        },
        'mastered': {
            'moves': [[1, 1], [1, -1], [-1, 0],
                     [2, 2], [2, -2], [-2, 0],
                     [3, 3], [3, -3], [-3, 0]],
            'maxSteps': 1,
            'canJump': True
        }
    }
}


class Piece:
    """軍儀の駒を表すクラス"""
    
    def __init__(self, piece_type: PieceType, owner: Player):
        self.piece_type = piece_type
        self.owner = owner

    def __str__(self):
        """駒の文字列表現（例: 'b帥', 'w大'）"""
        prefix = 'b' if self.owner == Player.BLACK else 'w'
        return f"{prefix}{PIECE_NAMES[self.piece_type]}"

    def __repr__(self):
        return f"Piece({self.piece_type.name}, {self.owner.name})"

    def get_move_pattern(self, stack_level: int = 1) -> dict:
        """
        この駒の移動パターンを返す（スタックレベルに応じて変化）
        stack_level: 駒が積まれているレベル（1-3）
        返り値: {'moves': list, 'maxSteps': int, 'canJump': bool}
        """
        patterns = PIECE_MOVE_PATTERNS.get(self.piece_type)
        if not patterns:
            return {'moves': [], 'maxSteps': 1, 'canJump': False}
        
        # スタックレベルに応じて動きを取得
        if stack_level == 1:
            return patterns['base']
        elif stack_level == 2 and 'evolved' in patterns:
            return patterns['evolved']
        elif stack_level == 3 and 'mastered' in patterns:
            return patterns['mastered']
        else:
            return patterns['base']

    def can_stack_on_other(self) -> bool:
        """この駒が他の駒の上に乗れるか"""
        # 砦は他の駒の上に乗れない
        return self.piece_type != PieceType.TORIDE

    def can_be_stacked_on(self) -> bool:
        """この駒の上に他の駒を乗せられるか"""
        # 帥の上には乗せられない
        return self.piece_type != PieceType.SUI
