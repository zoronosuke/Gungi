"""
pytest共通設定とフィクスチャ
"""

import pytest
import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def empty_board():
    """空の盤面を提供するフィクスチャ"""
    from src.engine import Board
    return Board()


@pytest.fixture
def initial_board():
    """公式初期配置の盤面を提供するフィクスチャ"""
    from src.engine.initial_setup import load_initial_board
    return load_initial_board()


@pytest.fixture
def initial_hand_pieces():
    """初期持ち駒を提供するフィクスチャ"""
    from src.engine.initial_setup import get_initial_hand_pieces
    from src.engine import Player
    return {
        Player.BLACK: get_initial_hand_pieces(Player.BLACK),
        Player.WHITE: get_initial_hand_pieces(Player.WHITE),
    }


@pytest.fixture
def black_player():
    """黒プレイヤーを提供するフィクスチャ"""
    from src.engine import Player
    return Player.BLACK


@pytest.fixture
def white_player():
    """白プレイヤーを提供するフィクスチャ"""
    from src.engine import Player
    return Player.WHITE
