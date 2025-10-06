"""
軍儀AI用のニューラルネットワークモデル
AlphaZero型の方策+価値ネットワーク
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """ResNetの残差ブロック"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class GungiNetwork(nn.Module):
    """
    軍儀用のニューラルネットワーク
    
    入力: 盤面の状態 (C, 9, 9)
    出力:
        - policy: 各手の確率分布
        - value: 局面の評価値 (-1 ~ 1)
    """
    
    def __init__(
        self,
        input_channels: int = 64,  # 駒の種類 × プレイヤー × スタックレベル
        num_res_blocks: int = 8,
        num_filters: int = 128,
        board_size: int = 9,
        num_actions: int = 729  # 9x9 x 9 (from → to の組み合わせ)
    ):
        super().__init__()
        
        self.board_size = board_size
        self.num_actions = num_actions
        
        # 入力層
        self.input_conv = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm2d(num_filters)
        
        # ResNetブロック
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        # Policy Head（方策）
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * board_size * board_size, num_actions)
        
        # Value Head（価値）
        self.value_conv = nn.Conv2d(num_filters, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        順伝播
        
        Args:
            x: 盤面の状態テンソル (batch, channels, 9, 9)
        
        Returns:
            policy: 方策の確率分布 (batch, num_actions)
            value: 価値の評価 (batch, 1)
        """
        # 入力層
        x = F.relu(self.input_bn(self.input_conv(x)))
        
        # ResNetブロック
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Policy Head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # Value Head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value


def encode_board_state(board, player) -> torch.Tensor:
    """
    盤面の状態をニューラルネットワークの入力テンソルに変換
    
    Args:
        board: Boardオブジェクト
        player: 現在のプレイヤー
    
    Returns:
        テンソル (1, channels, 9, 9)
    """
    # 各駒の種類 × プレイヤー × スタックレベルをチャンネルとして表現
    # 例: 14種類 × 2プレイヤー × 3レベル = 84チャンネル
    # 簡略化のため、ここでは基本的な実装のみ
    
    from ..engine import PieceType, Player, BOARD_SIZE
    
    num_piece_types = len(PieceType)
    num_players = 2
    max_stack = 3
    channels = num_piece_types * num_players * max_stack + 1  # +1は手番情報
    
    state = torch.zeros(1, channels, BOARD_SIZE, BOARD_SIZE)
    
    channel_idx = 0
    
    # 各駒タイプ × プレイヤー × スタックレベルをエンコード
    for piece_type in PieceType:
        for owner in [Player.BLACK, Player.WHITE]:
            for stack_level in range(max_stack):
                for row in range(BOARD_SIZE):
                    for col in range(BOARD_SIZE):
                        stack = board.get_stack((row, col))
                        if len(stack) > stack_level:
                            piece = stack.get_piece_at_level(stack_level)
                            if piece and piece.piece_type == piece_type and piece.owner == owner:
                                state[0, channel_idx, row, col] = 1.0
                channel_idx += 1
    
    # 手番情報（全マスに1または0）
    if player == Player.BLACK:
        state[0, channel_idx, :, :] = 1.0
    
    return state


def decode_action(action_idx: int, board_size: int = 9) -> Tuple[int, int, int, int]:
    """
    アクションインデックスを (from_row, from_col, to_row, to_col) に変換
    """
    # 簡易的な実装: from位置とto位置の組み合わせ
    from_pos = action_idx // (board_size * board_size)
    to_idx = action_idx % (board_size * board_size)
    
    from_row = from_pos // board_size
    from_col = from_pos % board_size
    to_row = to_idx // board_size
    to_col = to_idx % board_size
    
    return from_row, from_col, to_row, to_col


def create_model(device: str = 'cpu') -> GungiNetwork:
    """
    モデルを作成して初期化
    
    Args:
        device: 'cpu' or 'cuda'
    
    Returns:
        初期化されたGungiNetwork
    """
    model = GungiNetwork(
        input_channels=85,  # 14 types × 2 players × 3 levels + 1 turn
        num_res_blocks=5,   # Phase 1は軽量版
        num_filters=128,
        board_size=9,
        num_actions=729
    )
    
    model = model.to(device)
    return model


if __name__ == "__main__":
    # テスト
    model = create_model('cpu')
    
    # ダミー入力
    dummy_input = torch.randn(1, 85, 9, 9)
    policy, value = model(dummy_input)
    
    print(f"Policy shape: {policy.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Value: {value.item()}")
